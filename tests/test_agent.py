"""Agent 与工具层测试。

全部使用假 provider / 假检索器，不发任何网络请求。
重点覆盖几处**错了会静默出问题**的地方：

- 工具抛异常不能打断对话，必须转成错误结果回灌给模型；
- 达到步数上限的那一轮必须**不带工具**，否则模型可能永远不给答案；
- 带 tool_calls 的 assistant 消息必须回传 reasoning_content（DeepSeek 会 400）。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from bidding_qa_chatbot.agent import build_default_registry
from bidding_qa_chatbot.agent.agent import (
    AgentDone,
    RagAgent,
    ToolCallNotice,
    ToolResultNotice,
)
from bidding_qa_chatbot.agent.base import ToolRegistry, ToolResult, ToolSpec
from bidding_qa_chatbot.agent.rag_tool import RagSearchTool
from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.llm.base import (
    ChatMessage,
    LLMRequest,
    StreamEnd,
    StreamEvent,
    ToolCall,
)
from bidding_qa_chatbot.retrieval.service import RetrievalResult, RetrievedSource


# ---------------------------------------------------------------- 假件
class FakeTool:
    def __init__(
        self, name: str, result: ToolResult | None = None, raises: Exception | None = None
    ):
        self.spec = ToolSpec(
            name=name,
            description=f"测试工具 {name}",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        self._result = result
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    async def run(self, **kwargs: Any) -> ToolResult:
        self.calls.append(kwargs)
        if self._raises:
            raise self._raises
        return self._result or ToolResult(name=self.spec.name, content="ok")


class FakeChain:
    """按脚本逐轮返回。每轮 = 一组 StreamEvent + 一个 StreamEnd。"""

    def __init__(self, turns: list[dict[str, Any]]) -> None:
        self.turns = turns
        self.requests: list[LLMRequest] = []

    async def stream_with_fallback(self, req: LLMRequest) -> AsyncIterator:
        idx = len(self.requests)
        self.requests.append(req)
        turn = self.turns[min(idx, len(self.turns) - 1)]
        for text in turn.get("content", []):
            yield StreamEvent(kind="content", text=text)
        for text in turn.get("reasoning", []):
            yield StreamEvent(kind="reasoning", text=text)
        yield StreamEnd(
            finish_reason=turn.get("finish", "stop"),
            provider="fake",
            model="fake-model",
            tool_calls=turn.get("tool_calls", []),
            reasoning=turn.get("reasoning_text"),
        )

    async def aclose(self) -> None:
        pass


def _settings(**kw) -> Settings:
    base = {
        "qdrant_url": "http://x",
        "qdrant_api_key": "k",
        "deepseek_api_key": "sk-x",
        "zhipu_api_key": "i.s",
    }
    base.update(kw)
    return Settings(_env_file=None, **base)


def _call(name: str, args: str = '{"query":"测试"}', cid: str = "c1") -> ToolCall:
    return ToolCall(id=cid, name=name, arguments=args)


async def _collect(agent: RagAgent, msgs=None):
    out = []
    async for item in agent.astream(msgs or [ChatMessage(role="user", content="问题")]):
        out.append(item)
    return out


async def _collect_kw(agent: RagAgent, **kwargs):
    out = []
    async for item in agent.astream([ChatMessage(role="user", content="问题")], **kwargs):
        out.append(item)
    return out


# ---------------------------------------------------------------- 工具注册表
def test_registry_schemas_are_openai_shaped():
    reg = ToolRegistry()
    reg.register(FakeTool("t1"))
    schemas = reg.schemas()
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "t1"
    assert "parameters" in schemas[0]["function"]


def test_empty_registry_returns_no_schemas():
    assert ToolRegistry().schemas() == []


async def test_unknown_tool_returns_error_result_not_exception():
    """模型可能编造工具名。这必须变成一条错误信息回灌，而不是抛异常打断对话。"""
    reg = ToolRegistry()
    result = await reg.execute("不存在的工具", {})
    assert result.ok is False
    assert result.error == "unknown_tool"
    assert "不存在" in result.content


async def test_tool_exception_is_contained():
    reg = ToolRegistry()
    reg.register(FakeTool("boom", raises=RuntimeError("内部爆炸")))
    result = await reg.execute("boom", {})
    assert result.ok is False
    assert "内部爆炸" in result.content  # 让模型知道失败了
    assert result.error == "RuntimeError"


async def test_bad_arguments_are_reported_not_raised():
    """模型给的参数名不对时，应提示重试，而不是让整个请求 500。

    这里刻意用一个**签名严格**的工具（不吞 `**kwargs`）——
    RagSearchTool 带了 `**_` 兜底，反而触发不到这条分支。
    """

    class StrictTool:
        spec = ToolSpec(
            name="strict",
            description="签名严格的工具",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )

        async def run(self, query: str) -> ToolResult:  # 不接受多余参数
            return ToolResult(name="strict", content=query)

    reg = ToolRegistry()
    reg.register(StrictTool())
    result = await reg.execute("strict", {"错误的参数名": 1})

    assert result.ok is False
    assert result.error == "bad_arguments"
    assert "参数不正确" in result.content


# ---------------------------------------------------------------- Agent 循环
async def test_agent_answers_directly_without_tool():
    """模型认为不需要检索时，一轮就给出答案。"""
    chain = FakeChain([{"content": ["直接回答"]}])
    reg = ToolRegistry()
    reg.register(FakeTool("t"))

    items = await _collect(RagAgent(chain, reg, _settings()))
    done = next(i for i in items if isinstance(i, AgentDone))

    assert done.steps == 1
    assert done.tool_calls == 0
    assert [i.text for i in items if isinstance(i, StreamEvent)] == ["直接回答"]


async def test_agent_calls_tool_then_answers():
    chain = FakeChain(
        [
            {"tool_calls": [_call("search", '{"query":"投标保证金"}')]},
            {"content": ["根据资料，..."]},
        ]
    )
    reg = ToolRegistry()
    tool = FakeTool("search", ToolResult(name="search", content="资料内容"))
    reg.register(tool)

    items = await _collect(RagAgent(chain, reg, _settings()))

    call = next(i for i in items if isinstance(i, ToolCallNotice))
    assert call.name == "search"
    assert call.arguments["query"] == "投标保证金"
    assert "投标保证金" in call.display  # 展示文案里带上了查询词

    result = next(i for i in items if isinstance(i, ToolResultNotice))
    assert result.ok

    done = next(i for i in items if isinstance(i, AgentDone))
    assert done.steps == 2
    assert done.tool_calls == 1
    assert tool.calls == [{"query": "投标保证金"}]


async def test_agent_sends_tool_result_back_to_model():
    """工具结果必须作为 role=tool 的消息回灌，并带上 tool_call_id。"""
    chain = FakeChain([{"tool_calls": [_call("search", cid="call-xyz")]}, {"content": ["答案"]}])
    reg = ToolRegistry()
    reg.register(FakeTool("search", ToolResult(name="search", content="检索到的资料")))

    await _collect(RagAgent(chain, reg, _settings()))

    second = chain.requests[1].messages
    assistant = next(m for m in second if m.role == "assistant" and m.tool_calls)
    assert assistant.tool_calls[0].id == "call-xyz"

    tool_msg = next(m for m in second if m.role == "tool")
    assert tool_msg.tool_call_id == "call-xyz"
    assert tool_msg.content == "检索到的资料"


async def test_assistant_message_echoes_reasoning_content():
    """DeepSeek 要求：带 tool_calls 的 assistant 消息必须回传 reasoning_content，
    否则后续请求会被 400 拒绝。"""
    chain = FakeChain(
        [
            {
                "reasoning": ["我在想"],
                "reasoning_text": "我在想",
                "tool_calls": [_call("search")],
            },
            {"content": ["答案"]},
        ]
    )
    reg = ToolRegistry()
    reg.register(FakeTool("search"))

    await _collect(RagAgent(chain, reg, _settings()))

    assistant = next(
        m for m in chain.requests[1].messages if m.role == "assistant" and m.tool_calls
    )
    assert assistant.reasoning_content == "我在想"
    assert assistant.to_dict()["reasoning_content"] == "我在想"


async def test_last_step_force_drops_tools():
    """达到上限的那一轮必须不带工具，逼模型作答——否则可能无限检索、用户拿不到答案。"""
    # 每一轮都要求调用工具，模拟模型「停不下来」
    chain = FakeChain([{"tool_calls": [_call("search")]}] * 5)
    reg = ToolRegistry()
    reg.register(FakeTool("search"))

    items = await _collect(RagAgent(chain, reg, _settings(agent_max_steps=3)))
    done = next(i for i in items if isinstance(i, AgentDone))

    # 三次请求：前两次带工具，最后一次必须不带
    assert len(chain.requests) == 3
    assert chain.requests[0].tools, "第 1 轮应带工具"
    assert chain.requests[1].tools, "第 2 轮应带工具"
    assert chain.requests[2].tools is None, "最后一轮必须去掉工具，逼模型回答"

    # 这个假 provider 无视「没有工具」的约定、仍然索要工具调用，
    # 属于异常行为。循环必须给出明确的收尾（max_steps）而不是静默卡住或死循环。
    assert done.finish_reason == "max_steps"
    assert done.tool_calls == 3


async def test_tools_are_passed_only_when_registered():
    """没有注册工具时不应带 tools 参数（部分部署对空数组会报错）。"""
    chain = FakeChain([{"content": ["答案"]}])
    await _collect(RagAgent(chain, ToolRegistry(), _settings()))
    assert chain.requests[0].tools is None


async def test_multiple_tool_calls_in_one_turn_all_execute():
    """一轮里模型可能并行发起多个工具调用，每个都要执行。"""
    chain = FakeChain(
        [
            {
                "tool_calls": [
                    _call("search", '{"query":"甲"}', "c1"),
                    _call("search", '{"query":"乙"}', "c2"),
                ]
            },
            {"content": ["答案"]},
        ]
    )
    reg = ToolRegistry()
    tool = FakeTool("search")
    reg.register(tool)

    items = await _collect(RagAgent(chain, reg, _settings()))

    assert len([i for i in items if isinstance(i, ToolCallNotice)]) == 2
    assert len(tool.calls) == 2
    assert {c["query"] for c in tool.calls} == {"甲", "乙"}

    # 两条 tool 消息都要回灌，否则消息序列不完整、DeepSeek 会报错
    tool_msgs = [m for m in chain.requests[1].messages if m.role == "tool"]
    assert len(tool_msgs) == 2
    assert {m.tool_call_id for m in tool_msgs} == {"c1", "c2"}


# ---------------------------------------------------------------- 图谱工具接入 Agent
def test_default_registry_registers_graph_only_when_provided():
    reg_without = build_default_registry(FakeRetriever([]), _settings())
    assert reg_without.names == ["search_bidding_knowledge"]

    graph = SimpleNamespace(available=True)
    reg_with = build_default_registry(FakeRetriever([]), _settings(), graph=graph)
    assert reg_with.names == ["search_bidding_knowledge", "query_bidding_graph"]


async def test_graph_tool_display_and_summary():
    """图谱调用的界面展示：动作名 + 实体锚点；摘要优先用工具自报的 artifacts['summary']。"""
    chain = FakeChain(
        [
            {
                "tool_calls": [
                    _call("query_bidding_graph", '{"subject_matter":"空调","aspect":"overview"}')
                ]
            },
            {"content": ["空调的交易频次是 179。"]},
        ]
    )
    reg = ToolRegistry()
    reg.register(
        FakeTool(
            "query_bidding_graph",
            ToolResult(
                name="query_bidding_graph",
                content="【知识图谱·标的物概览】…",
                artifacts={"summary": "知识图谱查询到 1 条结果", "graph": {"rows": 1}},
            ),
        )
    )

    items = await _collect(RagAgent(chain, reg, _settings()))

    call = next(i for i in items if isinstance(i, ToolCallNotice))
    assert call.display.startswith("查询知识图谱：")
    assert "空调" in call.display

    result = next(i for i in items if isinstance(i, ToolResultNotice))
    # 图谱结果没有 sources；若走老的「按 sources 计数」逻辑会误显示成「未检索到相关内容」
    assert result.summary == "知识图谱查询到 1 条结果"
    assert result.sources == []


async def test_cypher_display_is_single_line_and_truncated():
    cypher = (
        "MATCH (s:SubjectMatter)\nWHERE s.name CONTAINS $kw\n"
        "RETURN s.name AS name, s.tradeFrequency AS trade_frequency LIMIT 3"
    )
    chain = FakeChain(
        [
            {"tool_calls": [_call("query_bidding_graph", json.dumps({"cypher": cypher}))]},
            {"content": ["答案"]},
        ]
    )
    reg = ToolRegistry()
    reg.register(FakeTool("query_bidding_graph"))

    items = await _collect(RagAgent(chain, reg, _settings()))

    call = next(i for i in items if isinstance(i, ToolCallNotice))
    assert "\n" not in call.display
    assert call.display.startswith("查询知识图谱：MATCH (s:SubjectMatter) WHERE")
    assert len(call.display) <= len("查询知识图谱：") + 40 + 1


async def test_graph_error_codes_have_chinese_labels():
    reg = ToolRegistry()
    reg.register(
        FakeTool(
            "query_bidding_graph",
            ToolResult(
                name="query_bidding_graph", content="x", ok=False, error="graph_unavailable"
            ),
        )
    )
    chain = FakeChain([{"tool_calls": [_call("query_bidding_graph", "{}")]}, {"content": ["好"]}])

    items = await _collect(RagAgent(chain, reg, _settings()))
    result = next(i for i in items if isinstance(i, ToolResultNotice))
    assert result.summary == "知识图谱不可用"


async def test_agent_loop_reinjects_graph_tool_result():
    chain = FakeChain(
        [
            {"tool_calls": [_call("query_bidding_graph", '{"subject_matter":"空调"}', "g1")]},
            {"content": ["空调的交易频次是 179。"]},
        ]
    )
    reg = ToolRegistry()
    reg.register(
        FakeTool(
            "query_bidding_graph",
            ToolResult(name="query_bidding_graph", content="【知识图谱】交易频次 179"),
        )
    )

    items = await _collect(RagAgent(chain, reg, _settings()))

    tool_msg = next(m for m in chain.requests[1].messages if m.role == "tool")
    assert tool_msg.tool_call_id == "g1"
    assert "知识图谱" in tool_msg.content
    done = next(i for i in items if isinstance(i, AgentDone))
    assert done.finish_reason == "stop"


# ---------------------------------------------------------------- 联网搜索工具接入 Agent
def test_default_registry_registers_web_tools_when_available():
    web_keyword = SimpleNamespace(keyword_available=True, semantic_available=False)
    reg = build_default_registry(FakeRetriever([]), _settings(), web_search=web_keyword)
    assert reg.names == ["search_bidding_knowledge", "search_web_keyword"]

    web_semantic = SimpleNamespace(keyword_available=False, semantic_available=True)
    reg2 = build_default_registry(FakeRetriever([]), _settings(), web_search=web_semantic)
    assert reg2.names == ["search_bidding_knowledge", "search_web_semantic"]


async def test_allowed_tools_filters_schemas():
    chain = FakeChain([{"content": ["答案"]}])
    reg = ToolRegistry()
    reg.register(FakeTool("t1"))
    reg.register(FakeTool("t2"))

    await _collect_kw(RagAgent(chain, reg, _settings()), allowed_tools={"t1"})

    names = [tool["function"]["name"] for tool in chain.requests[0].tools]
    assert names == ["t1"]


async def test_disallowed_tool_is_not_executed_even_if_model_asks():
    """只过滤 schema 不够：模型幻构/越权点名时执行侧也必须拒绝。"""
    denied = FakeTool("t2", raises=AssertionError("被禁用的工具不该执行"))
    reg = ToolRegistry()
    reg.register(FakeTool("t1"))
    reg.register(denied)
    chain = FakeChain([{"tool_calls": [_call("t2", "{}")]}, {"content": ["改用其他方式作答"]}])

    items = await _collect_kw(RagAgent(chain, reg, _settings()), allowed_tools={"t1"})

    result = next(i for i in items if isinstance(i, ToolResultNotice))
    assert result.ok is False
    assert result.summary == "工具未启用"
    assert denied.calls == []

    tool_msg = next(m for m in chain.requests[1].messages if m.role == "tool")
    assert "未启用" in tool_msg.content


async def test_web_tool_display_labels():
    chain = FakeChain(
        [
            {"tool_calls": [_call("search_web_keyword", '{"query":"2026 政府采购 新规"}')]},
            {"content": ["答案"]},
        ]
    )
    reg = ToolRegistry()
    reg.register(FakeTool("search_web_keyword"))

    items = await _collect(RagAgent(chain, reg, _settings()))
    call = next(i for i in items if isinstance(i, ToolCallNotice))
    assert call.display.startswith("联网搜索（关键词）：")
    assert "政府采购" in call.display


async def test_web_sources_flow_into_notice():
    web_source = {
        "url": "https://example.com/a",
        "title": "标题",
        "snippet": "摘要",
        "score": 0.9,
        "engine": "tavily",
        "published_date": None,
    }
    reg = ToolRegistry()
    reg.register(
        FakeTool(
            "search_web_keyword",
            ToolResult(
                name="search_web_keyword",
                content="【联网搜索结果】…",
                artifacts={"web_sources": [web_source], "summary": "联网搜索到 1 条结果"},
            ),
        )
    )
    chain = FakeChain(
        [{"tool_calls": [_call("search_web_keyword", '{"query":"q"}')]}, {"content": ["答案"]}]
    )

    items = await _collect(RagAgent(chain, reg, _settings()))
    result = next(i for i in items if isinstance(i, ToolResultNotice))

    assert result.web_sources == [web_source]
    assert result.sources == []
    assert result.summary == "联网搜索到 1 条结果"


# ---------------------------------------------------------------- RAG 工具
class FakeRetriever:
    def __init__(self, sources, degraded=False, reason=None, available=True):
        self._sources = sources
        self._degraded = degraded
        self._reason = reason
        self.available = available
        self.queries: list[str] = []

    async def retrieve(self, query: str, top_k=None) -> RetrievalResult:
        self.queries.append(query)
        return RetrievalResult(
            sources=self._sources, elapsed_ms=12, degraded=self._degraded, reason=self._reason
        )


def _source(cid="qa-0001"):
    return RetrievedSource(
        chunk_id=cid,
        source_label="样例数据 · 第2行",
        question="投标保证金什么时候退还？",
        answer="书面合同签订后5日内退还。",
        text="问\n答",
        excel_row=2,
        rrf_score=0.01,
        rerank_score=0.9,
    )


def test_rag_tool_spec_is_well_formed():
    spec = RagSearchTool(FakeRetriever([])).spec
    assert spec.name == "search_bidding_knowledge"
    assert spec.parameters["required"] == ["query"]
    # 描述必须讲清「什么时候用」，这是模型决策的唯一依据
    assert "必须" in spec.description


async def test_rag_tool_formats_sources_with_labels():
    tool = RagSearchTool(FakeRetriever([_source()]))
    result = await tool.run(query="保证金")
    assert result.ok
    assert "样例数据 · 第2行" in result.content  # 出处要带进资料里，模型才能标注
    assert "书面合同签订后5日内退还" in result.content
    assert len(result.artifacts["sources"]) == 1


async def test_rag_tool_empty_result_tells_model_not_to_guess():
    tool = RagSearchTool(FakeRetriever([]))
    result = await tool.run(query="无关问题")
    assert result.ok
    assert "无法确定" in result.content
    assert result.artifacts["sources"] == []


async def test_rag_tool_reports_unavailable_encoder_clearly():
    """本地模型未就绪时要给出**可转述**的失败信息，而不是抛 AttributeError。

    关键在于仍然注册这个工具：不注册的话模型会转而凭记忆回答招投标问题，
    这比「答不出来」危险得多。
    """
    retriever = FakeRetriever([], available=False)
    tool = RagSearchTool(retriever)

    result = await tool.run(query="投标保证金什么时候退？")

    assert result.ok is False
    assert result.error == "retriever_unavailable"
    assert "未就绪" in result.content
    assert "不要凭自身记忆" in result.content
    assert retriever.queries == [], "不可用时不该再去调检索"


async def test_agent_still_answers_when_tool_is_unavailable():
    """工具不可用不该让整个 Agent 挂掉——模型应收到失败信息后照常作答
    （可以如实说知识库不可用，或回答不需要检索的问题）。"""
    chain = FakeChain(
        [
            {"tool_calls": [_call("search_bidding_knowledge", '{"query":"保证金"}')]},
            {"content": ["知识库暂时无法访问，请稍后再试。"]},
        ]
    )
    reg = ToolRegistry()
    reg.register(RagSearchTool(FakeRetriever([], available=False)))

    items = await _collect(RagAgent(chain, reg, _settings()))

    result = next(i for i in items if isinstance(i, ToolResultNotice))
    assert result.ok is False
    assert "不可用" in result.summary

    done = next(i for i in items if isinstance(i, AgentDone))
    assert done.finish_reason == "stop"
    assert any(isinstance(i, StreamEvent) and "无法访问" in i.text for i in items)

    # 失败信息必须回灌给模型，它才能据此作答
    tool_msg = next(m for m in chain.requests[1].messages if m.role == "tool")
    assert "未就绪" in tool_msg.content


async def test_rag_tool_rejects_empty_query():
    tool = RagSearchTool(FakeRetriever([]))
    result = await tool.run(query="   ")
    assert result.ok is False
    assert result.error == "empty_query"


async def test_rag_tool_warns_when_degraded():
    """稀疏检索不可用时提醒模型保守作答。"""
    tool = RagSearchTool(FakeRetriever([_source()], degraded=True, reason="词表不一致"))
    result = await tool.run(query="保证金")
    assert "关键词检索不可用" in result.content


@pytest.mark.parametrize("bad", ['{"query":', "not json", ""])
async def test_tool_call_tolerates_malformed_arguments(bad):
    """模型给的 arguments 可能不是合法 JSON，不能因此崩掉。"""
    call = ToolCall(id="c", name="search", arguments=bad)
    assert call.parsed_arguments() == {}
