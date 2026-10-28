"""Langfuse 可观测性测试（完全离线）。

用一个假的 Langfuse client 记录 observation 树，验证三件事：

1. trace 的层级确实是 chat.turn → agent.step → generation/tool；
2. `LANGFUSE_CAPTURE_CONTENT=false` 时正文不出本机；
3. 任何 SDK 异常都不会冒泡到问答路径，反馈打分也不受采样影响。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from bidding_qa_chatbot.agent.agent import AgentDone, RagAgent, ToolCallNotice, ToolResultNotice
from bidding_qa_chatbot.agent.base import ToolRegistry, ToolResult, ToolSpec
from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.llm.base import (
    ChatMessage,
    StreamEnd,
    StreamEvent,
    ToolCall,
    Usage,
)
from bidding_qa_chatbot.observability import (
    LangfuseObservability,
    build_observability,
    messages_payload,
    usage_payload,
)


class FakeObservation:
    def __init__(self, name: str, as_type: str, kwargs: dict[str, Any]) -> None:
        self.name = name
        self.as_type = as_type
        self.kwargs = kwargs
        self.children: list[FakeObservation] = []
        self.updates: list[dict[str, Any]] = []
        self.ended = False

    def start_observation(self, *, name: str, as_type: str = "span", **kwargs: Any):
        child = FakeObservation(name, as_type, kwargs)
        self.children.append(child)
        return child

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def end(self) -> None:
        self.ended = True


class FakeLangfuseClient:
    """只实现本模块用到的 SDK 表面。"""

    trace_id = "a" * 32

    def __init__(self, *, fail_start: bool = False, fail_score: bool = False) -> None:
        self.roots: list[FakeObservation] = []
        self.scores: list[dict[str, Any]] = []
        self.flushed = False
        self.fail_start = fail_start
        self.fail_score = fail_score

    def create_trace_id(self) -> str:
        return self.trace_id

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        if self.fail_start:
            raise RuntimeError("langfuse unavailable")
        root = FakeObservation(kwargs["name"], kwargs["as_type"], kwargs)
        self.roots.append(root)
        return root

    def create_score(self, **kwargs: Any) -> None:
        if self.fail_score:
            raise RuntimeError("score endpoint unavailable")
        self.scores.append(kwargs)

    def flush(self) -> None:
        self.flushed = True


def _lf_settings(settings_factory, **overrides) -> Settings:
    base = {
        "langfuse_enabled": True,
        "langfuse_public_key": "pk-lf-test",
        "langfuse_secret_key": "sk-lf-test",
    }
    base.update(overrides)
    return settings_factory(**base)


def _turn(obs: LangfuseObservability):
    return obs.start_turn(
        user_id="user-1",
        conversation_id="conv-1",
        query="投标保证金多久退还？",
        mode="fast",
        web_search=True,
        has_attachments=False,
    )


# ---------------------------------------------------------------- 开关与降级
def test_missing_keys_disable_observability_without_raising(settings_factory):
    obs = build_observability(settings_factory())
    assert obs.enabled is False
    turn = _turn(obs)
    assert turn.trace_id is None
    assert turn.enabled is False
    obs.score_feedback(trace_id=None, message_id="m", rating="up")
    obs.flush()


def test_enabled_flag_false_skips_even_with_keys(settings_factory):
    obs = build_observability(
        settings_factory(
            langfuse_enabled=False,
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
        )
    )
    assert obs.enabled is False
    assert "LANGFUSE_ENABLED" in obs.reason


def test_one_key_is_not_enough(settings_factory):
    obs = build_observability(
        settings_factory(langfuse_enabled=True, langfuse_public_key="pk", langfuse_secret_key=None)
    )
    assert obs.enabled is False


def test_blank_keys_are_not_configured(settings_factory):
    settings = settings_factory(
        langfuse_enabled=True, langfuse_public_key="", langfuse_secret_key="   "
    )
    assert settings.langfuse_configured is False
    assert build_observability(settings).enabled is False


def test_sdk_failure_never_breaks_business(settings_factory):
    client = FakeLangfuseClient(fail_start=True, fail_score=True)
    obs = LangfuseObservability(_lf_settings(settings_factory), client=client)
    turn = _turn(obs)
    # 根 span 建不起来，但 trace id 仍保留，反馈还能尝试上报
    assert turn.trace_id == FakeLangfuseClient.trace_id
    assert turn.enabled is False
    turn.start_span("x").end(output="y")
    obs.score_feedback(trace_id=turn.trace_id, message_id="m", rating="up")
    obs.flush()
    assert client.flushed is True


# ---------------------------------------------------------------- trace 结构
def test_trace_hierarchy_and_content_capture(settings_factory):
    client = FakeLangfuseClient()
    obs = LangfuseObservability(_lf_settings(settings_factory), client=client)
    turn = _turn(obs)

    step = turn.start_span("agent.step", input={"message_count": 2}, metadata={"step": 1})
    generation = step.start_generation(
        "llm.step",
        input={"messages": [{"role": "user", "content": "你好"}]},
        metadata={"mode": "fast"},
    )
    tool = step.start_tool("tool.search", input={"arguments": {"query": "空调"}})
    tool.end(output={"content": "工具结果"}, metadata={"ok": True})
    generation.end(
        output={"content": "最终答案"},
        model="fake-model",
        usage_details={"input": 3, "output": 2},
    )
    step.end(output={"content": "最终答案"})
    turn.end(output={"answer": "最终答案"})

    assert turn.trace_id == FakeLangfuseClient.trace_id
    root = client.roots[0]
    assert root.kwargs["trace_context"] == {"trace_id": turn.trace_id}
    assert root.kwargs["input"] == {"query": "投标保证金多久退还？"}
    assert root.kwargs["metadata"]["mode"] == "fast"
    assert root.kwargs["metadata"]["web_search"] is True
    assert [child.name for child in root.children] == ["agent.step"]

    step_obs = root.children[0]
    assert [child.name for child in step_obs.children] == ["llm.step", "tool.search"]
    assert step_obs.children[0].updates[-1]["usage_details"] == {"input": 3, "output": 2}
    assert step_obs.children[0].updates[-1]["output"] == {"content": "最终答案"}
    assert step_obs.children[1].updates[-1]["output"] == {"content": "工具结果"}
    assert all(o.ended for o in (root, step_obs, *step_obs.children))


def test_content_capture_off_reports_lengths_only(settings_factory):
    client = FakeLangfuseClient()
    settings = _lf_settings(settings_factory, langfuse_capture_content=False)
    obs = LangfuseObservability(settings, client=client)
    turn = _turn(obs)

    generation = turn.start_generation(
        "llm.step", input={"messages": [{"role": "user", "content": "秘密正文"}]}
    )
    generation.end(output={"content": "回答正文"})
    turn.end(output={"answer": "回答正文"})

    root = client.roots[0]
    assert root.kwargs["input"] == {"query": {"chars": len("投标保证金多久退还？")}}

    generation_obs = root.children[0]
    assert generation_obs.kwargs["input"]["messages"][0]["content"] == {"chars": 4}
    assert generation_obs.updates[-1]["output"]["content"] == {"chars": 4}


def test_sampling_drops_spans_but_keeps_feedback_score(settings_factory):
    client = FakeLangfuseClient()
    settings = _lf_settings(settings_factory, langfuse_sample_rate=0.0)
    obs = LangfuseObservability(settings, client=client, sampler=lambda: 0.0)
    turn = _turn(obs)

    assert client.roots == []
    assert turn.trace_id == FakeLangfuseClient.trace_id
    assert turn.enabled is False

    obs.score_feedback(trace_id=turn.trace_id, message_id="m-1", rating="up")
    assert client.scores[-1]["value"] == 1.0


def test_feedback_score_is_upserted_by_message_id(settings_factory):
    client = FakeLangfuseClient()
    obs = LangfuseObservability(_lf_settings(settings_factory), client=client)

    obs.score_feedback(trace_id="a" * 32, message_id="m-1", rating="up")
    obs.score_feedback(trace_id="a" * 32, message_id="m-1", rating="down")
    obs.score_feedback(trace_id="a" * 32, message_id="m-1", rating=None)

    assert [s["value"] for s in client.scores] == [1.0, -1.0, 0.0]
    assert {s["score_id"] for s in client.scores} == {"user-feedback-m-1"}
    assert all(s["data_type"] == "NUMERIC" for s in client.scores)
    assert all(s["trace_id"] == "a" * 32 for s in client.scores)


# ---------------------------------------------------------------- payload
def test_messages_payload_keeps_tool_calls_and_roles():
    payload = messages_payload(
        [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="c1", name="search", arguments='{"query":"x"}')],
                reasoning_content="想一想",
            ),
            ChatMessage(role="tool", content="结果", tool_call_id="c1"),
        ]
    )
    assert payload[0]["tool_calls"] == [{"name": "search", "arguments": '{"query":"x"}'}]
    assert payload[0]["reasoning"] == "想一想"
    assert payload[1]["tool_call_id"] == "c1"


def test_usage_payload_omits_missing_values():
    assert usage_payload(None) is None
    assert usage_payload(Usage(prompt_tokens=1)) == {"input": 1}
    assert usage_payload(Usage(1, 2, 3)) == {"input": 1, "output": 2, "total": 3}


# ---------------------------------------------------------------- Agent 集成
class _FakeTool:
    spec = ToolSpec(
        name="search",
        description="测试检索",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )

    async def run(self, **kwargs: Any) -> ToolResult:
        return ToolResult(
            name="search",
            content="检索结果正文",
            artifacts={"sources": [{"chunk_id": "c1"}], "summary": "检索到 1 条"},
        )


class _FakeChain:
    def __init__(self, turns: list[dict[str, Any]]) -> None:
        self.turns = turns
        self.requests = []

    async def stream_with_fallback(self, req) -> AsyncIterator:
        index = len(self.requests)
        self.requests.append(req)
        turn = self.turns[min(index, len(self.turns) - 1)]
        # 带一段 reasoning，验证思维通道也会进入 generation 的 output
        yield StreamEvent(kind="reasoning", text="思考")
        for text in turn.get("content", []):
            yield StreamEvent(kind="content", text=text)
        yield StreamEnd(
            finish_reason="stop",
            provider="fake",
            model="fake-model",
            tool_calls=turn.get("tool_calls", []),
            usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )


async def test_agent_traces_steps_generations_and_tools(settings_factory):
    client = FakeLangfuseClient()
    obs = LangfuseObservability(_lf_settings(settings_factory), client=client)
    turn = _turn(obs)

    registry = ToolRegistry()
    registry.register(_FakeTool())
    chain = _FakeChain(
        [
            {"tool_calls": [ToolCall(id="call-1", name="search", arguments='{"query":"空调"}')]},
            {"content": ["最终答案"]},
        ]
    )
    agent = RagAgent(chain, registry, settings_factory(agent_max_steps=3))

    items = [
        item
        async for item in agent.astream(
            [ChatMessage(role="user", content="有哪些空调采购？")], trace=turn
        )
    ]
    turn.end(output={"answer": "最终答案"})

    assert any(isinstance(item, ToolCallNotice) for item in items)
    assert any(isinstance(item, ToolResultNotice) for item in items)
    assert isinstance(items[-1], AgentDone)

    root = client.roots[0]
    assert [child.name for child in root.children] == ["agent.step.1", "agent.step.2"]
    first, second = root.children
    assert [child.name for child in first.children] == ["llm.step.1", "tool.search"]
    assert [child.name for child in second.children] == ["llm.step.2"]

    tool_obs = first.children[1]
    assert tool_obs.kwargs["input"] == {"arguments": {"query": "空调"}}
    assert tool_obs.updates[-1]["output"]["content"] == "检索结果正文"
    assert tool_obs.updates[-1]["metadata"]["ok"] is True

    first_gen = first.children[0]
    assert first_gen.updates[-1]["output"]["tool_calls"][0]["name"] == "search"
    second_gen = second.children[0]
    assert second_gen.updates[-1]["output"]["content"] == "最终答案"
    assert second_gen.updates[-1]["usage_details"] == {"input": 3, "output": 2, "total": 5}
    assert all(o.ended for o in (root, first, second, *first.children, *second.children))


async def test_agent_without_trace_still_works(settings_factory):
    registry = ToolRegistry()
    chain = _FakeChain([{"content": ["直接回答"]}])
    agent = RagAgent(chain, registry, settings_factory(agent_max_steps=2))
    items = [item async for item in agent.astream([ChatMessage(role="user", content="q")])]
    assert isinstance(items[-1], AgentDone)
