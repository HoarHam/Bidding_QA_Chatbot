"""API 与 SSE 协议测试——不打网络、不加载模型。

做法是把 `app.state` 上的检索器与 LLM 链换成假实现。
`app.state` 是注入点（而不是 FastAPI 的 `Depends`），
所以这里直接赋值即可，不需要 monkeypatch 模块级全局变量。

不启动 lifespan（httpx 的 ASGITransport 默认不触发），
因此不会去下载模型或连 Qdrant。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

import httpx
import pytest

from bidding_qa_chatbot.api.app import create_app
from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.llm.base import (
    AllProvidersFailed,
    Attempt,
    MidStreamError,
    StreamEnd,
    StreamEvent,
)
from bidding_qa_chatbot.retrieval.service import RetrievalResult, RetrievedSource
from bidding_qa_chatbot.schemas import FallbackEvent, ProviderEvent
from tests.conftest import register_user


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        qdrant_url="http://x",
        qdrant_api_key="k",
        deepseek_api_key="sk-t",
        zhipu_api_key="i.s",
        jwt_secret="test-secret-" + "x" * 40,
    )


def _source(chunk_id: str = "qa0113-0000") -> RetrievedSource:
    return RetrievedSource(
        chunk_id=chunk_id,
        source_label="招投标问答样例数据0113 · 第2行",
        question="供应商针对单一来源异议被驳回，可以再次异议吗？",
        answer="对同一采购程序的异议只能提出一次。",
        text="问\n答",
        excel_row=2,
        rrf_score=0.016,
        rerank_score=0.87,
        images=[],
    )


class FakeRetriever:
    def __init__(self, sources=None) -> None:
        self._sources = sources if sources is not None else [_source()]

    async def retrieve(self, query: str, top_k=None) -> RetrievalResult:
        return RetrievalResult(sources=self._sources, elapsed_ms=12)


class FakeChain:
    """按脚本产出事件；`fail_before` 模拟所有 provider 首 token 前全挂。"""

    def __init__(
        self, script=None, fail_before: Exception | None = None, fail_mid: Exception | None = None
    ) -> None:
        self.script = script if script is not None else [("text", "根据《政府采购法》")]
        self.fail_before = fail_before
        self.fail_mid = fail_mid

    async def stream_with_fallback(self, req) -> AsyncIterator:
        if self.fail_before is not None:
            raise self.fail_before
        yield ProviderEvent(
            provider="deepseek", model="deepseek-flash", mode=req.mode, thinking_enabled=False
        )
        for kind, payload in self.script:
            if kind == "text":
                yield StreamEvent(kind="content", text=payload)
        if self.fail_mid is not None:
            raise self.fail_mid
        yield StreamEnd(finish_reason="stop", provider="deepseek", model="deepseek-flash")

    async def complete(self, req):
        raise NotImplementedError

    async def aclose(self) -> None:
        pass


def _make_app(chain, retriever=None, settings=None):
    app = create_app(settings or _settings())
    # 直接注入，绕过真实的 lifespan（否则会去下载模型、连 Qdrant）
    app.state.settings = settings or _settings()
    app.state.chain = chain
    app.state.encoder = object()
    app.state.reranker = None
    app.state.bm25 = None
    app.state.collection_vocab_sha1 = None
    app.state.__dict__["_fake_retriever"] = retriever or FakeRetriever()
    return app


@pytest.fixture
def client_factory(monkeypatch):
    """构造**已登录**的测试客户端，并把检索器替换成假的。

    聊天端点现在要求登录（用户对话历史要落到账号上），所以这里先注册一个用户、
    把 token 挂在客户端默认头上。数据库用内存 SQLite，不碰真实 Postgres。
    """
    from bidding_qa_chatbot.api import routes_chat
    from bidding_qa_chatbot.db.session import Database

    async def _make(chain, retriever=None):
        retriever = retriever or FakeRetriever()

        async def fake_retrieve(request, body):
            return await retriever.retrieve(body.query, body.top_k)

        monkeypatch.setattr(routes_chat, "_retrieve", fake_retrieve)

        settings = _settings()
        db = Database("sqlite+aiosqlite:///:memory:", settings)
        await db.create_all()

        app = _make_app(chain, settings=settings)
        app.state.db = db

        # 注册用**临时**客户端：httpx 的 client 只要发过请求就不能再 open，
        # 而测试里是用 `async with client:` 打开的，提前用过会报
        # "Cannot open a client instance more than once"。
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as bootstrap:
            token, _ = await register_user(bootstrap, username=f"u{uuid.uuid4().hex[:8]}")

        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        client.headers["Authorization"] = f"Bearer {token}"
        return client

    return _make


async def test_stream_event_order_is_sources_provider_tokens_done(client_factory):
    client = await client_factory(FakeChain([("text", "甲"), ("text", "乙")]))
    async with client:
        resp = await client.post("/api/chat/stream", json={"query": "问题", "mode": "fast"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text

    events = [line[7:] for line in body.splitlines() if line.startswith("event: ")]
    assert events == ["sources", "provider", "token", "token", "done"]


async def test_sources_carry_chinese_unescaped(client_factory):
    client = await client_factory(FakeChain())
    async with client:
        resp = await client.post("/api/chat/stream", json={"query": "问题"})
        body = resp.text

    # 中文必须是原样 UTF-8，不能是 \uXXXX 转义
    assert "单一来源" in body
    assert "\\u" not in body

    data = [line[6:] for line in body.splitlines() if line.startswith("data: ")][0]
    payload = json.loads(data)
    assert payload["sources"][0]["chunk_id"] == "qa0113-0000"
    assert payload["retrieved"] == 1


async def test_total_failure_returns_503_json_not_sse(client_factory):
    """首 token 前全挂 → 真正的 503 JSON，前端可按普通错误处理。"""
    attempts = [Attempt("deepseek", "deepseek-flash", "first_token", "TimeoutError", "超时", 15012)]
    chain = FakeChain(fail_before=AllProvidersFailed(attempts))
    client = await client_factory(chain)

    async with client:
        resp = await client.post("/api/chat/stream", json={"query": "问题"})

    assert resp.status_code == 503
    assert resp.headers["content-type"].startswith("application/json")
    payload = resp.json()
    assert payload["error"]["code"] == "all_providers_failed"
    assert payload["error"]["attempts"][0]["provider"] == "deepseek"


async def test_mid_stream_failure_keeps_partial_and_marks_partial(client_factory):
    """中途失败：已产出的 token 必须保留，并明确标记 partial。"""
    chain = FakeChain(
        [("text", "前半段")],
        fail_mid=MidStreamError("deepseek", "连接中断"),
    )
    client = await client_factory(chain)

    async with client:
        resp = await client.post("/api/chat/stream", json={"query": "问题"})
        assert resp.status_code == 200
        body = resp.text

    assert "前半段" in body
    assert "stream_interrupted" in body
    error_line = next(
        line
        for line in body.splitlines()
        if line.startswith("data: ") and "stream_interrupted" in line
    )
    assert json.loads(error_line[6:])["partial"] is True


async def test_fallback_event_surfaces_to_client(client_factory):
    class FallbackChain(FakeChain):
        async def stream_with_fallback(self, req):
            yield FallbackEvent(
                from_provider="deepseek",
                to_provider="zhipu",
                reason="first_token_timeout",
                attempts=[],
            )
            async for item in super().stream_with_fallback(req):
                yield item

    client = await client_factory(FallbackChain())
    async with client:
        resp = await client.post("/api/chat/stream", json={"query": "问题"})
        body = resp.text

    assert "event: fallback" in body
    fb = next(line for line in body.splitlines() if line.startswith("data: ") and '"to"' in line)
    assert json.loads(fb[6:])["from"] == "deepseek"


async def test_no_sources_still_streams_answer(client_factory):
    """检索为空时仍应作答，但提示词会要求模型声明无法确定。"""
    client = await client_factory(
        FakeChain([("text", "根据知识库无法确定")]), retriever=FakeRetriever(sources=[])
    )
    async with client:
        resp = await client.post("/api/chat/stream", json={"query": "无关问题"})
        body = resp.text

    assert "event: sources" in body
    assert "无法确定" in body


async def test_chat_request_validation(client_factory):
    client = await client_factory(FakeChain())
    async with client:
        resp = await client.post("/api/chat/stream", json={"query": ""})
    assert resp.status_code == 422


async def test_retrieve_endpoint_returns_sources(client_factory):
    client = await client_factory(FakeChain())
    async with client:
        resp = await client.post("/api/retrieve", json={"query": "问题"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["sources"][0]["source_label"].startswith("招投标问答样例数据")


async def test_health_is_liveness_only():
    app = create_app(_settings())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "deps": []}


def test_agent_built_even_when_local_models_missing():
    """回归守卫：AGENT_ENABLED=true 时必须建 Agent，**不能因 encoder 缺失而跳过**。

    这条逻辑曾经写反过：本地模型没就绪就不建 Agent，转而去走「固定检索路径」。
    但那条路径在缺 encoder 时是直接 503 的——所谓「回退」实际是把
    「部分可用」（模型仍能回答闲聊、并如实告知知识库不可用）变成了
    「完全不可用」。两者是正交的：Agent 决定由谁判断要不要检索，
    encoder 只影响工具内部能不能跑。
    """
    from bidding_qa_chatbot.api.app import build_agent

    settings = _settings()
    settings.agent_enabled = True

    agent = build_agent(
        settings,
        chain=object(),
        qdrant=object(),
        encoder=None,  # 本地模型未就绪
        reranker=None,
        bm25_artifact=None,
        collection_vocab_sha1=None,
    )

    assert agent is not None, "本地模型缺失不应阻止 Agent 建立"
    assert "search_bidding_knowledge" in agent.registry.names
    assert agent.registry.get("search_bidding_knowledge").retriever.available is False


def test_agent_not_built_when_disabled():
    from bidding_qa_chatbot.api.app import build_agent

    settings = _settings()
    settings.agent_enabled = False

    assert (
        build_agent(
            settings,
            chain=object(),
            qdrant=object(),
            encoder=object(),
            reranker=None,
            bm25_artifact=None,
            collection_vocab_sha1=None,
        )
        is None
    )


def test_agent_registers_graph_tool_only_when_service_provided():
    """图谱工具只在传入 GraphService 时注册（ENABLE_NEO4J=false 时模型不该看到它）。"""
    from types import SimpleNamespace

    from bidding_qa_chatbot.api.app import build_agent

    settings = _settings()
    settings.agent_enabled = True
    common = {
        "chain": object(),
        "qdrant": object(),
        "encoder": None,
        "reranker": None,
        "bm25_artifact": None,
        "collection_vocab_sha1": None,
    }

    with_graph = build_agent(settings, **common, graph=SimpleNamespace(available=True))
    assert "query_bidding_graph" in with_graph.registry.names
    assert with_graph.registry.names == ["search_bidding_knowledge", "query_bidding_graph"]

    without = build_agent(settings, **common)
    assert "query_bidding_graph" not in without.registry.names


async def test_assistant_answer_with_sources_is_persisted(client_factory):
    """回归守卫：**带引用来源**的回答必须能落库。

    曾经的 bug：`sources` 里存的是 `SourceRef` 这个 **pydantic 对象**，而 `sa.JSON`
    底层走 `json.dumps`，遇到 BaseModel 抛
    "Object of type SourceRef is not JSON serializable"。异常发生在回答已经
    流式返回**之后**，被 `_persist_assistant` 的 except 降级成一条日志——
    用户看到的是「回答显示正常，但切回历史时只剩提问、没有回答」，
    而且再提问时模型会把那个没被恢复的问题当成待答问题一并回答。

    只有带来源的回答才会复现（无来源时是空列表，能正常序列化），
    所以表现为「没调用 RAG 的回答正常」。

    既有的 SSE 测试没发现，是因为它们只断言事件流文本，**从不回查数据库**。
    """
    client = await client_factory(FakeChain([("text", "根据资料，保证金应在5日内退还。")]))
    async with client:
        resp = await client.post("/api/chat/stream", json={"query": "保证金什么时候退？"})
        assert resp.status_code == 200
        body = resp.text

        done_line = next(
            (
                line
                for line in body.splitlines()
                if line.startswith("data: ") and '"conversation_id"' in line
            ),
            None,
        )
        assert done_line, "done 事件里应带上 conversation_id"
        conversation_id = json.loads(done_line[6:])["conversation_id"]
        assert conversation_id

        # 关键：回查数据库
        detail = await client.get(f"/api/conversations/{conversation_id}")
        assert detail.status_code == 200

    messages = detail.json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"], (
        f"助手回答没有落库，实际消息：{[m['role'] for m in messages]}"
    )

    assistant = messages[1]
    assert assistant["content"] == "根据资料，保证金应在5日内退还。"
    assert len(assistant["sources"]) == 1, "引用来源必须一并存下来"
    assert assistant["sources"][0]["chunk_id"] == "qa0113-0000"
    assert assistant["sources"][0]["source_label"].startswith("招投标问答样例数据")


async def test_persisted_answer_survives_reload_with_sources(client_factory):
    """跨会话恢复：重新拉取会话时，带来源的回答要能完整还原。

    这条直接对应「恢复会话后只显示用户问题、不显示回答」的症状。
    """
    client = await client_factory(FakeChain([("text", "回答内容")]))
    async with client:
        resp = await client.post("/api/chat/stream", json={"query": "问题"})
        done_line = next(
            line
            for line in resp.text.splitlines()
            if line.startswith("data: ") and '"conversation_id"' in line
        )
        cid = json.loads(done_line[6:])["conversation_id"]

        # 模拟用户刷新页面后重新拉取
        first = (await client.get(f"/api/conversations/{cid}")).json()
        second = (await client.get(f"/api/conversations/{cid}")).json()

    assert len(first["messages"]) == 2
    assert first["messages"] == second["messages"], "两次拉取结果应一致"
    assert first["messages"][1]["sources"], "来源不能在往返中丢失"


def test_accumulate_dedupes_sources_by_chunk_id():
    """回归守卫：多次检索命中同一 chunk 时不能产生重复来源。

    Agent 常做多次检索（先查主问题、再补限定词），两次命中同一个 chunk
    就会出现重复条目。除了冗余，还会让前端以 chunk_id 为 key 渲染时触发
    React 的 "Encountered two children with the same key"。
    """
    from bidding_qa_chatbot.agent.agent import ToolResultNotice
    from bidding_qa_chatbot.api.routes_chat import _accumulate

    def src(cid: str) -> dict:
        return {
            "chunk_id": cid,
            "source_label": f"标签{cid}",
            "question": "q",
            "snippet": "s",
            "excel_row": 1,
            "rrf_score": 0.1,
            "rerank_score": 0.9,
            "images": [],
        }

    def notice(cids: list[str]) -> ToolResultNotice:
        return ToolResultNotice(
            name="t", ok=True, elapsed_ms=1, summary="", sources=[src(c) for c in cids]
        )

    answer: list = []
    reasoning: list = []
    sources: list = []
    finished: dict = {}

    _accumulate(notice(["a", "b"]), answer, reasoning, sources, finished)
    _accumulate(notice(["b", "c"]), answer, reasoning, sources, finished)

    assert [s.chunk_id for s in sources] == ["a", "b", "c"], "重复来源没有被去掉"
    # 去重后必须仍是 SourceRef（可 JSON 序列化），这是能落库的前提
    assert sources[0].__class__.__name__ == "SourceRef"


def test_http_client_timeout_is_constructible():
    """回归守卫：`httpx.Timeout` 只给 connect/read 会抛 ValueError。

    这个错误只在服务启动时才暴露，而测试不跑 lifespan，
    所以单独构造一次客户端，把这类问题挡在启动之前。
    """
    from bidding_qa_chatbot.api.app import build_http_client

    client = build_http_client(_settings())
    assert client.timeout.connect == _settings().llm_connect_timeout


def test_all_httpx_timeouts_in_codebase_are_valid():
    """扫描源码，确保没有第二个地方写了非法的 httpx.Timeout。

    同一个错误在 zhipu.py / vision.py / app.py 各犯过一次，
    这条扫描比逐个补测试更可靠。
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    pattern = re.compile(r"httpx\.Timeout\(\s*(?!\d|[a-z_]+\.)([^)]*)\)", re.IGNORECASE)
    offenders = []
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            args = match.group(1)
            # 合法写法：第一个参数是位置参数（不是 `name=` 形式），或四个关键字参数齐全
            has_positional = not re.match(r"\s*\w+\s*=", args)
            keywords = set(re.findall(r"(\w+)\s*=", args))
            if not has_positional and not {"connect", "read", "write", "pool"} <= keywords:
                offenders.append(f"{path.name}: httpx.Timeout({args.strip()[:60]})")
    assert not offenders, "非法的 httpx.Timeout 调用（必须给默认值或四个参数）: " + "; ".join(
        offenders
    )
