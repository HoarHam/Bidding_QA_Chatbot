"""API 与 SSE 协议测试——不打网络、不加载模型。

做法是把 `app.state` 上的检索器与 LLM 链换成假实现。
`app.state` 是注入点（而不是 FastAPI 的 `Depends`），
所以这里直接赋值即可，不需要 monkeypatch 模块级全局变量。

不启动 lifespan（httpx 的 ASGITransport 默认不触发），
因此不会去下载模型或连 Qdrant。
"""

from __future__ import annotations

import json
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


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        qdrant_url="http://x",
        qdrant_api_key="k",
        deepseek_api_key="sk-t",
        zhipu_api_key="i.s",
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
    """构造测试客户端，并把检索器替换成假的。"""
    from bidding_qa_chatbot.api import routes_chat

    async def _make(chain, retriever=None):
        retriever = retriever or FakeRetriever()

        async def fake_retrieve(request, body):
            return await retriever.retrieve(body.query, body.top_k)

        monkeypatch.setattr(routes_chat, "_retrieve", fake_retrieve)
        app = _make_app(chain)
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://test")

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
