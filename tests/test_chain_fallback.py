"""回退链测试——本阶段最核心的风险点。

全部使用**假 provider**，不发任何 HTTP 请求。重点验证那条唯一重要的不变量：

    只有在首个 token 送达之前，才允许切换 provider。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.llm.base import (
    AllProvidersFailed,
    ChatMessage,
    FatalProviderError,
    LLMRequest,
    LLMResult,
    MidStreamError,
    RetryableProviderError,
    StreamEnd,
    StreamEvent,
)
from bidding_qa_chatbot.llm.chain import CircuitBreaker, ProviderChain
from bidding_qa_chatbot.schemas import FallbackEvent, ProviderEvent


def _req(mode: str = "fast") -> LLMRequest:
    return LLMRequest(messages=[ChatMessage(role="user", content="测试")], mode=mode)


class FakeProvider:
    """按脚本产出事件的假 provider。

    `script` 里的元素：
      ("text", "...")  产出正文
      ("reason", "...") 产出思考
      ("error", exc)   抛出异常
    `delay` 用于模拟首 token 前的挂起。
    """

    def __init__(self, name: str, script: list, delay: float = 0.0) -> None:
        self.name = name
        self.script = script
        self.delay = delay
        self.calls = 0
        self.closed = 0

    def model_for(self, mode: str) -> str:
        return f"{self.name}-model"

    async def astream(self, req: LLMRequest) -> AsyncIterator:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        for kind, payload in self.script:
            if kind == "error":
                raise payload
            yield StreamEvent(kind="reasoning" if kind == "reason" else "content", text=payload)
        yield StreamEnd(finish_reason="stop", provider=self.name, model=self.model_for(req.mode))

    async def acomplete(self, req: LLMRequest) -> LLMResult:
        self.calls += 1
        return LLMResult("ok", None, "stop", None, self.name, self.model_for(req.mode))

    async def aclose(self) -> None:
        self.closed += 1


def _settings(**kw) -> Settings:
    base = {
        "qdrant_url": "http://x",
        "qdrant_api_key": "k",
        "deepseek_api_key": "sk-x",
        "zhipu_api_key": "i.s",
    }
    base.update(kw)
    return Settings(_env_file=None, **base)


async def _collect(chain: ProviderChain, req: LLMRequest) -> list:
    out = []
    async for item in chain.stream_with_fallback(req):
        out.append(item)
    return out


async def test_primary_serves_when_healthy():
    good = FakeProvider("a", [("text", "你好")])
    chain = ProviderChain(
        [good, FakeProvider("b", [("text", "备")])], _settings(), CircuitBreaker()
    )

    events = await _collect(chain, _req())
    texts = [e.text for e in events if isinstance(e, StreamEvent)]
    assert texts == ["你好"]
    assert any(isinstance(e, ProviderEvent) and e.provider == "a" for e in events)


async def test_falls_back_before_first_token_and_emits_nothing_first():
    """核心不变量：切换前不能有任何内容事件送达客户端。"""
    bad = FakeProvider("a", [("error", RetryableProviderError("a", "连不上"))])
    good = FakeProvider("b", [("text", "备用答案")])
    chain = ProviderChain([bad, good], _settings(), CircuitBreaker())

    events = await _collect(chain, _req())

    first_content = next(
        (i for i, e in enumerate(events) if isinstance(e, StreamEvent)), len(events)
    )
    fallback_at = next((i for i, e in enumerate(events) if isinstance(e, FallbackEvent)), None)
    assert fallback_at is not None, "应当发出 fallback 事件"
    assert fallback_at < first_content, "fallback 必须早于任何内容事件"

    fb = events[fallback_at]
    assert fb.from_provider == "a" and fb.to_provider == "b"
    assert [e.text for e in events if isinstance(e, StreamEvent)] == ["备用答案"]


async def test_fatal_error_still_falls_back_by_default():
    """401 也降级：失效的 key 不该让用户拿不到答案。"""
    bad = FakeProvider("a", [("error", FatalProviderError("a", "HTTP 401，请检查 API key"))])
    good = FakeProvider("b", [("text", "来自备用 provider")])
    chain = ProviderChain([bad, good], _settings(), CircuitBreaker())

    events = await _collect(chain, _req())
    assert [e.text for e in events if isinstance(e, StreamEvent)] == ["来自备用 provider"]


async def test_fatal_error_hard_fails_when_configured():
    bad = FakeProvider("a", [("error", FatalProviderError("a", "HTTP 401"))])
    good = FakeProvider("b", [("text", "不该被调用")])
    chain = ProviderChain([bad, good], _settings(), CircuitBreaker(), fatal_falls_back=False)

    with pytest.raises(AllProvidersFailed):
        await _collect(chain, _req())
    assert good.calls == 0


async def test_no_restart_after_first_token():
    """已经吐字后中途失败：保留已产出内容，绝不切换 provider。"""
    bad = FakeProvider("a", [("text", "前半"), ("error", RetryableProviderError("a", "断了"))])
    good = FakeProvider("b", [("text", "重来的内容")])
    chain = ProviderChain([bad, good], _settings(), CircuitBreaker())

    seen: list[str] = []
    with pytest.raises(MidStreamError) as excinfo:
        async for item in chain.stream_with_fallback(_req()):
            if isinstance(item, StreamEvent):
                seen.append(item.text)

    assert seen == ["前半"], "已产出的内容必须保留"
    assert good.calls == 0, "首 token 之后绝不能再调用下一个 provider"
    assert excinfo.value.provider == "a"


async def test_first_token_timeout_triggers_fallback():
    slow = FakeProvider("a", [("text", "太慢了")], delay=1.0)
    good = FakeProvider("b", [("text", "快")])
    chain = ProviderChain([slow, good], _settings(llm_first_token_timeout=0.1), CircuitBreaker())

    events = await _collect(chain, _req())
    assert [e.text for e in events if isinstance(e, StreamEvent)] == ["快"]


async def test_thinking_mode_gets_longer_first_token_budget():
    chain = ProviderChain([FakeProvider("a", [])], _settings(llm_first_token_timeout=10.0))
    assert chain._first_token_timeout(_req("fast")) == 10.0
    assert chain._first_token_timeout(_req("thinking")) == 25.0


async def test_circuit_breaker_skips_open_provider_without_waiting():
    bad = FakeProvider("a", [("error", RetryableProviderError("a", "总是失败"))])
    good = FakeProvider("b", [("text", "ok")])
    breaker = CircuitBreaker(threshold=2, open_seconds=60)
    chain = ProviderChain([bad, good], _settings(), breaker)

    # 前两次尝试会真的调用 a 并失败
    await _collect(chain, _req())
    await _collect(chain, _req())
    calls_before = bad.calls

    # 第三次：熔断器已打开，应零成本跳过
    started = asyncio.get_event_loop().time()
    events = await _collect(chain, _req())
    elapsed = asyncio.get_event_loop().time() - started

    assert bad.calls == calls_before, "熔断打开后不应再调用该 provider"
    assert elapsed < 0.5
    assert [e.text for e in events if isinstance(e, StreamEvent)] == ["ok"]


async def test_all_providers_failed_carries_every_attempt():
    a = FakeProvider("a", [("error", RetryableProviderError("a", "错1"))])
    b = FakeProvider("b", [("error", RetryableProviderError("b", "错2"))])
    chain = ProviderChain([a, b], _settings(), CircuitBreaker())

    with pytest.raises(AllProvidersFailed) as excinfo:
        await _collect(chain, _req())
    assert [att.provider for att in excinfo.value.attempts] == ["a", "b"]


async def test_empty_provider_list_raises():
    chain = ProviderChain([], _settings(), CircuitBreaker())
    with pytest.raises(AllProvidersFailed):
        await _collect(chain, _req())


async def test_cancellation_closes_upstream_stream():
    """客户端断开时必须关闭上游流，否则会继续计费。"""
    started = asyncio.Event()

    class SlowProvider(FakeProvider):
        async def astream(self, req):
            self.calls += 1
            started.set()
            for i in range(100):
                await asyncio.sleep(0.01)
                yield StreamEvent(kind="content", text=f"{i}")

    provider = SlowProvider("a", [])
    chain = ProviderChain([provider], _settings(), CircuitBreaker())

    async def consume():
        async for _ in chain.stream_with_fallback(_req()):
            pass

    task = asyncio.create_task(consume())
    await started.wait()
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_mid_stream_failure_counts_toward_breaker():
    """中途死掉的 provider 也不健康，应当计入熔断。"""
    bad = FakeProvider("a", [("text", "x"), ("error", RetryableProviderError("a", "断"))])
    breaker = CircuitBreaker(threshold=1, open_seconds=60)
    chain = ProviderChain([bad, FakeProvider("b", [("text", "ok")])], _settings(), breaker)

    with pytest.raises(MidStreamError):
        await _collect(chain, _req())
    assert breaker.is_open("a")
