"""Provider 回退链：重试、熔断、以及「首 token 边界」。

## 唯一重要的不变量

**只有在首个 token 送达客户端之前，才能安全地换 provider。**

一旦已经吐字，再重试就会产生重复或自相矛盾的文本。这里用**代码结构**保证这一点，
而不是靠自觉：拿到首个事件之前的所有失败都进入 `continue` 换下一个 provider；
一旦拿到首个事件，后面的 yield 就再无回退分支。

## 三档超时

- `connect`：TCP/TLS 建连。本地 vLLM 用 2 秒——连不上应当立刻失败。
- `first_token`：**这是让故障切换「感觉瞬时」的关键旋钮。** 它远小于读超时，
  否则一个无响应的 provider 会让用户干等 60 秒。
- `read`：两个 token 之间的静默上限。

## 熔断

连续失败 N 次后打开，窗口内直接跳过（零延迟），窗口过后放一次半开探测。
流中断的失败同样计数——中途死三次的 provider 并不健康。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from bidding_qa_chatbot.config import Settings, get_settings
from bidding_qa_chatbot.llm.base import (
    AllProvidersFailed,
    Attempt,
    FatalProviderError,
    LLMProvider,
    LLMRequest,
    MidStreamError,
    ProviderError,
    RetryableProviderError,
    Stopwatch,
    StreamEnd,
    StreamEvent,
)
from bidding_qa_chatbot.schemas import FallbackEvent, ProviderEvent

logger = logging.getLogger(__name__)

#: 熔断器做成进程级单例。每次调用新建实例的话，失败计数永远归零，永远不会触发。
_BREAKER: CircuitBreaker | None = None


@dataclass
class _BreakerState:
    failures: int = 0
    opened_at: float = 0.0
    half_open: bool = False


@dataclass
class CircuitBreaker:
    threshold: int = 3
    open_seconds: float = 60.0
    _states: dict[str, _BreakerState] = field(default_factory=dict)

    def is_open(self, provider: str) -> bool:
        state = self._states.get(provider)
        if state is None or state.opened_at == 0.0:
            return False
        if time.monotonic() - state.opened_at >= self.open_seconds:
            # 窗口已过，放行一次半开探测
            state.half_open = True
            state.opened_at = 0.0
            return False
        return True

    def record_success(self, provider: str) -> None:
        self._states[provider] = _BreakerState()

    def record_failure(self, provider: str) -> None:
        state = self._states.setdefault(provider, _BreakerState())
        if state.half_open:
            # 半开探测又失败，重新打开
            state.half_open = False
            state.opened_at = time.monotonic()
            state.failures = self.threshold
            logger.warning("provider %s 半开探测失败，熔断器重新打开", provider)
            return
        state.failures += 1
        if state.failures >= self.threshold:
            state.opened_at = time.monotonic()
            logger.warning(
                "provider %s 连续失败 %d 次，熔断 %.0f 秒",
                provider,
                state.failures,
                self.open_seconds,
            )

    def snapshot(self) -> dict[str, dict[str, float | int | bool]]:
        return {
            name: {
                "failures": s.failures,
                "open": s.opened_at > 0.0,
                "half_open": s.half_open,
            }
            for name, s in self._states.items()
        }

    def reset(self) -> None:
        self._states.clear()


def get_breaker(settings: Settings | None = None) -> CircuitBreaker:
    global _BREAKER
    settings = settings or get_settings()
    if _BREAKER is None:
        _BREAKER = CircuitBreaker(
            threshold=settings.llm_breaker_failures,
            open_seconds=float(settings.llm_breaker_open_seconds),
        )
    return _BREAKER


class ProviderChain:
    def __init__(
        self,
        providers: list[LLMProvider],
        settings: Settings | None = None,
        breaker: CircuitBreaker | None = None,
        *,
        fatal_falls_back: bool = True,
    ) -> None:
        self.providers = providers
        self.settings = settings or get_settings()
        self.breaker = breaker or get_breaker(self.settings)
        #: 401/400 默认也降级：需求要求「调用失败时自动回退」，
        #: 一个失效的 key 不该让用户拿不到答案。错误仍会记录并告警。
        self.fatal_falls_back = fatal_falls_back

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.providers]

    # ------------------------------------------------------------ 流式
    async def stream_with_fallback(
        self, req: LLMRequest
    ) -> AsyncIterator[StreamEvent | StreamEnd | ProviderEvent | FallbackEvent]:
        if not self.providers:
            raise AllProvidersFailed([])

        attempts: list[Attempt] = []
        primary = self.providers[0].name

        for index, provider in enumerate(self.providers):
            if self.breaker.is_open(provider.name):
                attempts.append(
                    Attempt(
                        provider=provider.name,
                        model=provider.model_for(req.mode),
                        phase="connect",
                        error_type="CircuitOpen",
                        message="熔断器打开，本次跳过",
                        elapsed_ms=0,
                    )
                )
                continue

            watch = Stopwatch()
            stream = provider.astream(req)

            # ---- 首 token 边界 ----
            try:
                first = await asyncio.wait_for(
                    anext(stream), timeout=self._first_token_timeout(req)
                )
            except StopAsyncIteration:
                # provider 一个事件都没产出就结束了，视为可重试失败
                attempts.append(
                    Attempt(
                        provider.name,
                        provider.model_for(req.mode),
                        "first_token",
                        "EmptyStream",
                        "未产生任何内容",
                        watch.elapsed_ms,
                    )
                )
                self.breaker.record_failure(provider.name)
                await _close(stream)
                continue
            except TimeoutError:
                attempts.append(
                    Attempt(
                        provider.name,
                        provider.model_for(req.mode),
                        "first_token",
                        "TimeoutError",
                        f"首 token 超时（{self._first_token_timeout(req):.0f}s）",
                        watch.elapsed_ms,
                    )
                )
                self.breaker.record_failure(provider.name)
                await _close(stream)
                continue
            except Exception as exc:  # noqa: BLE001
                error = _as_provider_error(provider.name, exc)
                attempts.append(
                    Attempt(
                        provider.name,
                        provider.model_for(req.mode),
                        "first_token",
                        type(error).__name__,
                        error.message,
                        watch.elapsed_ms,
                    )
                )
                self.breaker.record_failure(provider.name)
                await _close(stream)
                if isinstance(error, FatalProviderError) and not self.fatal_falls_back:
                    raise AllProvidersFailed(attempts) from exc
                logger.warning("provider %s 首 token 前失败: %s", provider.name, error.message)
                continue

            # ---- 已拿到首个事件：此后不再回退 ----
            self.breaker.record_success(provider.name)
            model = provider.model_for(req.mode)

            if index > 0:
                yield FallbackEvent(
                    from_provider=primary,
                    to_provider=provider.name,
                    reason=attempts[-1].error_type if attempts else "unknown",
                    attempts=[a.to_model() for a in attempts],
                )
            yield ProviderEvent(
                provider=provider.name,
                model=model,
                mode=req.mode,
                thinking_enabled=req.mode == "thinking",
            )

            emitted = 0
            try:
                async for event in _prepend(first, stream):
                    if isinstance(event, StreamEvent):
                        emitted += 1
                    yield event
            except Exception as exc:  # noqa: BLE001
                error = _as_provider_error(provider.name, exc)
                self.breaker.record_failure(provider.name)
                # 已经吐过字就绝不能重试，否则用户会看到重复内容。
                raise MidStreamError(provider.name, error.message, attempts) from exc
            finally:
                await _close(stream)

            if emitted == 0:
                # 只有 StreamEnd、没有任何 token：当作失败处理，但文本已无法撤回，
                # 因此仍走 MidStreamError 语义（partial 但没有内容）。
                raise MidStreamError(provider.name, "provider 未产出任何内容", attempts)
            return

        raise AllProvidersFailed(attempts)

    def _first_token_timeout(self, req: LLMRequest) -> float:
        base = self.settings.llm_first_token_timeout
        # 思考模式首 token 天然更慢，给更宽的预算。
        return base * 2.5 if req.mode == "thinking" else base

    # ------------------------------------------------------------ 一次性
    async def complete(self, req: LLMRequest):
        attempts: list[Attempt] = []
        last_error: Exception | None = None

        for provider in self.providers:
            if self.breaker.is_open(provider.name):
                continue
            watch = Stopwatch()
            try:
                result = await provider.acomplete(req)
                self.breaker.record_success(provider.name)
                return result
            except Exception as exc:  # noqa: BLE001
                error = _as_provider_error(provider.name, exc)
                last_error = error
                attempts.append(
                    Attempt(
                        provider.name,
                        provider.model_for(req.mode),
                        "connect",
                        type(error).__name__,
                        error.message,
                        watch.elapsed_ms,
                    )
                )
                self.breaker.record_failure(provider.name)
                if isinstance(error, FatalProviderError) and not self.fatal_falls_back:
                    break
        raise AllProvidersFailed(attempts) from last_error

    async def aclose(self) -> None:
        for provider in self.providers:
            with contextlib.suppress(Exception):
                await provider.aclose()


async def _prepend(first, stream) -> AsyncIterator:
    yield first
    async for event in stream:
        yield event


async def _close(stream) -> None:
    """确定性关闭上游流，避免半截流泄漏连接、继续计费。"""
    with contextlib.suppress(Exception):
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()


def _as_provider_error(provider: str, exc: Exception) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return RetryableProviderError(provider, "超时")
    if isinstance(exc, (ConnectionError, OSError)):
        return RetryableProviderError(provider, f"{type(exc).__name__}: {exc}")
    return RetryableProviderError(provider, f"{type(exc).__name__}: {exc}")
