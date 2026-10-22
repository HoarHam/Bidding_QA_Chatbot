"""OpenAI 兼容协议 provider：DeepSeek / vLLM / Ollama。

**只有这一个文件导入 `openai` SDK。** 这样将来把 openai 从 2.x 升到 3.x 时，
改动面被限制在这里（3.x 改用 httpx2，respx 无法再拦截，测试策略要一起换）。

两个刻意的选择：

1. **`max_retries=0`**：SDK 自带的重试会与回退链的重试策略叠加，
   更糟的是它可能在**流已经开始之后**重试，白白吃掉首 token 的时限预算。
   重试策略只允许存在于 `chain.py` 一处。
2. **显式关闭流**：`openai` 的 `AsyncStream` 提供的是 `close()` 而不是 `aclose()`，
   写成 `aclosing()` 会抛 `AttributeError`。这里用自己的异步生成器包一层，
   `finally` 里 `await stream.close()`，保证被放弃的半截流不会泄漏上游连接。
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from bidding_qa_chatbot.config import Settings, get_settings
from bidding_qa_chatbot.llm.base import (
    FatalProviderError,
    LLMRequest,
    LLMResult,
    RetryableProviderError,
    Stopwatch,
    StreamEnd,
    StreamEvent,
    ToolCall,
    Usage,
)
from bidding_qa_chatbot.llm.dsml import DSMLStreamFilter, strip_dsml
from bidding_qa_chatbot.schemas import DepHealth

logger = logging.getLogger(__name__)

#: 这些状态码重试没有意义（配置/鉴权问题）
_FATAL_STATUS = {400, 401, 403, 404, 422}


def _classify(provider: str, exc: Exception) -> Exception:
    """把 SDK 异常映射成可重试 / 致命两类。"""
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)

    if status is not None:
        if status in _FATAL_STATUS:
            hint = "请检查 API key" if status in (401, 403) else "请检查请求参数"
            return FatalProviderError(provider, f"HTTP {status}，{hint}", status_code=status)
        return RetryableProviderError(provider, f"HTTP {status}", status_code=status)

    name = type(exc).__name__
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError)):
        return RetryableProviderError(provider, f"{name}: {exc}")
    return RetryableProviderError(provider, f"{name}: {exc}")


class OpenAICompatProvider:
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        settings: Settings | None = None,
        *,
        supports_thinking_toggle: bool = False,
        thinking_style: str = "generic",
        connect_timeout: float | None = None,
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.settings = settings or get_settings()
        #: 是否支持用请求参数开关思考模式（DeepSeek 支持；vLLM 上的推理模型不需要）
        self.supports_thinking_toggle = supports_thinking_toggle
        self.thinking_style = thinking_style
        self._connect_timeout = connect_timeout or self.settings.llm_connect_timeout
        self._client: Any = None

    # ------------------------------------------------------------ 内部
    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                max_retries=0,  # 重试统一由 chain 负责
                timeout=httpx.Timeout(
                    connect=self._connect_timeout,
                    read=self.settings.llm_read_timeout,
                    write=30.0,
                    pool=10.0,
                ),
            )
        return self._client

    def model_for(self, mode: str) -> str:
        if self.name == "vllm":
            # vLLM 的 model 字段是**服务端文件系统路径**，必须原样透传。
            return self.settings.vllm_model
        if self.name == "ollama":
            return self.settings.ollama_model
        return self.settings.model_for(self.name, mode)

    def _request_kwargs(self, req: LLMRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_for(req.mode),
            "messages": [m.to_dict() for m in req.messages],
        }
        thinking = req.mode == "thinking"

        # 只在真的注册了工具时才带 tools：部分部署对空数组会直接报错。
        if req.tools:
            kwargs["tools"] = req.tools
            kwargs["tool_choice"] = "auto"

        if self.thinking_style == "deepseek" and self.supports_thinking_toggle:
            # DeepSeek 默认**开启**思考；快速模式必须显式关掉，
            # 否则每次「快速」回答都要付推理延迟和费用。
            kwargs["extra_body"] = {"thinking": {"type": "enabled" if thinking else "disabled"}}

        # 思考模式下 DeepSeek 会拒绝 temperature / top_p，因此只在非思考模式下传。
        if not thinking:
            if req.temperature is not None:
                kwargs["temperature"] = req.temperature
            if req.max_tokens is not None:
                kwargs["max_tokens"] = req.max_tokens
        elif req.max_tokens is not None:
            # 思考模式仍可限制输出长度（影响的是正文，不含思考过程）
            kwargs["max_tokens"] = req.max_tokens

        return kwargs

    # ------------------------------------------------------------ 流式
    async def astream(self, req: LLMRequest) -> AsyncIterator[StreamEvent | StreamEnd]:
        client = self._ensure_client()
        kwargs = self._request_kwargs(req)

        try:
            stream = await client.chat.completions.create(stream=True, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise _classify(self.name, exc) from exc

        emitted = 0
        usage: Usage | None = None
        finish_reason = "stop"
        # tool_call 的分片是分批到达的：id 与 name 通常一次给全，arguments 是逐段拼的。
        tool_buffer: dict[int, dict[str, str]] = {}
        reasoning_parts: list[str] = []
        # DeepSeek 可能把工具调用以 DSML 标记混在正文/思维内容里流式返回。
        # 过滤器负责不外泄标记，并把可识别的调用还原成 ToolCall。
        content_filter = DSMLStreamFilter()
        reasoning_filter = DSMLStreamFilter()
        dsml_calls: list[ToolCall] = []

        try:
            async for chunk in stream:
                if not chunk.choices:
                    # 最后一个 chunk 只带 usage，不带 choices
                    if getattr(chunk, "usage", None):
                        usage = _to_usage(chunk.usage)
                    continue

                delta = chunk.choices[0].delta
                # reasoning_content 不在类型签名里，只能 getattr 取。
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    visible, calls = reasoning_filter.feed(reasoning)
                    dsml_calls.extend(calls)
                    if visible:
                        reasoning_parts.append(visible)
                        emitted += 1
                        yield StreamEvent(kind="reasoning", text=visible)

                if delta.content:
                    visible, calls = content_filter.feed(delta.content)
                    dsml_calls.extend(calls)
                    if visible:
                        emitted += 1
                        yield StreamEvent(kind="content", text=visible)

                for tc in getattr(delta, "tool_calls", None) or []:
                    idx = getattr(tc, "index", 0) or 0
                    entry = tool_buffer.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if getattr(tc, "id", None):
                        entry["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        # name 一次性给全，赋值而非累加，避免个别实现重发导致重复
                        if getattr(fn, "name", None) and not entry["name"]:
                            entry["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            entry["arguments"] += fn.arguments

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
                if getattr(chunk, "usage", None):
                    usage = _to_usage(chunk.usage)
        except Exception as exc:  # noqa: BLE001
            if emitted == 0:
                raise _classify(self.name, exc) from exc
            raise
        finally:
            with contextlib.suppress(Exception):
                await stream.close()

        # 冲刷过滤器里暂存的尾部（可能包含跨 chunk 拆分的标记）
        tail_reasoning, calls = reasoning_filter.finish()
        dsml_calls.extend(calls)
        if tail_reasoning:
            reasoning_parts.append(tail_reasoning)
            emitted += 1
            yield StreamEvent(kind="reasoning", text=tail_reasoning)

        tail_content, calls = content_filter.finish()
        dsml_calls.extend(calls)
        if tail_content:
            emitted += 1
            yield StreamEvent(kind="content", text=tail_content)

        yield StreamEnd(
            finish_reason=finish_reason,
            usage=usage,
            provider=self.name,
            model=self.model_for(req.mode),
            tool_calls=_dedupe_tool_calls(
                [
                    ToolCall(
                        id=e["id"] or f"call_{i}",
                        name=e["name"],
                        arguments=e["arguments"],
                    )
                    for i, e in sorted(tool_buffer.items())
                    if e["name"]
                ]
                + dsml_calls
            ),
            reasoning="".join(reasoning_parts) or None,
        )

    # ------------------------------------------------------------ 一次性
    async def acomplete(self, req: LLMRequest) -> LLMResult:
        client = self._ensure_client()
        try:
            resp = await client.chat.completions.create(stream=False, **self._request_kwargs(req))
        except Exception as exc:  # noqa: BLE001
            raise _classify(self.name, exc) from exc

        choice = resp.choices[0]
        message = choice.message
        return LLMResult(
            # 非流式响应也可能夹带 DSML 标记：至少保证不下发原始标记
            text=strip_dsml(message.content or ""),
            reasoning=strip_dsml(getattr(message, "reasoning_content", None)),
            finish_reason=choice.finish_reason or "stop",
            usage=_to_usage(getattr(resp, "usage", None)),
            provider=self.name,
            model=self.model_for(req.mode),
            tool_calls=_extract_tool_calls(message),
        )

    # ------------------------------------------------------------ 其他
    async def aprobe(self) -> DepHealth:
        """`/models` 探针不消耗 token。"""
        watch = Stopwatch()
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(
                    f"{self._base_url}/models", headers={"Authorization": f"Bearer {self._api_key}"}
                )
            latency = watch.elapsed_ms
            if resp.status_code in (401, 403):
                return DepHealth(
                    name=self.name,
                    status="unavailable",
                    latency_ms=latency,
                    message=f"认证失败 ({resp.status_code})",
                )
            if resp.status_code >= 500:
                return DepHealth(
                    name=self.name,
                    status="degraded",
                    latency_ms=latency,
                    message=f"HTTP {resp.status_code}",
                )
            return DepHealth(name=self.name, status="ok", latency_ms=latency)
        except Exception as exc:  # noqa: BLE001
            return DepHealth(name=self.name, status="unavailable", message=f"{type(exc).__name__}")

    async def aclose(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.close()
            self._client = None


def _dedupe_tool_calls(calls: list[ToolCall]) -> list[ToolCall]:
    """结构化 tool_calls 与 DSML 还原结果可能重复，按 (name, arguments) 去重。"""
    seen: set[tuple[str, str]] = set()
    unique: list[ToolCall] = []
    for call in calls:
        key = (call.name, call.arguments)
        if key in seen:
            continue
        seen.add(key)
        unique.append(call)
    return unique


def _extract_tool_calls(message: Any) -> list[ToolCall]:
    """从非流式响应的 message 里取出 tool_calls。"""
    out: list[ToolCall] = []
    for tc in getattr(message, "tool_calls", None) or []:
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        out.append(
            ToolCall(
                id=getattr(tc, "id", "") or f"call_{len(out)}",
                name=getattr(fn, "name", "") or "",
                arguments=getattr(fn, "arguments", "") or "",
            )
        )
    return out


def _to_usage(raw: Any) -> Usage | None:
    if raw is None:
        return None
    return Usage(
        prompt_tokens=getattr(raw, "prompt_tokens", None),
        completion_tokens=getattr(raw, "completion_tokens", None),
        total_tokens=getattr(raw, "total_tokens", None),
    )


def _json_loads(raw: str) -> dict:
    return json.loads(raw)
