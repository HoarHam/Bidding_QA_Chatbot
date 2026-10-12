"""智谱 provider（zai-sdk）。

## 为什么用 `ZhipuAiClient` 而不是 `ZaiClient`

实测两个类的默认 base_url 不同：

    ZaiClient.default_base_url     = https://api.z.ai/api/paas/v4      ← 国际站
    ZhipuAiClient.default_base_url = https://open.bigmodel.cn/api/paas/v4  ← 国内站

`.env` 里的 key 是 `{id}.{secret}` 形式的**国内站** key（SDK 内部按 `.` 切分并签
HS256 JWT）。用 `ZaiClient` 会静默打到国际站、鉴权失败。选 `ZhipuAiClient`
天然对齐，同时仍显式传 `base_url` 便于覆盖。

## 为什么要桥线程

`zai-sdk` **没有异步客户端**（实测：`_client.py` 只有 `BaseClient` / `ZaiClient` /
`ZhipuAiClient`，全部同步）。这里把同步流放进工作线程，用
`loop.call_soon_threadsafe` 把 chunk 推进 `asyncio.Queue`。

取消时**不能**在工作线程里 `await` 任何东西，也不能从事件循环线程去关闭响应
（可能阻塞在 httpcore 的锁上）。做法是设一个 `threading.Event`，
由工作线程自己发现并退出——这样回退链的首 token 超时依然是「瞬时」的。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import AsyncIterator
from typing import Any

import httpx

from bidding_qa_chatbot.config import Settings, get_settings
from bidding_qa_chatbot.llm.base import (
    FatalProviderError,
    LLMRequest,
    LLMResult,
    ProviderError,
    RetryableProviderError,
    StreamEnd,
    StreamEvent,
    ToolCall,
    Usage,
)
from bidding_qa_chatbot.schemas import DepHealth

logger = logging.getLogger(__name__)

_FATAL_STATUS = {400, 401, 403, 404, 422}


class ZhipuProvider:
    name = "zhipu"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: Any = None
        self._http_client: httpx.Client | None = None

    # ------------------------------------------------------------ 内部
    def _ensure_client(self) -> Any:
        if self._client is None:
            from zai import ZhipuAiClient

            # 复用同一个 httpx.Client 做连接池；zai-sdk 依赖 httpx>=0.23。
            self._http_client = httpx.Client(
                # httpx.Timeout 要么给默认值，要么四个参数全给；
                # 只给 connect/read 会直接抛 ValueError。
                timeout=httpx.Timeout(
                    self.settings.llm_read_timeout,
                    connect=self.settings.llm_connect_timeout,
                ),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
            self._client = ZhipuAiClient(
                api_key=(
                    self.settings.zhipu_api_key.get_secret_value()
                    if self.settings.zhipu_api_key
                    else ""
                ),
                base_url=self.settings.zhipu_base_url,
                max_retries=0,  # 重试统一由 chain 负责
                http_client=self._http_client,
            )
        return self._client

    def model_for(self, mode: str) -> str:
        return self.settings.model_for("zhipu", mode)

    def _request_kwargs(self, req: LLMRequest) -> dict[str, Any]:
        thinking_on = req.mode == "thinking"
        kwargs: dict[str, Any] = {
            "model": self.model_for(req.mode),
            "messages": [m.to_dict() for m in req.messages],
            # zai-sdk 只剔除 NOT_GIVEN，不剔除 None；若传 None，请求体里会出现
            # "thinking": null。所以两种模式都显式传值。
            "thinking": {"type": "enabled" if thinking_on else "disabled"},
        }
        if req.tools:
            kwargs["tools"] = req.tools
            kwargs["tool_choice"] = "auto"
        if req.max_tokens is not None:
            kwargs["max_tokens"] = req.max_tokens
        # 与 DeepSeek 同理：思考模式下不传采样参数。
        if not thinking_on and req.temperature is not None:
            kwargs["temperature"] = req.temperature
        return kwargs

    @staticmethod
    def _classify(exc: Exception) -> Exception:
        status = getattr(exc, "status_code", None)
        if status is None:
            status = getattr(getattr(exc, "response", None), "status_code", None)
        if status is not None:
            if status in _FATAL_STATUS:
                hint = "请检查 API key" if status in (401, 403) else "请检查请求参数"
                return FatalProviderError("zhipu", f"HTTP {status}，{hint}", status_code=status)
            return RetryableProviderError("zhipu", f"HTTP {status}", status_code=status)
        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
            return RetryableProviderError("zhipu", f"{type(exc).__name__}: {exc}")
        return RetryableProviderError("zhipu", f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------ 流式
    async def astream(self, req: LLMRequest) -> AsyncIterator[StreamEvent | StreamEnd]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        cancel = threading.Event()
        kwargs = self._request_kwargs(req)
        client = self._ensure_client()

        def _push(item: tuple[str, Any]) -> None:
            # 事件循环已关闭时静默丢弃，避免线程里抛异常。
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(queue.put_nowait, item)

        def worker() -> None:
            try:
                stream = client.chat.completions.create(stream=True, **kwargs)
                for chunk in stream:
                    if cancel.is_set():
                        break
                    _push(("chunk", chunk))
            except Exception as exc:  # noqa: BLE001
                _push(("error", exc))
            finally:
                _push(("done", None))

        thread = threading.Thread(target=worker, name="zhipu-stream", daemon=True)
        thread.start()

        emitted = 0
        usage: Usage | None = None
        finish_reason = "stop"
        tool_buffer: dict[int, dict[str, str]] = {}
        reasoning_parts: list[str] = []

        try:
            while True:
                kind, payload = await queue.get()

                if kind == "done":
                    break

                if kind == "error":
                    classified = self._classify(payload)
                    raise classified from payload

                chunk = payload
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    usage = _to_usage(chunk_usage)

                chunk_choices = getattr(chunk, "choices", None)
                if not chunk_choices:
                    continue

                choice = chunk_choices[0]
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                        emitted += 1
                        yield StreamEvent(kind="reasoning", text=reasoning)
                    content = getattr(delta, "content", None)
                    if content:
                        emitted += 1
                        yield StreamEvent(kind="content", text=content)

                    for tc in getattr(delta, "tool_calls", None) or []:
                        idx = getattr(tc, "index", 0) or 0
                        entry = tool_buffer.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if getattr(tc, "id", None):
                            entry["id"] = tc.id
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            if getattr(fn, "name", None) and not entry["name"]:
                                entry["name"] = fn.name
                            if getattr(fn, "arguments", None):
                                entry["arguments"] += fn.arguments

                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
        finally:
            # 让工作线程自行退出；不 join，避免阻塞事件循环。
            cancel.set()

        yield StreamEnd(
            finish_reason=finish_reason,
            usage=usage,
            provider=self.name,
            model=self.model_for(req.mode),
            tool_calls=[
                ToolCall(id=e["id"] or f"call_{i}", name=e["name"], arguments=e["arguments"])
                for i, e in sorted(tool_buffer.items())
                if e["name"]
            ],
            reasoning="".join(reasoning_parts) or None,
        )

    # ------------------------------------------------------------ 一次性
    async def acomplete(self, req: LLMRequest) -> LLMResult:
        """一次性返回。SDK 是同步的，整体丢进线程池即可。"""

        def _call() -> LLMResult:
            client = self._ensure_client()
            resp = client.chat.completions.create(stream=False, **self._request_kwargs(req))
            choice = resp.choices[0]
            message = choice.message
            calls: list[ToolCall] = []
            for tc in getattr(message, "tool_calls", None) or []:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                calls.append(
                    ToolCall(
                        id=getattr(tc, "id", "") or f"call_{len(calls)}",
                        name=getattr(fn, "name", "") or "",
                        arguments=getattr(fn, "arguments", "") or "",
                    )
                )
            return LLMResult(
                text=getattr(message, "content", "") or "",
                reasoning=getattr(message, "reasoning_content", None),
                finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
                usage=_to_usage(getattr(resp, "usage", None)),
                provider=self.name,
                model=self.model_for(req.mode),
                tool_calls=calls,
            )

        try:
            return await asyncio.to_thread(_call)
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise self._classify(exc) from exc

    # ------------------------------------------------------------ 其他
    async def aprobe(self) -> DepHealth:
        """zai-sdk 没有轻量探针接口，做 TCP+TLS 连通性检查即可。"""
        from urllib.parse import urlparse

        parsed = urlparse(self.settings.zhipu_base_url)
        host, port = parsed.hostname or "open.bigmodel.cn", parsed.port or 443
        try:
            _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=3.0)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            return DepHealth(name=self.name, status="ok")
        except Exception as exc:  # noqa: BLE001
            return DepHealth(name=self.name, status="unavailable", message=type(exc).__name__)

    async def aclose(self) -> None:
        self._client = None
        if self._http_client is not None:
            with contextlib.suppress(Exception):
                self._http_client.close()
            self._http_client = None


def _to_usage(raw: Any) -> Usage | None:
    if raw is None:
        return None
    return Usage(
        prompt_tokens=getattr(raw, "prompt_tokens", None),
        completion_tokens=getattr(raw, "completion_tokens", None),
        total_tokens=getattr(raw, "total_tokens", None),
    )
