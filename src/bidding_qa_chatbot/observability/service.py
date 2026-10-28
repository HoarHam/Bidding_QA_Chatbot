"""Langfuse 可观测性：一次问答的 trace 与用户反馈打分。

## 三条硬约束

1. **可观测性失败绝不打断问答。** 未配置密钥、SDK 未安装、网络不可达、上报格式
   报错——全部降级为 no-op，只在日志里记一次。用户的问题不能因为追踪系统挂了而失败。
2. **内容是否出本机由 `LANGFUSE_CAPTURE_CONTENT` 决定。** 关闭时只上报长度/条数，
   prompt、回答、工具结果正文都不会离开进程。
3. **采样只影响 span。** 反馈打分是业务数据，不能因为 `LANGFUSE_SAMPLE_RATE` 被丢掉。
   因此采样在本地按 trace 决定，SDK 自身的 sample_rate 固定为 1.0。

## 为什么用显式 observation，而不是 `@observe` / 上下文管理器

问答是异步生成器（Agent 循环、SSE 转发），`with` 的作用域跨不过 `yield`。
这里直接持有 Langfuse 的 observation 对象、显式 `start/update/end`，
并用显式父 observation 建立层级，不依赖当前 OTel context 的隐式栈。
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.log import redact

logger = logging.getLogger(__name__)

#: 单次进程内同一类上报错误只告警一次，避免 Langfuse 不可达时每个 token 刷屏。
_MAX_REPORTED_FAILURES = 32


def _content_payload(value: Any, *, capture: bool, max_chars: int) -> Any:
    """按内容开关准备要上报的 payload。

    capture=True：递归截断超长字符串（附件 OCR 可达上万字）。
    capture=False：只保留结构（键名、role、列表长度）与字符数。
    """
    if capture:
        return _truncate(value, max_chars)
    return _summarize(value)


def _truncate(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + "…[truncated]"
    if isinstance(value, dict):
        return {str(k): _truncate(v, max_chars) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_truncate(v, max_chars) for v in value]
    return value


def _summarize(value: Any) -> Any:
    if isinstance(value, str):
        return {"chars": len(value)}
    if isinstance(value, dict):
        return {str(k): _summarize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize(v) for v in value]
    return value


def _safe_metadata(metadata: dict[str, Any] | None, *, max_chars: int) -> dict[str, Any] | None:
    if not metadata:
        return None
    return {str(k): _truncate(v, max_chars) for k, v in metadata.items()}


class _FailureReporter:
    """吞掉 Langfuse SDK 的异常，并以限频方式记录。"""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def report(self, operation: str, exc: BaseException) -> None:
        message = redact(f"{type(exc).__name__}: {exc}")[:300]
        if operation not in self._seen and len(self._seen) < _MAX_REPORTED_FAILURES:
            self._seen.add(operation)
            logger.warning("Langfuse 上报失败（%s，不影响业务）: %s", operation, message)
        else:
            logger.debug("Langfuse 上报失败（%s）: %s", operation, message)


@dataclass(slots=True)
class TraceHandle:
    """一个 trace 或 observation 的轻量句柄。

    `observation is None` 时所有方法都是无操作。子 span 仍然携带同一个
    `trace_id`，因此即使采样把根 span 丢了，用户反馈仍能打到这个 trace 上。
    """

    observation: Any | None = None
    trace_id: str | None = None
    capture_content: bool = True
    max_content_chars: int = 12000
    user_id: str | None = None
    session_id: str | None = None
    reporter: _FailureReporter = field(default_factory=_FailureReporter)
    _ended: bool = False

    @classmethod
    def noop(cls) -> TraceHandle:
        return cls()

    @property
    def enabled(self) -> bool:
        return self.observation is not None and not self._ended

    def _noop_child(self) -> TraceHandle:
        return TraceHandle(
            trace_id=self.trace_id,
            capture_content=self.capture_content,
            max_content_chars=self.max_content_chars,
            user_id=self.user_id,
            session_id=self.session_id,
            reporter=self.reporter,
        )

    def _child(self, *, name: str, as_type: str, input: Any, metadata: Any, model: str | None):
        if self.observation is None or self._ended:
            return self._noop_child()
        try:
            # 显式父 observation 负责层级；propagate_attributes 让 user/session
            # 也落到子 observation 上，便于按用户/会话聚合成本。
            from langfuse import propagate_attributes

            with propagate_attributes(user_id=self.user_id, session_id=self.session_id):
                observation = self.observation.start_observation(
                    name=name,
                    as_type=as_type,
                    input=_content_payload(
                        input, capture=self.capture_content, max_chars=self.max_content_chars
                    ),
                    metadata=_safe_metadata(metadata, max_chars=self.max_content_chars),
                    model=model,
                )
            return TraceHandle(
                observation=observation,
                trace_id=self.trace_id,
                capture_content=self.capture_content,
                max_content_chars=self.max_content_chars,
                user_id=self.user_id,
                session_id=self.session_id,
                reporter=self.reporter,
            )
        except Exception as exc:  # noqa: BLE001 — 追踪失败不能冒泡到业务
            self.reporter.report(f"start_{as_type}", exc)
            return self._noop_child()

    def start_span(
        self, name: str, *, input: Any = None, metadata: dict[str, Any] | None = None
    ) -> TraceHandle:
        return self._child(name=name, as_type="span", input=input, metadata=metadata, model=None)

    def start_generation(
        self,
        name: str,
        *,
        model: str | None = None,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceHandle:
        return self._child(
            name=name, as_type="generation", input=input, metadata=metadata, model=model
        )

    def start_tool(
        self, name: str, *, input: Any = None, metadata: dict[str, Any] | None = None
    ) -> TraceHandle:
        return self._child(name=name, as_type="tool", input=input, metadata=metadata, model=None)

    def update(
        self,
        *,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
        usage_details: dict[str, int] | None = None,
        level: str | None = None,
        status_message: str | None = None,
        completion_start_time: Any = None,
    ) -> None:
        if self.observation is None or self._ended:
            return
        kwargs: dict[str, Any] = {}
        if output is not None:
            kwargs["output"] = _content_payload(
                output, capture=self.capture_content, max_chars=self.max_content_chars
            )
        if metadata:
            kwargs["metadata"] = _safe_metadata(metadata, max_chars=self.max_content_chars)
        if model:
            kwargs["model"] = model
        if usage_details:
            kwargs["usage_details"] = {k: int(v) for k, v in usage_details.items() if v is not None}
        if level:
            kwargs["level"] = level
        if status_message:
            kwargs["status_message"] = redact(str(status_message))[:500]
        if completion_start_time is not None:
            kwargs["completion_start_time"] = completion_start_time
        if not kwargs:
            return
        try:
            self.observation.update(**kwargs)
        except Exception as exc:  # noqa: BLE001
            self.reporter.report("update_observation", exc)

    def end(
        self,
        *,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
        usage_details: dict[str, int] | None = None,
        level: str | None = None,
        status_message: str | None = None,
        completion_start_time: Any = None,
    ) -> None:
        if self.observation is None or self._ended:
            return
        self.update(
            output=output,
            metadata=metadata,
            model=model,
            usage_details=usage_details,
            level=level,
            status_message=status_message,
            completion_start_time=completion_start_time,
        )
        self._ended = True
        try:
            self.observation.end()
        except Exception as exc:  # noqa: BLE001
            self.reporter.report("end_observation", exc)


class NoopObservability:
    """未启用时的实现：所有方法直接返回，业务代码无需到处写 if。"""

    enabled = False

    def __init__(self, reason: str = "未启用") -> None:
        self.reason = reason

    def start_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        query: str,
        mode: str,
        web_search: bool,
        has_attachments: bool,
        regenerate: bool = False,
        endpoint: str = "stream",
    ) -> TraceHandle:
        return TraceHandle.noop()

    def score_feedback(self, *, trace_id: str | None, message_id: str, rating: str | None) -> None:
        return None

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class LangfuseObservability:
    """Langfuse SDK 的薄封装。SDK 只在这里出现，其他模块不依赖它。"""

    enabled = True

    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
        *,
        sampler: Callable[[], float] | None = None,
    ) -> None:
        self.settings = settings
        self.reason = ""
        self._client = client if client is not None else _create_client(settings)
        self._reporter = _FailureReporter()
        self._sampler = sampler or random.random

    def start_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        query: str,
        mode: str,
        web_search: bool,
        has_attachments: bool,
        regenerate: bool = False,
        endpoint: str = "stream",
    ) -> TraceHandle:
        trace_id: str | None = None
        try:
            trace_id = self._client.create_trace_id()
        except Exception as exc:  # noqa: BLE001
            self._reporter.report("create_trace_id", exc)
            return TraceHandle.noop()

        # 采样在本地做：SDK 的 sample_rate 保持 1.0，避免把反馈打分一起采样掉。
        if self._sampler() >= self.settings.langfuse_sample_rate:
            return TraceHandle(
                trace_id=trace_id,
                capture_content=self.settings.langfuse_capture_content,
                max_content_chars=self.settings.langfuse_max_content_chars,
                user_id=str(user_id),
                session_id=str(conversation_id),
                reporter=self._reporter,
            )

        try:
            from langfuse import propagate_attributes

            with propagate_attributes(
                user_id=str(user_id),
                session_id=str(conversation_id),
                trace_name="chat",
                environment=self.settings.langfuse_environment,
                tags=["chat", mode],
            ):
                root = self._client.start_observation(
                    name="chat.turn",
                    as_type="span",
                    trace_context={"trace_id": trace_id},
                    input=_content_payload(
                        {"query": query},
                        capture=self.settings.langfuse_capture_content,
                        max_chars=self.settings.langfuse_max_content_chars,
                    ),
                    metadata={
                        "endpoint": endpoint,
                        "mode": mode,
                        "web_search": web_search,
                        "has_attachments": has_attachments,
                        "regenerate": regenerate,
                        "agent_enabled": self.settings.agent_enabled,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            self._reporter.report("start_turn", exc)
            # 即使根 span 建不起来，也保留 trace_id：用户反馈还能打到一个
            # 只有分数的 trace 上，不会被静默丢弃。
            return TraceHandle(
                trace_id=trace_id,
                capture_content=self.settings.langfuse_capture_content,
                max_content_chars=self.settings.langfuse_max_content_chars,
                user_id=str(user_id),
                session_id=str(conversation_id),
                reporter=self._reporter,
            )

        return TraceHandle(
            observation=root,
            trace_id=trace_id,
            capture_content=self.settings.langfuse_capture_content,
            max_content_chars=self.settings.langfuse_max_content_chars,
            user_id=str(user_id),
            session_id=str(conversation_id),
            reporter=self._reporter,
        )

    def score_feedback(self, *, trace_id: str | None, message_id: str, rating: str | None) -> None:
        """把 👍/👎 写成 trace 上的数值分。

        `score_id` 固定为 `user-feedback-{message_id}`：Langfuse 的 scores API
        按 id upsert，用户反复切换评价时不会堆出一串互相矛盾的历史分数。
        up=1 / down=-1 / 清除=0。
        """
        if not trace_id:
            return
        values = {"up": 1.0, "down": -1.0, None: 0.0}
        if rating not in values:
            return
        try:
            self._client.create_score(
                name="user_feedback",
                value=values[rating],
                trace_id=trace_id,
                score_id=f"user-feedback-{message_id}",
                data_type="NUMERIC",
                comment="1=👍, -1=👎, 0=清除评价",
                metadata={"message_id": message_id},
                environment=self.settings.langfuse_environment,
            )
        except Exception as exc:  # noqa: BLE001
            self._reporter.report("create_score", exc)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as exc:  # noqa: BLE001
            self._reporter.report("flush", exc)

    def shutdown(self) -> None:
        """停止后台导出线程并 flush。只在进程退出时调用。"""
        try:
            shutdown = getattr(self._client, "shutdown", None)
            if shutdown is not None:
                shutdown()
            else:
                self._client.flush()
        except Exception as exc:  # noqa: BLE001
            self._reporter.report("shutdown", exc)


def _create_client(settings: Settings) -> Any:
    """构造 Langfuse SDK 客户端。

    **不要走系统代理**：本机的 127.0.0.1:10808 代理连不通境外 Langfuse，
    与 Qdrant 的坑完全相同。SDK 默认的 httpx.Client 会读取代理环境变量，
    所以这里显式注入一个 `trust_env` 可配置的客户端。
    """
    import httpx
    from langfuse import Langfuse

    assert settings.langfuse_public_key is not None
    assert settings.langfuse_secret_key is not None

    timeout = max(1, int(settings.langfuse_timeout))
    http_client = httpx.Client(
        timeout=httpx.Timeout(settings.langfuse_timeout, connect=settings.langfuse_timeout),
        follow_redirects=True,
        trust_env=settings.http_trust_env,
    )
    return Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        timeout=timeout,
        environment=settings.langfuse_environment,
        # 采样由本模块按 trace 决定（见 start_turn），SDK 侧保持全量，
        # 这样用户反馈的 score 不会跟着 span 一起被采样掉。
        sample_rate=1.0,
        tracing_enabled=True,
        httpx_client=http_client,
    )


def build_observability(settings: Settings) -> NoopObservability | LangfuseObservability:
    """按配置构造可观测性实现；任何异常都降级为 no-op。"""
    if not settings.langfuse_enabled:
        return NoopObservability("LANGFUSE_ENABLED=false")
    if not settings.langfuse_configured:
        return NoopObservability("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 未配置")

    try:
        return LangfuseObservability(settings)
    except Exception as exc:  # noqa: BLE001 — 追踪永远不能阻止服务启动
        logger.warning(
            "Langfuse 初始化失败，可观测性关闭: %s", redact(f"{type(exc).__name__}: {exc}")
        )
        return NoopObservability("初始化失败")
