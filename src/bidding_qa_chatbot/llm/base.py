"""Provider 无关的 LLM 类型与协议。

统一 `StreamEvent` 是这里最重要的设计决定：SSE 层、回退链、前端都**不需要知道**
token 出自哪家厂商。「深度思考」靠 `kind` 区分，而不是靠模型名——
这一点在本项目里尤其关键，因为 `DEEPSEEK_MODEL` 和 `DEEPSEEK_THINKING_MODEL`
当前是**同一个值**（`deepseek-flash`），模式是请求参数而非模型名。
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from bidding_qa_chatbot.log import redact

Mode = Literal["fast", "thinking"]
EventKind = Literal["reasoning", "content"]
FailPhase = Literal["connect", "first_token", "mid_stream"]


# ---------------------------------------------------------------- 数据结构
@dataclass(slots=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def to_model(self):
        from bidding_qa_chatbot.schemas import Usage as UsageModel

        return UsageModel(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
        )


@dataclass(slots=True)
class ToolCall:
    """模型请求调用某个工具。`arguments` 是模型给的**原始 JSON 字符串**，
    可能不合法，解析失败要作为工具错误回灌给模型，而不是抛异常。"""

    id: str
    name: str
    arguments: str = ""

    def parsed_arguments(self) -> dict[str, Any]:
        if not self.arguments.strip():
            return {}
        import json

        try:
            data = json.loads(self.arguments)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


@dataclass(slots=True)
class ChatMessage:
    """对话消息。

    assistant 消息可以带 `tool_calls`；tool 消息用 `tool_call_id` 回指是哪次调用。
    `reasoning_content` 只在 DeepSeek 上必须**原样回传**——
    带 `tools` 的请求若丢掉它会被服务端拒绝（400）。
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}

        if self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments or "{}"},
                }
                for tc in self.tool_calls
            ]
            # DeepSeek 要求：带 tool_calls 的 assistant 消息必须回传 reasoning_content
            if self.reasoning_content:
                out["reasoning_content"] = self.reasoning_content

        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id

        return out


@dataclass(slots=True)
class StreamEvent:
    """增量事件。`reasoning` 是思考过程，`content` 是正文。"""

    kind: EventKind
    text: str


@dataclass(slots=True)
class StreamEnd:
    """流正常结束。携带收尾信息，否则 `finish_reason=length`（被截断）
    和 token 用量就无从得知。

    工具调用也在这里交付：provider 在流式过程中把 tool_call 的分片
    （id / name / arguments 是分批到达的）累积起来，结束时一次性给出。
    这样上层不必关心分片拼装的细节。
    """

    finish_reason: str
    usage: Usage | None = None
    provider: str | None = None
    model: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str | None = None


@dataclass(slots=True)
class LLMRequest:
    messages: list[ChatMessage]
    mode: Mode = "fast"
    temperature: float | None = None
    max_tokens: int | None = None
    #: OpenAI 风格的 function calling 声明。为空时不带该参数——
    #: 有些模型/部署带空 tools 数组反而会报错。
    tools: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class LLMResult:
    text: str
    reasoning: str | None
    finish_reason: str
    usage: Usage | None
    provider: str
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(slots=True)
class Attempt:
    """一次失败的尝试。会一路带到 SSE 的 fallback/error 事件和健康检查里。"""

    provider: str
    model: str
    phase: FailPhase
    error_type: str
    message: str
    elapsed_ms: int

    def to_model(self):
        from bidding_qa_chatbot.schemas import AttemptModel

        return AttemptModel(
            provider=self.provider,
            model=self.model,
            phase=self.phase,
            error_type=self.error_type,
            # 消息可能来自底层库，务必再过一次脱敏。
            message=redact(self.message),
            elapsed_ms=self.elapsed_ms,
        )


# ---------------------------------------------------------------- 异常
class ProviderError(Exception):
    """provider 调用失败。

    构造时就对消息脱敏——异常最容易在 traceback 里把 key 带出去。
    """

    fatal = False

    def __init__(self, provider: str, message: str, *, status_code: int | None = None) -> None:
        self.provider = provider
        self.status_code = status_code
        self.raw_message = message
        self.message = redact(message)
        super().__init__(f"{provider}: {self.message}")


class RetryableProviderError(ProviderError):
    """超时 / 连接失败 / 429 / 5xx——可重试，也可降级。"""


class FatalProviderError(ProviderError):
    """401 / 400 / 配置错误——重试没有意义，但**默认仍会降级**到下一个 provider。

    理由：需求要求「线上调用失败时自动回退」。若 DeepSeek 的 key 失效，
    用户仍应从智谱拿到答案，而不是看到报错。但错误会记进 Attempt 并打 WARNING 日志，
    不会被悄悄吞掉。
    """

    fatal = True


class AllProvidersFailed(Exception):
    """链路里所有 provider 都在首个 token 之前失败了。"""

    def __init__(self, attempts: list[Attempt]) -> None:
        self.attempts = attempts
        detail = (
            "; ".join(f"{a.provider}({a.phase}): {a.message}" for a in attempts)
            or "无可用 provider"
        )
        super().__init__(f"所有 LLM provider 均失败: {detail}")


class MidStreamError(Exception):
    """已经开始产出 token 之后才失败。

    此时**不能**换 provider 重试——用户已经看到了部分文字，重来会造成重复或矛盾。
    前端据此保留已显示内容并提示中断。
    """

    def __init__(self, provider: str, message: str, attempts: list[Attempt] | None = None) -> None:
        self.provider = provider
        self.message = redact(message)
        self.attempts = attempts or []
        super().__init__(f"{provider} 中途断开: {self.message}")


# ---------------------------------------------------------------- 协议
class LLMProvider(Protocol):
    name: str

    def model_for(self, mode: Mode) -> str: ...

    def astream(self, req: LLMRequest) -> AsyncIterator[StreamEvent | StreamEnd]: ...

    async def acomplete(self, req: LLMRequest) -> LLMResult: ...

    async def aprobe(self): ...

    async def aclose(self) -> None: ...


# ---------------------------------------------------------------- 计时
class Stopwatch:
    __slots__ = ("_started",)

    def __init__(self) -> None:
        self._started = time.perf_counter()

    @property
    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._started) * 1000)


@dataclass(slots=True)
class StreamState:
    """回退链在流式过程中的状态，便于诊断。"""

    provider: str = ""
    model: str = ""
    emitted: int = 0
    attempts: list[Attempt] = field(default_factory=list)
    fell_back: bool = False
