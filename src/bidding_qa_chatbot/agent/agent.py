"""Agent 循环：由模型自行决定是否调用工具。

## 循环形态

    用户问题
      └─ 第 1 轮：带工具调用请求 →模型决定调用 search_bidding_knowledge
           ├─ 执行工具（检索知识库）
           └─ 把 assistant(tool_calls) + tool(结果) 追加进消息，进入下一轮
      └─ 第 N 轮：模型不再要求工具 → 产出最终回答（这一步是流式的）

## 两个刻意的设计

1. **最后一轮强制不带工具**（`tools=None`）。否则模型可能一直要求检索、
   永远不给答案，用户什么都拿不到。到了上限就必须作答。

2. **中间轮次的内容也照常流给用户**。模型有时会先来一句「让我查一下知识库」，
   这属于正常表达，直接透传比丢弃更自然；丢弃反而会让界面先卡一下再突然出字。
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from bidding_qa_chatbot.agent.base import ToolRegistry, ToolResult
from bidding_qa_chatbot.config import Settings, get_settings
from bidding_qa_chatbot.llm.base import (
    AllProvidersFailed,
    ChatMessage,
    LLMRequest,
    StreamEnd,
    StreamEvent,
)
from bidding_qa_chatbot.observability import (
    TraceHandle,
    messages_payload,
    tool_output_payload,
    usage_payload,
)
from bidding_qa_chatbot.schemas import FallbackEvent, ProviderEvent

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolCallNotice:
    """模型要求调用工具，转发给前端用于展示「正在检索…」。"""

    name: str
    arguments: dict[str, Any]
    display: str


@dataclass(slots=True)
class ToolResultNotice:
    name: str
    ok: bool
    elapsed_ms: int
    summary: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    #: 联网搜索来源（与本地知识库 sources 分开走，前端分区渲染）
    web_sources: list[dict[str, Any]] = field(default_factory=list)
    degraded: bool = False
    #: 回灌给模型的工具结果正文。前端/SSE 不使用；评测需要它作为 Ragas 的
    #: retrieved_contexts（图谱表格、知识库全文、联网摘要都在这里）。
    content: str = ""


@dataclass(slots=True)
class AgentDone:
    finish_reason: str
    steps: int
    tool_calls: int
    provider: str | None = None
    model: str | None = None
    usage: Any | None = None


AgentItem = (
    StreamEvent | StreamEnd | ProviderEvent | FallbackEvent | ToolCallNotice | ToolResultNotice
)


#: 工具名 → 给用户看的中文动作名。加新工具时在这里补一行。
_TOOL_DISPLAY_LABELS = {
    "search_bidding_knowledge": "检索知识库",
    "query_bidding_graph": "查询知识图谱",
    "search_web_keyword": "联网搜索（关键词）",
    "search_web_semantic": "联网搜索（语义）",
}

#: 展示用参数：按顺序取第一个非空值。`query` 必须排第一，
#: 否则既有检索工具在界面上的展示会退化。
_DISPLAY_ARG_KEYS = (
    "query",
    "subject_matter",
    "supplier",
    "purchaser",
    "agency",
    "location",
    "cypher",
)
_DISPLAY_VALUE_CHARS = 40


def _display_for(name: str, args: dict[str, Any]) -> str:
    """给用户看的一句话，避免把原始 JSON 直接怼到界面上。

    标签查 `_TOOL_DISPLAY_LABELS`（缺省「调用 {name}」）；
    展示参数按 `_DISPLAY_ARG_KEYS` 逐个取第一个非空值——按「参数角色」而不是
    按工具名硬编码，加新工具时多数情况下无需再改这里。
    """
    label = _TOOL_DISPLAY_LABELS.get(name, f"调用 {name}")
    for key in _DISPLAY_ARG_KEYS:
        value = str(args.get(key) or "").strip()
        if not value:
            continue
        value = " ".join(value.split())  # cypher 可能带换行
        if len(value) > _DISPLAY_VALUE_CHARS:
            value = value[:_DISPLAY_VALUE_CHARS] + "…"
        return f"{label}：{value}"
    return label


class RagAgent:
    def __init__(
        self,
        chain: Any,
        registry: ToolRegistry,
        settings: Settings | None = None,
    ) -> None:
        self.chain = chain
        self.registry = registry
        self.settings = settings or get_settings()

    async def astream(
        self,
        messages: list[ChatMessage],
        mode: str = "fast",
        *,
        allowed_tools: set[str] | None = None,
        trace: TraceHandle | None = None,
    ) -> AsyncIterator[AgentItem | AgentDone]:
        """跑一轮 Agent。

        `allowed_tools` 是本请求允许调用的工具集合（None = 全部允许）。
        注册表是应用级单例，**不能按请求修改**；这里在发请求前过滤 schema，
        并在执行侧同样校验——模型若点名一个未被允许的工具，不能真的执行。

        `trace` 是可选的 Langfuse 句柄。为 None 时 all no-op，Agent 行为与
        未接入可观测性时完全一致。
        """
        schemas = self.registry.schemas()
        if allowed_tools is not None:
            schemas = [s for s in schemas if s["function"]["name"] in allowed_tools]
        max_steps = max(1, self.settings.agent_max_steps)
        steps = 0
        total_tool_calls = 0
        root_trace = trace or TraceHandle.noop()

        for step in range(max_steps):
            steps = step + 1
            # 最后一轮不给工具，逼模型作答（否则可能无限检索下去）
            is_last = step == max_steps - 1
            req = LLMRequest(
                messages=messages,
                mode=mode,  # type: ignore[arg-type]
                tools=None if is_last else (schemas or None),
            )
            tool_names = [s["function"]["name"] for s in (req.tools or [])]

            # 每个 Agent 轮次一个 span，LLM 调用与工具执行都是它的子节点，
            # 这样在 Langfuse 里能直接看出「哪一步查了什么、模型据此说了什么」。
            step_trace = root_trace.start_span(
                f"agent.step.{steps}",
                input={"message_count": len(messages)},
                metadata={"step": steps, "last_step": is_last, "tools": tool_names},
            )
            generation = step_trace.start_generation(
                f"llm.step.{steps}",
                input={"messages": messages_payload(messages)},
                metadata={
                    "step": steps,
                    "mode": mode,
                    "last_step": is_last,
                    "tools": tool_names,
                },
            )

            turn_content: list[str] = []
            turn_reasoning: list[str] = []
            end: StreamEnd | None = None
            completion_start_time: datetime | None = None
            fallback_attempts: list[dict[str, Any]] = []

            try:
                async for item in self.chain.stream_with_fallback(req):
                    if isinstance(item, StreamEvent):
                        if item.kind == "content":
                            turn_content.append(item.text)
                        else:
                            turn_reasoning.append(item.text)
                        if completion_start_time is None:
                            completion_start_time = datetime.now(UTC)
                        # 内容与思考都实时转发，保持流式观感
                        yield item
                    elif isinstance(item, StreamEnd):
                        end = item
                    elif isinstance(item, FallbackEvent):
                        fallback_attempts.extend(
                            attempt.model_dump(mode="json") for attempt in item.attempts
                        )
                        yield item
                    elif isinstance(item, ProviderEvent):
                        # 先到的 provider 事件就能确定实际执行者（回退后不是主 provider）
                        generation.update(model=item.model, metadata={"provider": item.provider})
                        if step == 0:
                            # 只在首轮上报 provider，避免多轮重复刷新界面
                            yield item

                if end is None:
                    # 链路没有一个 provider 成功（异常已在 chain 内抛出），
                    # 走到这里说明是空流，交给上层按错误处理
                    raise AllProvidersFailed([])
            except Exception as exc:  # noqa: BLE001 — 记录后原样抛出
                generation.end(
                    level="ERROR",
                    status_message=str(exc),
                    output={
                        "content": "".join(turn_content),
                        "reasoning": "".join(turn_reasoning) or None,
                    },
                    completion_start_time=completion_start_time,
                )
                step_trace.end(
                    level="ERROR",
                    status_message=str(exc),
                    metadata={"step": steps, "error_type": type(exc).__name__},
                )
                raise
            except BaseException:
                # 取消 / 客户端断开：不是模型失败，但 span 必须收尾。
                generation.end(
                    level="WARNING",
                    status_message="cancelled",
                    output={"content": "".join(turn_content)},
                    completion_start_time=completion_start_time,
                )
                step_trace.end(level="WARNING", status_message="cancelled")
                raise

            generation_output = {
                "content": "".join(turn_content),
                "reasoning": "".join(turn_reasoning) or None,
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments} for tc in end.tool_calls
                ]
                or None,
            }
            generation.end(
                output=generation_output,
                model=end.model,
                usage_details=usage_payload(end.usage),
                metadata={
                    "finish_reason": end.finish_reason,
                    "provider": end.provider,
                    "fallback_attempts": fallback_attempts or None,
                },
                level="WARNING" if end.finish_reason == "length" else "DEFAULT",
                completion_start_time=completion_start_time,
            )

            if not end.tool_calls:
                step_trace.end(
                    output={
                        "content": generation_output["content"],
                        "tool_calls": 0,
                    },
                    metadata={"finish_reason": end.finish_reason},
                )
                yield AgentDone(
                    finish_reason=end.finish_reason,
                    steps=steps,
                    tool_calls=total_tool_calls,
                    provider=end.provider,
                    model=end.model,
                    usage=end.usage,
                )
                return

            # ---- 有工具调用：把这一轮记进历史，执行工具，进入下一轮 ----
            messages.append(
                ChatMessage(
                    role="assistant",
                    content="".join(turn_content),
                    tool_calls=end.tool_calls,
                    # DeepSeek 要求带 tool_calls 的 assistant 消息回传 reasoning_content，
                    # 否则后续请求会被 400 拒绝。
                    reasoning_content=end.reasoning,
                )
            )

            tool_results: list[dict[str, Any]] = []
            for tc in end.tool_calls:
                total_tool_calls += 1
                args = tc.parsed_arguments()
                yield ToolCallNotice(
                    name=tc.name,
                    arguments=args,
                    display=_display_for(tc.name, args),
                )
                tool_trace = step_trace.start_tool(
                    f"tool.{tc.name}",
                    input={"arguments": args},
                    metadata={"step": steps, "tool_call_id": tc.id},
                )

                started = time.perf_counter()
                try:
                    if allowed_tools is not None and tc.name not in allowed_tools:
                        # 只过滤 schema 还不够：模型可能幻构/越权点名未启用的工具
                        result = ToolResult(
                            name=tc.name,
                            ok=False,
                            error="tool_not_enabled",
                            content=(
                                f"错误：工具 {tc.name} 本次请求未启用（用户没有开启对应功能）。"
                                "请改用其他可用工具，或基于已有信息作答。"
                            ),
                        )
                    else:
                        result = await self.registry.execute(tc.name, args)
                except BaseException:
                    tool_trace.end(level="WARNING", status_message="cancelled")
                    step_trace.end(level="WARNING", status_message="cancelled")
                    raise
                if not result.elapsed_ms:
                    result.elapsed_ms = int((time.perf_counter() - started) * 1000)

                tool_trace.end(
                    output=tool_output_payload(result),
                    metadata={
                        "step": steps,
                        "ok": result.ok,
                        "error": result.error,
                        "elapsed_ms": result.elapsed_ms,
                        "degraded": bool(result.artifacts.get("degraded")),
                    },
                    level="DEFAULT" if result.ok else "ERROR",
                    status_message=result.error,
                )
                tool_results.append({"name": result.name, "ok": result.ok})

                yield ToolResultNotice(
                    name=result.name,
                    ok=result.ok,
                    elapsed_ms=result.elapsed_ms,
                    summary=_summarize(result),
                    sources=list(result.artifacts.get("sources") or []),
                    web_sources=list(result.artifacts.get("web_sources") or []),
                    degraded=bool(result.artifacts.get("degraded")),
                    content=result.content,
                )

                messages.append(
                    ChatMessage(role="tool", content=result.content, tool_call_id=tc.id)
                )

            step_trace.end(
                output={
                    "content": generation_output["content"],
                    "tool_calls": [tc.name for tc in end.tool_calls],
                    "results": tool_results,
                },
                metadata={"finish_reason": end.finish_reason},
            )

        # 理论上到不了：最后一轮不带工具，必然产出回答。
        # 真到了说明 provider 行为异常，给一个明确的收尾而不是静默卡住。
        logger.warning("Agent 达到最大步数 %d 仍未产出回答", max_steps)
        yield AgentDone(
            finish_reason="max_steps",
            steps=steps,
            tool_calls=total_tool_calls,
        )


#: 错误码 → 给用户看的中文。这段文案会直接显示在界面上，
#: 不要暴露 `retriever_unavailable` 这类内部标识。
_TOOL_ERROR_LABELS = {
    "retriever_unavailable": "知识库不可用",
    "empty_query": "查询为空",
    "unknown_tool": "工具不存在",
    "bad_arguments": "参数错误",
    "graph_unavailable": "知识图谱不可用",
    "graph_timeout": "知识图谱查询超时",
    "graph_query_failed": "知识图谱查询失败",
    "invalid_cypher": "Cypher 被拒绝（只允许只读查询）",
    "conflicting_arguments": "参数冲突",
    "unknown_aspect": "查询类型未知",
    "web_search_unavailable": "联网搜索不可用",
    "web_search_failed": "联网搜索失败",
    "web_search_timeout": "联网搜索超时",
    "tool_not_enabled": "工具未启用",
}


def _summarize(result: Any) -> str:
    if not result.ok:
        return _TOOL_ERROR_LABELS.get(result.error or "", "调用失败")
    # 工具可以自报用户可见摘要（artifacts["summary"]，如图谱工具）；
    # 没有时按检索语义计数（既有检索工具的行为，保持不变）。
    explicit = str(result.artifacts.get("summary") or "").strip()
    if explicit:
        return explicit
    count = len(result.artifacts.get("sources") or [])
    if count == 0:
        return "未检索到相关内容"
    return f"检索到 {count} 条相关资料"
