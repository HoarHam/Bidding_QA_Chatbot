"""会话历史自动摘要（前情提要）。

当会话消息数超过保留窗口时，把更早的消息用 LLM 压缩成一段摘要，保存到
`conversations.summary`（并记录覆盖到的 seq），后续每轮自动注入提示词。

- 摘要在**请求处理期间**完成并等待（用户不可见）；超时/失败只记日志、跳过，
  绝不阻塞正常问答。
- 只摘要「窗口之外」的消息；最近 N 条始终以原文进入上下文。
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.db import repository as repo
from bidding_qa_chatbot.db.models import Message
from bidding_qa_chatbot.llm.base import ChatMessage, LLMRequest
from bidding_qa_chatbot.log import redact
from bidding_qa_chatbot.observability import TraceHandle, messages_payload, usage_payload

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """你负责压缩对话历史，供后续轮次回顾。把给定对话压缩成一段简洁的中文前情提要：
- 保留用户的核心意图与约束、已确认的事实与结论、已解决的问题；
- 保留仍待解决、用户明确追问过的问题；
- 不要编造，不要加入新的建议或结论；
- 直接输出摘要正文，不要加标题或解释，不超过 {max_chars} 字。"""


async def ensure_history_summary(
    *,
    chain,
    db,
    settings: Settings,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    trace: TraceHandle | None = None,
) -> str | None:
    """按需生成/更新会话摘要，返回当前可注入的摘要（失败时返回旧摘要或 None）。

    `trace` 是当前问答的 Langfuse 句柄；摘要用的 LLM 调用会作为它的
    generation 子节点被观测到。
    """
    try:
        return await _ensure(chain, db, settings, user_id, conversation_id, trace)
    except Exception as exc:  # noqa: BLE001 — 摘要失败不能拖垮对话
        logger.warning("生成会话摘要失败（本轮跳过）: %s", redact(str(exc))[:200])
        return None


async def _ensure(chain, db, settings, user_id, conversation_id, trace=None) -> str | None:
    async with db.session() as session:
        conversation = await repo.get_conversation(
            session, user_id=user_id, conversation_id=conversation_id
        )
        if conversation is None:
            return None
        total = conversation.message_count or 0
        window = max(1, settings.history_window_messages)
        cutoff = total - window
        prior_upto = conversation.summary_upto_seq or 0
        if cutoff <= prior_upto:
            return conversation.summary or None

        messages = await repo.list_messages(
            session, user_id=user_id, conversation_id=conversation_id, limit=1000
        )
        prior_summary = conversation.summary or ""

    overflow = [m for m in messages if prior_upto < m.seq <= cutoff]
    if not overflow:
        return prior_summary or None

    transcript = _format_transcript(overflow, settings.history_summary_input_max_chars)
    if prior_summary:
        user_content = f"【已有摘要】\n{prior_summary}\n\n【需要并入摘要的新对话】\n{transcript}"
    else:
        user_content = f"【需要压缩的对话】\n{transcript}"

    request = LLMRequest(
        messages=[
            ChatMessage(
                role="system",
                content=SUMMARY_SYSTEM_PROMPT.format(max_chars=settings.history_summary_max_chars),
            ),
            ChatMessage(role="user", content=user_content),
        ],
        mode="fast",
    )
    generation = (trace or TraceHandle.noop()).start_generation(
        "history.summary",
        input={"messages": messages_payload(request.messages)},
        metadata={"timeout_s": settings.history_summary_timeout},
    )
    try:
        result = await asyncio.wait_for(
            chain.complete(request), timeout=settings.history_summary_timeout
        )
    except Exception as exc:  # noqa: BLE001 — 记到 trace 后再交给外层降级
        generation.end(level="ERROR", status_message=str(exc))
        raise
    summary = (getattr(result, "text", "") or "").strip()
    generation.end(
        output={"summary": summary},
        model=getattr(result, "model", None),
        usage_details=usage_payload(getattr(result, "usage", None)),
        metadata={"provider": getattr(result, "provider", None)},
    )
    if not summary:
        return prior_summary or None
    if len(summary) > settings.history_summary_max_chars:
        summary = summary[: settings.history_summary_max_chars] + "…"

    async with db.session() as session:
        await repo.update_conversation_summary(
            session,
            user_id=user_id,
            conversation_id=conversation_id,
            summary=summary,
            upto_seq=cutoff,
        )
    logger.info("会话摘要已更新: conversation=%s upto_seq=%d", conversation_id, cutoff)
    return summary


def _format_transcript(messages: list[Message], max_chars: int) -> str:
    """把消息拼成压缩输入；单条截断、总量截断（宁可少摘要，不撑爆上下文）。"""
    lines: list[str] = []
    used = 0
    for message in messages:
        label = "用户" if message.role == "user" else "助手"
        body = " ".join((message.content or "").split())[:800]
        line = f"{label}：{body}"
        if used + len(line) > max_chars and lines:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)
