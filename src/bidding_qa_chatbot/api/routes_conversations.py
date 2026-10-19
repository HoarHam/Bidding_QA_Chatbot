"""对话历史：列表、详情（跨会话恢复）、删除、搜索。"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, status

from bidding_qa_chatbot.auth.deps import CurrentUser, DbSession
from bidding_qa_chatbot.db import repository as repo
from bidding_qa_chatbot.db.models import Conversation, Message
from bidding_qa_chatbot.schemas import (
    ConversationDetail,
    ConversationListResponse,
    ConversationSummary,
    MessageOut,
    SearchHit,
    SearchResponse,
    SourceRef,
    WebSource,
)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])

#: 搜索结果里每条命中显示的字符数
SNIPPET_CHARS = 120


def _summary(conv: Conversation) -> ConversationSummary:
    return ConversationSummary(
        id=str(conv.id),
        title=conv.title,
        message_count=conv.message_count or 0,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


def _message_out(msg: Message) -> MessageOut:
    # 存进去的是来源 dict 列表：本地来源是 SourceRef 原样，联网来源带 kind="web"。
    # 老数据没有 kind → 按本地来源解析（向后兼容）；单条来源损坏不该让整个会话打不开。
    raw_sources = msg.sources or []
    sources: list[SourceRef] = []
    web_sources: list[WebSource] = []
    for item in raw_sources:
        try:
            if isinstance(item, dict) and item.get("kind") == "web":
                web_sources.append(WebSource.model_validate(item))
            else:
                sources.append(SourceRef.model_validate(item))
        except Exception:  # noqa: BLE001
            continue
    return MessageOut(
        id=str(msg.id),
        role=msg.role,  # type: ignore[arg-type]
        content=msg.content,
        reasoning=msg.reasoning,
        sources=sources,
        web_sources=web_sources,
        provider=msg.provider,
        model=msg.model,
        created_at=msg.created_at,
    )


def _parse_uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="会话 id 格式非法"
        ) from exc


# ---- 注意：/search 必须声明在 /{conversation_id} 之前 ----
# FastAPI 按声明顺序匹配路由。反过来的话，"search" 会被当成会话 id 去解析，
# 结果是搜索接口永远返回「id 格式非法」。
@router.get("/search", response_model=SearchResponse)
async def search_conversations(
    user: CurrentUser,
    session: DbSession,
    q: str = Query(min_length=1, max_length=100, description="检索关键词"),
    limit: int = Query(default=30, ge=1, le=100),
) -> SearchResponse:
    """在自己**全部**对话历史里搜索关键词。"""
    rows = await repo.search_messages(session, user_id=user.id, query=q, limit=limit)
    hits = [
        SearchHit(
            conversation_id=str(conv.id),
            conversation_title=conv.title,
            message_id=str(msg.id),
            role=msg.role,  # type: ignore[arg-type]
            snippet=_snippet(msg.content, q),
            created_at=msg.created_at,
        )
        for msg, conv in rows
    ]
    return SearchResponse(query=q, hits=hits, total=len(hits))


def _snippet(content: str, keyword: str) -> str:
    """截取关键词附近的片段。从正文开头截会经常看不到命中的地方。"""
    flat = " ".join(content.split())
    idx = flat.lower().find(keyword.lower())
    if idx <= 0:
        return flat[:SNIPPET_CHARS] + ("…" if len(flat) > SNIPPET_CHARS else "")
    start = max(0, idx - SNIPPET_CHARS // 3)
    end = start + SNIPPET_CHARS
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(flat) else ""
    return f"{prefix}{flat[start:end]}{suffix}"


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    user: CurrentUser,
    session: DbSession,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> ConversationListResponse:
    """按最后更新时间倒序返回会话列表——跨会话恢复的入口。"""
    items = await repo.list_conversations(session, user_id=user.id, limit=limit, offset=offset)
    total = await repo.count_conversations(session, user_id=user.id)
    return ConversationListResponse(items=[_summary(c) for c in items], total=total)


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str, user: CurrentUser, session: DbSession
) -> ConversationDetail:
    """取某个会话的全部消息（点开历史对话时调用）。"""
    conv_id = _parse_uuid(conversation_id)
    # 仓储层同时按 user_id 过滤：拿到别人的会话 id 也读不到
    conv = await repo.get_conversation(session, user_id=user.id, conversation_id=conv_id)
    if conv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")

    messages = await repo.list_messages(session, user_id=user.id, conversation_id=conv_id)
    return ConversationDetail(
        id=str(conv.id),
        title=conv.title,
        message_count=conv.message_count or 0,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=[_message_out(m) for m in messages],
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: str, user: CurrentUser, session: DbSession) -> None:
    conv_id = _parse_uuid(conversation_id)
    deleted = await repo.delete_conversation(session, user_id=user.id, conversation_id=conv_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
