"""数据访问层。

## 一条贯穿全文件的原则

**每个查询都带上 `user_id` 条件。** 只按主键取会话/消息（`WHERE id = ?`）会形成
越权漏洞：任何人拿到一个 UUID 就能读到别人的对话。把授权写进查询本身，
而不是依赖调用方记得检查——调用方会忘，`WHERE` 不会。
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from typing import Any

import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from bidding_qa_chatbot.db.models import Conversation, Message, User

#: 会话标题取首条用户消息的前 N 个字符
TITLE_MAX_CHARS = 40


def _json_safe(value: Any) -> Any:
    """把 pydantic 模型与容器转成可直接写入 JSON 列的形式。

    这是必要的防御。`sa.JSON` 底层走 `json.dumps`，遇到 `BaseModel` 会抛
    `Object of type SourceRef is not JSON serializable`。而这个异常发生在
    **回答已经流式返回之后**，表现出来是「界面上回答正常、但历史里没有」——
    用户侧极难察觉，只会发现跨会话恢复时缺了回答。

    放在仓储层而不是调用方：写 JSON 列的地方都该自动获得这个保护，
    漏掉一处就会重现同一个问题。
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    return value


# ------------------------------------------------------------------ 用户
async def create_user(
    session: AsyncSession,
    *,
    username: str,
    email: str,
    password_hash: str,
    display_name: str | None = None,
) -> User:
    user = User(
        username=username,
        email=email,
        password_hash=password_hash,
        display_name=display_name,
    )
    session.add(user)
    await session.flush()  # 拿到自生成的 id，但不提交（交给外层事务）
    return user


async def get_user_by_username(session: AsyncSession, username: str) -> User | None:
    return (
        await session.execute(sa.select(User).where(User.username == username))
    ).scalar_one_or_none()


async def get_user_by_email(session: AsyncSession, email: str) -> User | None:
    return (await session.execute(sa.select(User).where(User.email == email))).scalar_one_or_none()


async def get_user_by_id(session: AsyncSession, user_id: uuid.UUID) -> User | None:
    return (await session.execute(sa.select(User).where(User.id == user_id))).scalar_one_or_none()


async def username_or_email_taken(session: AsyncSession, username: str, email: str) -> str | None:
    """返回冲突的字段名，都可用时返回 None。

    一次查询取两个冲突，比注册时连着查两次少一次往返；
    也不必依赖「捕获唯一约束异常」——那样拿不到是哪个字段冲突，
    只能给用户一句笼统的「注册失败」。
    """
    stmt = sa.select(User.username, User.email).where(
        sa.or_(User.username == username, User.email == email)
    )
    for row in (await session.execute(stmt)).all():
        if row.username == username:
            return "username"
        if row.email == email:
            return "email"
    return None


# ------------------------------------------------------------------ 会话
async def create_conversation(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    title: str | None = None,
) -> Conversation:
    conv = Conversation(user_id=user_id, title=title or "新对话")
    session.add(conv)
    await session.flush()
    return conv


async def get_conversation(
    session: AsyncSession, *, user_id: uuid.UUID, conversation_id: uuid.UUID
) -> Conversation | None:
    """按 id 取会话，**同时限定归属用户**。"""
    stmt = sa.select(Conversation).where(
        Conversation.id == conversation_id, Conversation.user_id == user_id
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def list_conversations(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    limit: int = 50,
    offset: int = 0,
) -> list[Conversation]:
    """按最后更新时间倒序返回会话列表（跨会话恢复的入口）。"""
    stmt = (
        sa.select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list((await session.execute(stmt)).scalars().all())


async def count_conversations(session: AsyncSession, *, user_id: uuid.UUID) -> int:
    stmt = (
        sa.select(sa.func.count()).select_from(Conversation).where(Conversation.user_id == user_id)
    )
    return int((await session.execute(stmt)).scalar_one())


async def delete_conversation(
    session: AsyncSession, *, user_id: uuid.UUID, conversation_id: uuid.UUID
) -> bool:
    """删除会话。返回是否真的删掉了（未找到或不属于该用户 → False）。"""
    conv = await get_conversation(session, user_id=user_id, conversation_id=conversation_id)
    if conv is None:
        return False
    await session.delete(conv)  # 级联删除消息（relationship 上配了 cascade）
    return True


# ------------------------------------------------------------------ 消息
async def add_message(
    session: AsyncSession,
    *,
    conversation: Conversation,
    role: str,
    content: str,
    reasoning: str | None = None,
    sources: list | None = None,
    attachments: list | None = None,
    timings: dict | None = None,
    provider: str | None = None,
    model: str | None = None,
    title_from: bool = False,
) -> Message:
    """追加一条消息，并维护会话的冗余计数与更新时间。

    `title_from=True` 时（首条用户消息），顺带把会话标题设成内容摘要——
    列表页要有个能认出来的名字，让模型额外生成标题则是又一次调用与延迟。
    """
    # 序号取「当前条数 + 1」。conversation.message_count 与 seq 同源，
    # 因此两者天然一致；(conversation_id, seq) 上有唯一约束，并发写入撞车会报错
    # 而不是产生两条顺序相同的消息。
    next_seq = (conversation.message_count or 0) + 1

    msg = Message(
        conversation_id=conversation.id,
        seq=next_seq,
        role=role,
        content=content,
        reasoning=reasoning,
        # 归一化后再写入：调用方可能传的是 SourceRef 这类 pydantic 对象
        sources=_json_safe(sources),
        attachments=_json_safe(attachments),
        timings=_json_safe(timings),
        provider=provider,
        model=model,
    )
    session.add(msg)

    conversation.message_count = next_seq
    conversation.updated_at = datetime.now(UTC)
    if title_from and conversation.title in (None, "", "新对话"):
        conversation.title = _make_title(content)

    await session.flush()
    return msg


def _make_title(text: str) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= TITLE_MAX_CHARS:
        return cleaned or "新对话"
    return cleaned[:TITLE_MAX_CHARS] + "…"


async def list_messages(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    limit: int = 200,
) -> list[Message]:
    """取某会话的全部消息（跨会话恢复时调用）。

    同样限定 user_id：会话 id 泄露的情况下也不能读到别人的对话。
    """
    stmt = (
        sa.select(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(Message.conversation_id == conversation_id, Conversation.user_id == user_id)
        # 按 seq 排序，不能按 created_at：同一事务插入的消息时间戳可能完全相同
        .order_by(Message.seq)
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())


async def update_conversation_summary(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    summary: str,
    upto_seq: int,
) -> bool:
    """保存会话摘要及其覆盖到的消息序号（用户隔离）。"""
    conversation = await get_conversation(session, user_id=user_id, conversation_id=conversation_id)
    if conversation is None:
        return False
    conversation.summary = summary
    conversation.summary_upto_seq = upto_seq
    conversation.summary_updated_at = datetime.now(UTC)
    await session.flush()
    return True


async def get_message(
    session: AsyncSession, *, user_id: uuid.UUID, message_id: uuid.UUID
) -> Message | None:
    """按 id 取消息，**查询本身带 user_id**（越权读取防护）。"""
    stmt = (
        sa.select(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(Message.id == message_id, Conversation.user_id == user_id)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def get_message_by_seq(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    seq: int,
) -> Message | None:
    """按会话内序号取消息（重新生成时定位前一条用户消息）。"""
    stmt = (
        sa.select(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(
            Message.conversation_id == conversation_id,
            Message.seq == seq,
            Conversation.user_id == user_id,
        )
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def set_message_feedback(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    message_id: uuid.UUID,
    rating: str | None,
) -> bool:
    """写/清除助手消息的反馈。非本人或非助手消息返回 False。"""
    message = await get_message(session, user_id=user_id, message_id=message_id)
    if message is None or message.role != "assistant":
        return False
    message.feedback = rating
    message.feedback_at = datetime.now(UTC) if rating else None
    await session.flush()
    return True


async def update_assistant_message(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    message_id: uuid.UUID,
    content: str,
    reasoning: str | None,
    sources: list,
    timings: dict | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> bool:
    """重新生成：原地更新同一条助手消息（不改 seq/message_count，清空原反馈）。"""
    message = await get_message(session, user_id=user_id, message_id=message_id)
    if message is None or message.role != "assistant":
        return False
    message.content = content
    message.reasoning = reasoning
    message.sources = _json_safe(sources)
    message.timings = _json_safe(timings)
    message.provider = provider
    message.model = model
    message.feedback = None
    message.feedback_at = None
    await session.flush()
    return True


async def recent_user_attachments(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    limit: int = 1,
) -> list[list[dict]]:
    """取该会话最近 N 条「带附件」的用户消息的附件列表（最新在前）。

    用于同一会话的追问：把最近附件的 OCR 文本注入新一轮提示词。
    查询必须带 user_id（与文件里所有查询一致），避免越权读取他人附件文本。
    """
    stmt = (
        sa.select(Message.attachments)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(
            Message.conversation_id == conversation_id,
            Conversation.user_id == user_id,
            Message.role == "user",
            Message.attachments.is_not(None),
        )
        .order_by(Message.seq.desc())
        .limit(max(1, limit))
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [row for row in rows if isinstance(row, list)]


# ------------------------------------------------------------------ 搜索
async def search_messages(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    query: str,
    limit: int = 30,
) -> list[tuple[Message, Conversation]]:
    """在**该用户自己**的对话历史里做关键词搜索。

    用 `ILIKE` 而非 Postgres 全文检索：中文没有默认分词器，
    `to_tsvector` 对中文基本不工作（会把整句当成一个词）。
    数据量再大一些时，正确做法是接 `pg_trgm` 或外部分词，
    而不是改成 tsvector。

    返回 (消息, 所属会话) —— 前端要显示「这条来自哪个对话」并能跳过去。
    """
    keyword = query.strip()
    if not keyword:
        return []

    # 转义 LIKE 的通配符：用户搜 "100%" 时不该把 % 当成通配
    escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f"%{escaped}%"

    stmt = (
        sa.select(Message, Conversation)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(
            Conversation.user_id == user_id,
            Message.content.ilike(pattern, escape="\\"),
        )
        # created_at 会有并列（同事务写入），用 (conversation_id, seq) 兜底保证
        # 结果顺序稳定——否则同一查询两次调用可能返回不同顺序，分页时会漏数据
        .order_by(Message.created_at.desc(), Message.conversation_id, Message.seq.desc())
        .limit(limit)
    )
    return [(row[0], row[1]) for row in (await session.execute(stmt)).all()]
