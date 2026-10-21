"""ORM 模型：用户、会话、消息。

## 类型选择

- `sa.Uuid`：在 Postgres 上映射为原生 `uuid`，在其他数据库上是 `CHAR(32)`。
  用可移植类型是为了让 ORM 逻辑能在 SQLite 上跑测试——CI 里没有 Postgres。
- `sa.JSON().with_variant(JSONB, "postgresql")`：Postgres 上用 JSONB（可索引、
  可查询内部字段），其他数据库退化为普通 JSON。

## 为什么消息要存 `sources`

引用来源是回答的一部分——没有它，用户回看历史时无法核对依据。
存成 JSON 而不是单独的关联表：来源只随消息整体读写，从不单独查询，
单独建表只会增加连接开销。
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from bidding_qa_chatbot.db.base import Base

#: 可移植的 JSON 类型：Postgres 用 JSONB，其余退化为 JSON
JSONType = sa.JSON().with_variant(JSONB, "postgresql")

#: 附件列专用：Python None 写成 **SQL NULL**（而不是 JSON "null"）。
#: 否则 `attachments IS NOT NULL` 会把每条无附件消息也当成有值，
#: 追问注入的 limit 会被空消息挤占。
AttachmentJSONType = sa.JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(sa.String(64), unique=True, index=True)
    email: Mapped[str] = mapped_column(sa.String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(sa.String(128))
    display_name: Mapped[str | None] = mapped_column(sa.String(64), default=None)
    is_active: Mapped[bool] = mapped_column(sa.Boolean, default=True, server_default=sa.true())
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=_utcnow, server_default=sa.func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, server_default=sa.func.now()
    )

    conversations: Mapped[list[Conversation]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )

    def __repr__(self) -> str:
        # 绝不包含 password_hash
        return f"<User {self.username}>"


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(sa.String(120), default="新对话")
    # 冗余计数：会话列表要显示条数，不这样就得对每条会话做一次子查询
    message_count: Mapped[int] = mapped_column(sa.Integer, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=_utcnow, server_default=sa.func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, server_default=sa.func.now()
    )

    user: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    __table_args__ = (
        # 会话列表按「最后更新时间倒序」展示，这条复合索引直接服务该查询
        sa.Index("ix_conversations_user_updated", "user_id", sa.text("updated_at DESC")),
    )

    def __repr__(self) -> str:
        return f"<Conversation {self.id} {self.title!r}>"


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    #: 会话内序号，从 1 开始。**消息顺序只许按它排，不能按 created_at。**
    #: 同一事务里连插两条时，两者的 created_at 会完全相同（Windows 上
    #: datetime.now() 只有约 15ms 分辨率），此时兜底比较的是随机 UUID，
    #: 顺序就成了随机的——表现为用户回看历史时消息错位。
    seq: Mapped[int] = mapped_column(sa.Integer)
    role: Mapped[str] = mapped_column(sa.String(16))
    content: Mapped[str] = mapped_column(sa.Text)
    reasoning: Mapped[str | None] = mapped_column(sa.Text, default=None)
    #: 引用来源（SourceRef 列表）。回看历史时用来核对依据。
    sources: Mapped[list | None] = mapped_column(JSONType, default=None)
    #: 用户消息携带的附件元数据 + OCR 文本（仅随消息整体读写）。
    #: 对外接口只下发元数据，OCR 文本用于同一会话的追问上下文。
    attachments: Mapped[list | None] = mapped_column(AttachmentJSONType, default=None)
    provider: Mapped[str | None] = mapped_column(sa.String(32), default=None)
    model: Mapped[str | None] = mapped_column(sa.String(128), default=None)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=_utcnow, server_default=sa.func.now()
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    __table_args__ = (
        # 按会话取消息并按序号排序，是最高频的查询
        sa.Index("ix_messages_conversation_seq", "conversation_id", "seq"),
        # 同一会话内序号唯一：并发写入导致序号撞车时要**报错**，而不是
        # 悄悄出现两条顺序相同的消息
        sa.UniqueConstraint("conversation_id", "seq", name="uq_messages_conversation_seq"),
    )

    def __repr__(self) -> str:
        return f"<Message {self.role} {self.content[:20]!r}>"
