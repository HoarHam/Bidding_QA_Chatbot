"""建立账号与对话历史表

Revision ID: 0001
Revises:
Create Date: 2026-10-12 10:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

#: 与 db/models.py 保持一致：Postgres 用 JSONB（可建 GIN 索引），其余数据库退化为普通 JSON。
#: 迁移里**显式内联**而不是从模型导入——迁移是历史快照，不该随应用代码变化。
JSONType = sa.JSON().with_variant(JSONB(), "postgresql")

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=128), nullable=False),
        sa.Column("display_name", sa.String(length=64), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
    )
    # 唯一索引，**不是**「唯一约束 + 普通索引」两套：
    # 模型里是 `unique=True, index=True`，SQLAlchemy 渲染出来的就是
    # `CREATE UNIQUE INDEX`。两者都建会在同一列上留两套索引结构（纯冗余），
    # 而且模型与数据库不一致会让后续 --autogenerate 反复报差异。
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "conversations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=120), nullable=False),
        sa.Column("message_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name="pk_conversations"),
        # 删用户时会话跟着删；消息再由 conversations 级联下去
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_conversations_user_id_users", ondelete="CASCADE"
        ),
    )
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])
    # 会话列表按「最后更新倒序」展示，这条复合索引直接服务该查询
    op.create_index(
        "ix_conversations_user_updated",
        "conversations",
        ["user_id", sa.text("updated_at DESC")],
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column("sources", JSONType, nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=True),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name="pk_messages"),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_messages_conversation_id_conversations",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])
    op.create_index(
        "ix_messages_conversation_created", "messages", ["conversation_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_messages_conversation_created", table_name="messages")
    op.drop_index("ix_messages_conversation_id", table_name="messages")
    op.drop_table("messages")

    op.drop_index("ix_conversations_user_updated", table_name="conversations")
    op.drop_index("ix_conversations_user_id", table_name="conversations")
    op.drop_table("conversations")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
