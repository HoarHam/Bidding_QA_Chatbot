"""消息表增加会话内序号 seq

改用 seq 排序的原因：原本按 (created_at, id) 排序，而同一事务里连续插入的消息
会拿到**完全相同**的 created_at（Windows 上 datetime.now() 只有约 15ms 分辨率），
此时兜底比较的是随机 UUID，导致顺序随机——用户回看历史时消息会错位。

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-12 17:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 先允许为空，回填后再收紧为 NOT NULL——直接加 NOT NULL 列会因为已有行没有值而失败
    op.add_column("messages", sa.Column("seq", sa.Integer(), nullable=True))

    # 回填：按 (created_at, id) 在会话内编号。这是**尽力而为**——对已有数据而言
    # 没有更可靠的信息可用（时间戳并列时本来就没有确定的先后）。
    # 本功能上线前数据库里不会有真实对话，所以这个近似是安全的。
    op.execute(
        """
        UPDATE messages
        SET seq = (
            SELECT COUNT(*)
            FROM messages AS m2
            WHERE m2.conversation_id = messages.conversation_id
              AND (
                    m2.created_at < messages.created_at
                 OR (m2.created_at = messages.created_at AND m2.id <= messages.id)
              )
        )
        """
    )

    # 用 batch 模式：SQLite 不支持 ALTER COLUMN / 加约束，batch 会做整表重建。
    # 这不是为了「支持 SQLite 生产环境」——本项目的目标库是 Postgres；
    # 而是因为在没有 Postgres 的机器上也能**实测迁移是否可执行**。
    # batch 模式在 Postgres 上就是普通 DDL，没有额外开销。
    with op.batch_alter_table("messages") as batch:
        batch.alter_column("seq", nullable=False)
        batch.drop_index("ix_messages_conversation_created")
        batch.create_index("ix_messages_conversation_seq", ["conversation_id", "seq"])
        # 同一会话内序号唯一：并发写入撞车要报错，而不是产生两条顺序相同的消息
        batch.create_unique_constraint("uq_messages_conversation_seq", ["conversation_id", "seq"])


def downgrade() -> None:
    with op.batch_alter_table("messages") as batch:
        batch.drop_constraint("uq_messages_conversation_seq", type_="unique")
        batch.drop_index("ix_messages_conversation_seq")
        batch.create_index("ix_messages_conversation_created", ["conversation_id", "created_at"])
        batch.drop_column("seq")
