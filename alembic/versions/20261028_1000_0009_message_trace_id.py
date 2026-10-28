"""messages.trace_id: 关联 Langfuse trace，用于用户反馈打分

助手消息落库时保存本轮问答的 Langfuse trace id；用户点 👍/👎 时据此把
分数打到原 trace 上。可观测性未启用时该列为 NULL，不影响消息本身。

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-28 10:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("trace_id", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "trace_id")
