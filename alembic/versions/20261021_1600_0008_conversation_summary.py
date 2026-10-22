"""conversations.summary: 历史消息自动摘要（前情提要）

当会话消息超过保留窗口时，用 LLM 把较早消息压缩成一段摘要并保存；
后续每轮自动注入提示词，帮助模型回顾更早的上下文。

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-21 16:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("conversations", sa.Column("summary", sa.Text(), nullable=True))
    op.add_column(
        "conversations",
        sa.Column("summary_upto_seq", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "conversations",
        sa.Column("summary_updated_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("conversations", "summary_updated_at")
    op.drop_column("conversations", "summary_upto_seq")
    op.drop_column("conversations", "summary")
