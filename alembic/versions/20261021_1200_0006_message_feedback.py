"""messages.feedback: 助手消息的用户反馈（👍/👎）

反馈用于后续分析检索与生成策略；只在助手消息上记录，可清除。

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-21 12:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("feedback", sa.String(length=8), nullable=True))
    op.add_column("messages", sa.Column("feedback_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "feedback_at")
    op.drop_column("messages", "feedback")
