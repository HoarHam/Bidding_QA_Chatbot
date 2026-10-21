"""messages.timings: 助手回答的分阶段用时

持久化后，跨会话恢复时也能展示总用时/思考/工具执行/生成回答。

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-21 14:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSONType = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("messages", sa.Column("timings", JSONType, nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "timings")
