"""messages.attachments: 用户附件元数据与 OCR 文本

附件随用户消息一起落库（元数据 + 仅服务端可见的 OCR 文本），用于：
1. 历史消息回显附件 chip；
2. 同一会话后续追问时注入最近附件的 OCR 上下文。

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-20 12:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: 与 db/models.py 的 JSONType 保持一致：Postgres 用 JSONB，SQLite 用 JSON
JSONType = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("messages", sa.Column("attachments", JSONType, nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "attachments")
