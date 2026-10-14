"""messages.sources 由 json 改为 jsonb

`db/models.py` 里声明的是 `sa.JSON().with_variant(JSONB, "postgresql")`，
但 0001 里手写成了裸 `sa.JSON()`，实际建出来的是 `json`。
模型与数据库不一致会让后续 `--autogenerate` **每次都生成一条无意义的
alter column**，而且 JSONB 的 GIN 索引等能力也用不上。

0001 已同步修正（新库直接建 JSONB），本迁移负责把已存在的库改过来。

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-13 10:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    # 只有 Postgres 有 JSONB；SQLite 上 JSON 就是正确形态，不动
    if bind.dialect.name != "postgresql":
        return

    # postgresql_using 是必需的：没有它，Postgres 不知道怎么把已有值转过去
    op.alter_column(
        "messages",
        "sources",
        type_=JSONB(),
        existing_type=sa.JSON(),
        existing_nullable=True,
        postgresql_using="sources::jsonb",
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.alter_column(
        "messages",
        "sources",
        type_=sa.JSON(),
        existing_type=JSONB(),
        existing_nullable=True,
        postgresql_using="sources::json",
    )
