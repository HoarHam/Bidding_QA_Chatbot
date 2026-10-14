"""users 的唯一约束收敛为唯一索引

0001 在 username / email 上同时建了「唯一约束」与「普通索引」——
同一列两套索引结构，纯冗余；而模型声明的是唯一索引
（`mapped_column(..., unique=True, index=True)` 渲染为 `CREATE UNIQUE INDEX`）。

模型与数据库不一致的代价是 `--autogenerate` **每次都报差异**，
久而久之没人再信任自动生成的迁移。

0001 已同步修正（新库直接建唯一索引），本迁移负责改已有的库。
仅作用于 Postgres：SQLite 只用于测试（走 create_all），不做这类结构迁移。

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-13 11:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    op.drop_constraint("uq_users_username", "users", type_="unique")
    op.drop_constraint("uq_users_email", "users", type_="unique")
    # 普通索引要一并删掉，否则重建唯一索引时会重名
    op.drop_index("ix_users_username", table_name="users")
    op.drop_index("ix_users_email", table_name="users")

    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True)


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_username", table_name="users")

    op.create_index("ix_users_username", "users", ["username"])
    op.create_index("ix_users_email", "users", ["email"])
    op.create_unique_constraint("uq_users_username", "users", ["username"])
    op.create_unique_constraint("uq_users_email", "users", ["email"])
