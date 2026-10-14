"""Alembic 环境。

数据库连接串来自应用配置（`.env` 的 `POSTGRES_*`），而不是 alembic.ini——
写两处必然会不同步。

用异步引擎执行迁移，driver 与应用保持一致（`postgresql+psycopg`）。
"""

from __future__ import annotations

import asyncio
import sys
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from bidding_qa_chatbot.config import get_settings

if sys.platform == "win32":
    # psycopg 的异步模式无法在 Windows 默认的 ProactorEventLoop 上运行。
    # 必须在 asyncio.run() 之前设置，否则迁移会以一个和配置毫无关系的
    # InterfaceError 失败。
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 必须导入模型，否则 autogenerate 会认为「没有任何表」而生成空迁移
from bidding_qa_chatbot.db import models  # noqa: F401
from bidding_qa_chatbot.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 连接串直接传给引擎，**不经过 config.set_main_option**。
# configparser 会对配置值做 `%` 插值，而 URL 编码后的密码里几乎必然含 `%`
# （`@` → `%40`、中文 → `%E4%BD%A0`…），一旦写进去就会抛
# "invalid interpolation syntax"。绕开它比转义 `%%` 更省心。
DB_URL = get_settings().database_url

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL，不连库。用于产出供 DBA 审阅的脚本。"""
    context.configure(
        url=DB_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # compare_type：列类型变了也要被 autogenerate 检测出来，
        # 否则把 String(64) 改成 String(255) 这类改动会被静默忽略
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = create_async_engine(
        DB_URL,
        poolclass=pool.NullPool,  # 迁移是一次性动作，不需要连接池
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
