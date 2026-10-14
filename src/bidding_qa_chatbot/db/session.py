"""异步引擎与会话工厂。

用 psycopg3 的异步驱动（`postgresql+psycopg://`），与全异步的 FastAPI 应用一致，
不需要把数据库调用丢进线程池。

测试用 SQLite（`sqlite+aiosqlite://`），因此连接池参数只在非 SQLite 时传入——
SQLite 的内存库不支持这些参数，传了会直接报错。
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from bidding_qa_chatbot.config import Settings, get_settings
from bidding_qa_chatbot.db.base import Base

# 必须导入模型模块，否则模型不会注册到 Base.metadata 上，
# create_all() 会「成功」地什么都不建——然后以 "no such table" 的形式
# 在别处炸掉。运行中的服务碰巧没事（路由会间接导入模型），
# 但这属于依赖导入顺序的隐患，显式导入才可靠。
from bidding_qa_chatbot.db import models  # noqa: F401  # isort: skip

logger = logging.getLogger(__name__)


def _ensure_psycopg_compatible_loop() -> None:
    """Windows 上把事件循环策略设为 Selector。

    psycopg 的异步模式无法在 `ProactorEventLoop`（Windows 默认）上运行，会抛
    `InterfaceError: Psycopg cannot use the 'ProactorEventLoop'`。

    放在这里而不是只放在 CLI：`cli.py` 显式建 Selector 循环解决了常规启动方式，
    但**绕过 CLI 的启动路径**（直接 `uvicorn bidding_qa_chatbot.api.app:app`、
    自定义脚本、某些部署框架）拿不到那份修复。db 模块是所有这些路径的必经之处。

    只在**没有运行中的循环**时设置：循环一旦跑起来，改策略既无效也会告警。
    真正需要它生效的场景（进程启动早期）恰好满足这个条件。
    """
    if sys.platform != "win32":
        return
    try:
        asyncio.get_running_loop()
        return  # 已在循环内，改了也没用
    except RuntimeError:
        pass
    if not isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


_ensure_psycopg_compatible_loop()

#: 这些 URL 前缀走 SQLite（测试），不传连接池参数
_SQLITE_PREFIXES = ("sqlite",)


def _is_sqlite(url: str) -> bool:
    return url.split("://", 1)[0].startswith(_SQLITE_PREFIXES)


class Database:
    def __init__(self, url: str, settings: Settings | None = None) -> None:
        self.url = url
        self.settings = settings or get_settings()

        kwargs: dict[str, Any] = {"echo": self.settings.db_echo, "future": True}
        if _is_sqlite(url):
            if ":memory:" in url:
                # 内存 SQLite 的**每个连接都是独立的库**。不加静态池的话，
                # create_all 建的表在下一个连接里就不存在，测试会以
                # "no such table" 的形式诡异失败。
                kwargs["poolclass"] = StaticPool
        else:
            # pool_pre_ping：连接被数据库端掐断后（空闲超时、重启）能自动重建，
            # 否则会持续抛 "server closed the connection unexpectedly"
            kwargs.update(
                pool_pre_ping=True,
                pool_size=self.settings.db_pool_size,
                max_overflow=self.settings.db_max_overflow,
            )

        self.engine: AsyncEngine = create_async_engine(url, **kwargs)
        # expire_on_commit=False：提交后仍可读取对象属性，否则序列化响应时
        # 会触发一次意料之外的懒加载（在 async 下还会直接抛 MissingGreenlet）
        self.sessionmaker = async_sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """事务边界：正常退出提交，异常回滚。"""
        async with self.sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def create_all(self) -> None:
        """按模型建表。

        仅用于测试与首次本地搭建；正式环境的表结构变更走 Alembic 迁移
        （`alembic upgrade head`），因为 create_all 不会修改已存在的表。
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_all(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def healthcheck(self) -> tuple[bool, str | None]:
        """探活。返回 (是否可用, 错误信息)。"""
        try:
            async with self.engine.connect() as conn:
                await conn.execute(sa.text("SELECT 1"))
            return True, None
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    async def dispose(self) -> None:
        await self.engine.dispose()


@lru_cache(maxsize=1)
def get_database() -> Database:
    return Database(get_settings().database_url)
