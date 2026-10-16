"""Neo4j 异步驱动的构造。"""

from __future__ import annotations

from typing import Any

from neo4j import AsyncGraphDatabase

from bidding_qa_chatbot.config import Settings, get_settings


def build_driver(settings: Settings | None = None, **overrides: Any) -> Any:
    """构造异步驱动。**不会立刻建连**——首次真实查询时才建连，启动成本近似为零。

    凭据走 `auth=(user, password)`，**不拼进 URI**：拼进 URI 后驱动一旦报错，
    连接串会连同密码一起出现在异常消息里——而异常消息会进日志与健康检查响应。
    """
    settings = settings or get_settings()
    password = settings.neo4j_password.get_secret_value() if settings.neo4j_password else ""
    return AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, password),
        max_connection_lifetime=settings.neo4j_max_connection_lifetime,
        max_connection_pool_size=settings.neo4j_max_connection_pool_size,
        connection_timeout=settings.neo4j_connection_timeout,
        # 取连接的最长等待：驱动默认 60s，在 SSE 流里等于界面卡死
        connection_acquisition_timeout=settings.neo4j_connection_acquisition_timeout,
        **overrides,
    )
