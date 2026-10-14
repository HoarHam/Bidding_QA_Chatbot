"""第二阶段集成的最小接口桩。

本包**刻意不实现任何业务逻辑**，也**刻意不导入** `neo4j` / `psycopg` /
`sqlalchemy` / `tavily-python` 等驱动——它们留在
`pyproject.toml` 的 `[project.optional-dependencies].phase2` 里，
本阶段安装时根本不存在。

那为什么还要有这些文件？因为 `.env` 里已经有 Neo4j / PostgreSQL / Tavily / Exa
的配置项，而「配置项写了却完全没人读」是最容易腐烂的状态。这里把配置读进来、
把可达性探测定下来（用裸 TCP，不需要驱动），第二阶段接入时只需填实现。
"""

from bidding_qa_chatbot.integrations.stubs import (
    ExaStub,
    Neo4jStub,
    TavilyStub,
    build_phase2_status,
)

# 注意：PostgreSQL **已不是桩**，账号与对话历史的实现在 bidding_qa_chatbot/db/
__all__ = [
    "ExaStub",
    "Neo4jStub",
    "TavilyStub",
    "build_phase2_status",
]
