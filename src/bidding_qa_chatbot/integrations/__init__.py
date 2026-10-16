"""尚未实现的第二阶段集成（联网搜索）的最小接口桩。

本包**刻意不实现任何业务逻辑**，也**刻意不导入** `tavily-python` 等驱动——
它们留在 `pyproject.toml` 的 `[project.optional-dependencies].phase2` 里，
本阶段安装时根本不存在。

那为什么还要有这些文件？因为 `.env` 里已经有 Tavily / Exa 的配置项，
而「配置项写了却完全没人读」是最容易腐烂的状态。这里把配置读进来、
把可达性探测定下来（用裸 TCP，不需要驱动），第三阶段接入时只需填实现。

PostgreSQL（`bidding_qa_chatbot/db/`）与 Neo4j（`bidding_qa_chatbot/graph/`）
**已不是桩**，实现在各自包内，驱动也已提升为主依赖。
"""

from bidding_qa_chatbot.integrations.stubs import ExaStub, TavilyStub, build_phase2_status

__all__ = [
    "ExaStub",
    "TavilyStub",
    "build_phase2_status",
]
