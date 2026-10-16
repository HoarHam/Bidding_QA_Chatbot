"""尚未实现的集成（联网搜索）的接口桩实现。

每个桩都满足两件事：
1. 能报告「是否已配置」（读 `.env`）；
2. 能报告「是否就绪」（配置项齐不齐）。

任何实际查询都抛 `NotImplementedError` 并带上明确的阶段说明，
避免调用方误以为功能已经可用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.schemas import DepHealth

logger = logging.getLogger(__name__)

_STAGE = "第二阶段功能，当前阶段未实现"


@dataclass(slots=True)
class _BaseStub:
    settings: Settings
    name: str
    enabled: bool

    def status(self) -> str:
        if not self.enabled:
            return "not_configured"
        return "configured"


# 注：PostgreSQL 的桩已移除——账号与对话历史**已经实现**，
# 见 `bidding_qa_chatbot/db/`（SQLAlchemy 模型 + 仓储 + Alembic 迁移），
# 健康检查也改成了真实连接探测而非裸 TCP。
# 同理，Neo4j 知识图谱的桩也已移除——查询**已经实现**，
# 见 `bidding_qa_chatbot/graph/`（异步驱动 + 只读 Cypher 校验 + 参数化模板）。
# 仍然只是桩的只有下面几个。


class TavilyStub(_BaseStub):
    """第二阶段：关键词联网搜索。"""

    def __init__(self, settings: Settings, name: str = "tavily") -> None:
        super().__init__(settings, name, settings.enable_web_search)

    async def probe(self) -> DepHealth:
        if not self.enabled:
            return DepHealth(name=self.name, status="not_configured", message=_STAGE)
        if self.settings.tavily_api_key is None:
            return DepHealth(
                name=self.name, status="not_configured", message="TAVILY_API_KEY 未配置"
            )
        return DepHealth(name=self.name, status="ok", message="已配置（未启用）")

    async def search(self, query: str) -> None:
        raise NotImplementedError(f"Tavily 联网搜索属于{_STAGE}")


class ExaStub(_BaseStub):
    """第二阶段：语义联网搜索（通过 npx 拉起的 ExaMCPServer）。"""

    def __init__(self, settings: Settings, name: str = "exa") -> None:
        super().__init__(settings, name, settings.enable_web_search)

    async def probe(self) -> DepHealth:
        if not self.enabled:
            return DepHealth(name=self.name, status="not_configured", message=_STAGE)
        if self.settings.exa_api_key is None:
            return DepHealth(name=self.name, status="not_configured", message="EXA_API_KEY 未配置")
        return DepHealth(name=self.name, status="ok", message="已配置（未启用）")

    async def search(self, query: str) -> None:
        raise NotImplementedError(f"Exa 语义搜索属于{_STAGE}")


def build_phase2_status(settings: Settings) -> dict[str, str]:
    """汇总**尚未实现**的集成状态，供 `/api/health/deps` 或文档使用。

    不含 postgres（见 `bidding_qa_chatbot/db/`）与 neo4j
    （见 `bidding_qa_chatbot/graph/`）：两者都已实现，继续列在「未实现」里会误导。
    """
    return {
        "web_search": "enabled" if settings.enable_web_search else "disabled",
        "stage": "phase2",
    }
