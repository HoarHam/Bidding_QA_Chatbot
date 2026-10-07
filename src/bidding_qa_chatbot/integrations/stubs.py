"""第二阶段集成的接口桩实现。

每个桩都满足两件事：
1. 能报告「是否已配置」（读 `.env`）；
2. 能报告「是否可达」（裸 TCP / HTTP，无需第三方驱动）。

任何实际查询都抛 `NotImplementedError` 并带上明确的阶段说明，
避免调用方误以为功能已经可用。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.log import redact
from bidding_qa_chatbot.schemas import DepHealth

logger = logging.getLogger(__name__)

_STAGE = "第二阶段功能，当前阶段未实现"
_PROBE_TIMEOUT = 3.0


@dataclass(slots=True)
class _BaseStub:
    settings: Settings
    name: str
    enabled: bool

    def status(self) -> str:
        if not self.enabled:
            return "not_configured"
        return "configured"

    async def _tcp_probe(self, host: str, port: int) -> DepHealth:
        if not self.enabled:
            return DepHealth(name=self.name, status="not_configured", message=_STAGE)
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=_PROBE_TIMEOUT
            )
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            return DepHealth(name=self.name, status="ok")
        except Exception as exc:  # noqa: BLE001
            return DepHealth(
                name=self.name,
                status="unavailable",
                message=redact(f"{type(exc).__name__}: {exc}")[:200],
            )


class Neo4jStub(_BaseStub):
    """第二阶段：招投标知识图谱。"""

    def __init__(self, settings: Settings, name: str = "neo4j") -> None:
        super().__init__(settings, name, settings.enable_neo4j)

    async def probe(self) -> DepHealth:
        parsed = urlparse(self.settings.neo4j_uri)
        return await self._tcp_probe(parsed.hostname or "localhost", parsed.port or 7687)

    async def query(self, cypher: str, **params: object) -> None:
        raise NotImplementedError(f"Neo4j 知识图谱查询属于{_STAGE}")


class PostgresStub(_BaseStub):
    """第二阶段：招投标数据、会话记录与用户反馈。"""

    def __init__(self, settings: Settings, name: str = "postgres") -> None:
        super().__init__(settings, name, settings.enable_postgres)

    async def probe(self) -> DepHealth:
        return await self._tcp_probe(self.settings.postgres_host, self.settings.postgres_port)

    async def session_store(self) -> None:
        raise NotImplementedError(f"会话记录持久化属于{_STAGE}")


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
    """汇总第二阶段的启用状态，供 `/api/health/deps` 或文档使用。"""
    return {
        "neo4j": "enabled" if settings.enable_neo4j else "disabled",
        "postgres": "enabled" if settings.enable_postgres else "disabled",
        "web_search": "enabled" if settings.enable_web_search else "disabled",
        "stage": "phase2",
    }
