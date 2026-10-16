"""知识图谱查询服务。

三条硬约束：

1. **所有查询在只读事务（`execute_read`）里执行** —— 读事务中的写操作由服务端直接
   拒绝，这是真正的安全边界（`guard.py` 的文本校验只是可用性层）。
2. **事务超时通过回调函数的 `timeout` 属性传入**（驱动 6.x 的写法，见 `run_cypher`
   内的注释；`tx.run(timeout=...)` 是 4.x 写法，已移除）。
3. **行数在客户端截断**：一次多取一行用于判断「是否被截断」，绝不把整表拉进内存。

异常统一映射为 `GraphQueryError`（含中文、已脱敏的消息，可直接回灌给模型）。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from neo4j.exceptions import AuthError, ConfigurationError, Neo4jError, ServiceUnavailable

from bidding_qa_chatbot.config import Settings, get_settings
from bidding_qa_chatbot.log import redact


@dataclass(slots=True)
class GraphQueryResult:
    rows: list[dict[str, Any]]
    truncated: bool  # 因上限被裁掉了至少一行
    elapsed_ms: int


class GraphQueryError(RuntimeError):
    """查询失败。消息已脱敏、可直接回灌给模型。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class GraphService:
    def __init__(self, driver: Any, settings: Settings | None = None) -> None:
        self.driver = driver  # 鸭子类型：测试注入 FakeDriver
        self.settings = settings or get_settings()

    @property
    def available(self) -> bool:
        """驱动是否已被创建（**静态**能力位，语义同 `Retriever.available`）。

        不把运行期探活失败置 False：Neo4j 恢复后工具应自动可用，
        而不是必须重启服务；服务器是否活着由查询抛 `GraphQueryError` 表达。
        """
        return self.driver is not None

    async def healthcheck(self) -> tuple[bool, str | None]:
        """真实连接探测（含鉴权与数据库切换）。永不抛异常。"""
        if self.driver is None:
            return False, "知识图谱未接入（ENABLE_NEO4J=false）"
        try:
            async with self.driver.session(
                database=self.settings.neo4j_database, default_access_mode="READ"
            ) as session:
                result = await session.run("RETURN 1 AS ok")
                await result.single()
            return True, None
        except Exception as exc:  # noqa: BLE001 — 探活不抛
            return False, f"{type(exc).__name__}: {exc}"

    async def run_cypher(
        self, cypher: str, params: dict[str, Any] | None = None, *, max_rows: int | None = None
    ) -> GraphQueryResult:
        """执行只读查询。失败抛 `GraphQueryError`（code 供上层映射中文标签）。"""
        if self.driver is None:
            raise GraphQueryError("graph_unavailable", "知识图谱未接入（ENABLE_NEO4J=false）")

        cap = max(1, min(max_rows or self.settings.neo4j_max_rows, self.settings.neo4j_max_rows))
        timeout = float(self.settings.neo4j_query_timeout)
        started = time.perf_counter()

        async def _work(tx: Any) -> list[dict[str, Any]]:
            result = await tx.run(cypher, dict(params or {}))
            # 多取一行：用它判断「是否被截断」，比 COUNT(*) 便宜，也避免全量拉取
            records = await result.fetch(cap + 1)
            return [record.data() for record in records]

        # 驱动 6.x：`execute_read` 从**回调函数的 timeout 属性**读取事务超时
        # （neo4j/_async/work/session.py 的 _run_transaction 用 getattr 取）。
        # **这一行不是死代码，不要删**——删掉会静默失去服务端查询超时。
        _work.timeout = timeout  # type: ignore[attr-defined]

        try:
            async with self.driver.session(
                database=self.settings.neo4j_database,
                default_access_mode="READ",  # 双保险：读事务里服务端直接拒绝写操作
            ) as session:
                rows = await asyncio.wait_for(session.execute_read(_work), timeout=timeout + 2.0)
        except TimeoutError as exc:
            raise GraphQueryError(
                "graph_timeout", f"知识图谱查询超时（超过 {timeout:.0f} 秒）"
            ) from exc
        except (ServiceUnavailable, AuthError, ConfigurationError) as exc:
            raise GraphQueryError(
                "graph_query_failed",
                f"无法连接知识图谱（{type(exc).__name__}）："
                "请确认 Neo4j 已启动、NEO4J_URI/USERNAME/PASSWORD 正确",
            ) from exc
        except Neo4jError as exc:
            if "TransactionTimedOut" in (getattr(exc, "code", "") or ""):
                raise GraphQueryError(
                    "graph_timeout", f"知识图谱查询超时（超过 {timeout:.0f} 秒）"
                ) from exc
            raise GraphQueryError("graph_query_failed", _explain(exc)) from exc
        except Exception as exc:  # noqa: BLE001 — 取结果阶段的断连等也归入查询失败
            raise GraphQueryError("graph_query_failed", _explain(exc)) from exc

        elapsed = int((time.perf_counter() - started) * 1000)
        truncated = len(rows) > cap
        return GraphQueryResult(rows=rows[:cap], truncated=truncated, elapsed_ms=elapsed)

    async def close(self) -> None:
        if self.driver is not None:
            await self.driver.close()


def _explain(exc: Exception) -> str:
    return f"知识图谱查询失败：{redact(f'{type(exc).__name__}: {exc}')[:200]}"


async def probe_with_timeout(
    service: GraphService, timeout: float = 3.0
) -> tuple[bool, str | None]:
    """带独立超时的探活。

    驱动 `connection_timeout` 默认 60s：Neo4j 不在时直接 `await healthcheck()` 会让
    启动白等一分钟。**所有探活调用方都必须走这个包装。**
    """
    try:
        return await asyncio.wait_for(service.healthcheck(), timeout=timeout)
    except TimeoutError:
        return False, f"探活超时（{timeout:.0f} 秒）"
