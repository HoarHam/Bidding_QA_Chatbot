"""健康检查与统计。

`/api/health` 是存活探针，**绝不触碰任何外部依赖**——它被调用得最频繁，
一旦它依赖外部服务，外部抖动就会把整个服务标记为不健康。

`/api/health/deps` 才做逐项可达性探测，且各项独立超时并发执行，
一个挂掉的依赖不能拖住整个端点。

数据库类依赖（PostgreSQL / Neo4j）做**真实鉴权探测**而非裸 TCP：
端口通不代表库存在、凭据正确、数据已就绪。

联网搜索（Tavily / Exa）只做**配置级探测**（密钥是否配置、Exa 所需的 npx 是否存在）：
搜索探活会消耗付费额度，不适合放进健康检查。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import time
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Request

from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.log import redact
from bidding_qa_chatbot.schemas import DepHealth, HealthResponse, StatsResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["health"])

PROBE_TIMEOUT = 3.0


async def _probe_qdrant(settings: Settings, client: httpx.AsyncClient) -> DepHealth:
    if not settings.qdrant_url:
        return DepHealth(name="qdrant", status="not_configured", message="QDRANT_URL 未配置")
    started = time.perf_counter()
    try:
        resp = await client.get(
            f"{settings.qdrant_url.rstrip('/')}/collections",
            headers=_qdrant_headers(settings),
            timeout=PROBE_TIMEOUT,
        )
        latency = int((time.perf_counter() - started) * 1000)
        if resp.status_code == 401:
            return DepHealth(
                name="qdrant",
                status="unavailable",
                latency_ms=latency,
                message="认证失败 (401)，请检查 QDRANT_API_KEY",
            )
        if resp.status_code in (403, 503):
            # Qdrant Cloud 闲置一周会暂停集群，这是可恢复状态而非配置错误。
            return DepHealth(
                name="qdrant",
                status="degraded",
                latency_ms=latency,
                message="集群可能已暂停，请到 Qdrant Cloud 控制台 Resume",
            )
        resp.raise_for_status()
        return DepHealth(name="qdrant", status="ok", latency_ms=latency)
    except Exception as exc:  # noqa: BLE001
        return DepHealth(
            name="qdrant",
            status="unavailable",
            message=redact(f"{type(exc).__name__}: {exc}")[:200],
        )


def _qdrant_headers(settings: Settings) -> dict[str, str]:
    if settings.qdrant_api_key:
        return {"api-key": settings.qdrant_api_key.get_secret_value()}
    return {}


async def _probe_openai_compatible(
    name: str, base_url: str, api_key: str | None, client: httpx.AsyncClient
) -> DepHealth:
    if not base_url:
        return DepHealth(name=name, status="not_configured")
    started = time.perf_counter()
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # /models 不消耗 token，是零成本探针。
        resp = await client.get(
            f"{base_url.rstrip('/')}/models", headers=headers, timeout=PROBE_TIMEOUT
        )
        latency = int((time.perf_counter() - started) * 1000)
        if resp.status_code in (401, 403):
            return DepHealth(
                name=name,
                status="unavailable",
                latency_ms=latency,
                message=f"认证失败 ({resp.status_code})",
            )
        if resp.status_code >= 500:
            return DepHealth(
                name=name, status="degraded", latency_ms=latency, message=f"HTTP {resp.status_code}"
            )
        return DepHealth(name=name, status="ok", latency_ms=latency)
    except Exception as exc:  # noqa: BLE001
        return DepHealth(
            name=name, status="unavailable", message=redact(f"{type(exc).__name__}: {exc}")[:200]
        )


async def _probe_tcp(name: str, host: str, port: int) -> DepHealth:
    """裸 TCP 探测，不需要任何数据库驱动。"""
    started = time.perf_counter()
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=PROBE_TIMEOUT
        )
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        return DepHealth(
            name=name, status="ok", latency_ms=int((time.perf_counter() - started) * 1000)
        )
    except Exception as exc:  # noqa: BLE001
        return DepHealth(
            name=name, status="unavailable", message=redact(f"{type(exc).__name__}: {exc}")[:200]
        )


def _host_port(uri: str, default_port: int) -> tuple[str, int]:
    parsed = urlparse(uri)
    return parsed.hostname or "localhost", parsed.port or default_port


async def _probe_zhipu(settings: Settings) -> DepHealth:
    """zai-sdk 没有 /models 风格的探针，做 TCP+TLS 连通性检查即可。"""
    if settings.zhipu_api_key is None:
        return DepHealth(name="zhipu", status="not_configured")
    host, port = _host_port(settings.zhipu_base_url, 443)
    return await _probe_tcp("zhipu", host, port)


async def _probe_database(settings: Settings, db: Any = None) -> DepHealth:
    """探测关系数据库。

    比 TCP 探测严格：端口通不代表库存在、凭据正确、表已建好。
    用一次真实连接（`SELECT 1`）来判断。
    """
    if not settings.postgres_db:
        return DepHealth(name="postgres", status="not_configured", message="POSTGRES_DB 未配置")

    if db is None:
        from bidding_qa_chatbot.db.session import Database

        db = Database(settings.database_url, settings)

    ok, error = await db.healthcheck()
    if ok:
        return DepHealth(name="postgres", status="ok")
    return DepHealth(
        name="postgres",
        status="unavailable",
        message=redact(str(error))[:200],
    )


async def _probe_graph(settings: Settings, graph: Any = None) -> DepHealth:
    """探测知识图谱（Neo4j）。

    同样比 TCP 严格：端口通不代表数据库存在、凭据正确。走驱动的真实连接
    （`RETURN 1`）+ 独立超时（驱动 connection_timeout 是 60s，不能直连等）。
    `graph` 为 None 时（CLI 路径）临时建一个驱动，用完关掉。
    """
    if not settings.enable_neo4j:
        return DepHealth(
            name="neo4j", status="not_configured", message="未启用（ENABLE_NEO4J=false）"
        )

    from bidding_qa_chatbot.graph import GraphService, build_driver, probe_with_timeout

    if graph is None:
        temp = GraphService(build_driver(settings), settings)
        try:
            ok, error = await probe_with_timeout(temp, PROBE_TIMEOUT)
        finally:
            with contextlib.suppress(Exception):
                await temp.close()
    else:
        ok, error = await probe_with_timeout(graph, PROBE_TIMEOUT)

    if ok:
        return DepHealth(name="neo4j", status="ok")
    return DepHealth(name="neo4j", status="unavailable", message=redact(str(error))[:200])


async def _probe_tavily(settings: Settings) -> DepHealth:
    """Tavily 关键词搜索：只检查配置，不发付费请求。"""
    if not settings.enable_web_search:
        return DepHealth(
            name="tavily", status="not_configured", message="未启用（ENABLE_WEB_SEARCH=false）"
        )
    if settings.tavily_api_key is None:
        return DepHealth(name="tavily", status="not_configured", message="TAVILY_API_KEY 未配置")
    return DepHealth(name="tavily", status="ok", message="已配置（不做付费探活）")


async def _probe_exa(settings: Settings) -> DepHealth:
    """Exa 语义搜索：主路径是 HTTP API，npx 只影响 MCP 回退能力。不发付费请求。"""
    if not settings.enable_web_search:
        return DepHealth(
            name="exa", status="not_configured", message="未启用（ENABLE_WEB_SEARCH=false）"
        )
    if settings.exa_api_key is None:
        return DepHealth(name="exa", status="not_configured", message="EXA_API_KEY 未配置")
    if shutil.which("npx") is None:
        return DepHealth(
            name="exa",
            status="ok",
            message="已配置（主路径 Exa API；未找到 npx，MCP 回退不可用）",
        )
    return DepHealth(name="exa", status="ok", message="已配置（不做付费探活）")


async def probe_all(
    settings: Settings, client: httpx.AsyncClient, db: Any = None, graph: Any = None
) -> list[DepHealth]:
    """并发探测所有依赖。CLI 的 `check` 与 `/api/health/deps` 共用这一份实现。"""
    tasks = [
        _probe_qdrant(settings, client),
        _probe_openai_compatible(
            "deepseek",
            settings.deepseek_base_url,
            settings.deepseek_api_key.get_secret_value() if settings.deepseek_api_key else None,
            client,
        ),
        _probe_zhipu(settings),
        _probe_openai_compatible(
            "vllm",
            settings.vllm_base_url,
            settings.vllm_api_key.get_secret_value() if settings.vllm_api_key else None,
            client,
        ),
        _probe_openai_compatible(
            "ollama",
            settings.ollama_base_url,
            settings.ollama_api_key.get_secret_value() if settings.ollama_api_key else None,
            client,
        ),
    ]

    # 数据库与图谱都是实打实的功能依赖（账号/历史、图谱查询），
    # 不只做 TCP 探测：端口通不代表库存在、凭据正确、数据已就绪。
    tasks.append(_probe_graph(settings, graph))
    tasks.append(_probe_database(settings, db))
    # 联网搜索只做配置级探测（密钥 / npx），不消耗付费额度
    tasks.append(_probe_tavily(settings))
    tasks.append(_probe_exa(settings))

    return list(await asyncio.gather(*tasks))


#: CLI 侧的别名，语义更直白。
health_deps_for_cli = probe_all


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """存活探针：不触碰任何外部依赖。"""
    return HealthResponse(status="ok", deps=[])


@router.get("/health/deps", response_model=HealthResponse)
async def health_deps(request: Request) -> HealthResponse:
    settings: Settings = request.app.state.settings
    client: httpx.AsyncClient = request.app.state.http

    deps = await probe_all(
        settings,
        client,
        getattr(request.app.state, "db", None),
        getattr(request.app.state, "graph", None),
    )
    overall = "ok" if all(d.status in ("ok", "not_configured") for d in deps) else "degraded"
    return HealthResponse(status=overall, deps=deps)


async def _not_configured(name: str) -> DepHealth:
    return DepHealth(name=name, status="not_configured", message="未启用")


@router.get("/stats", response_model=StatsResponse)
async def stats(request: Request) -> StatsResponse:
    settings: Settings = request.app.state.settings
    chunks: int | None = None
    try:
        from bidding_qa_chatbot.retrieval.qdrant_store import count_points

        chunks = await count_points(request.app.state.qdrant, settings)
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取集合点数失败: %s", redact(str(exc)))

    # 前端据此决定「联网搜索」开关是否可点：服务端不可用时置灰，
    # 避免用户开了开关却静默无效。
    web_search = getattr(request.app.state, "web_search", None)
    web_search_available = bool(web_search is not None and web_search.available)

    return StatsResponse(
        collection=settings.qdrant_collection,
        vector_size=settings.qdrant_vector_size,
        chunks=chunks,
        provider_chain=settings.fallback_chain,
        embed_model=settings.embed_model_name,
        rerank_model=settings.rerank_model_name,
        rerank_enabled=settings.rerank_enabled,
        web_search_available=web_search_available,
    )
