"""FastAPI 应用工厂与生命周期。

生命周期里做三件重活，且**只在启动时做一次**：
1. 构建共享的 `httpx.AsyncClient`（连接池 + 统一超时）；
2. 创建 Qdrant 客户端；
3. 加载嵌入与精排模型（2-5 秒），并把 BM25 工件读进内存。

放在 lifespan 里而不是每个请求里，是因为模型加载如果按请求触发，
第一个用户会等好几秒，且并发请求会重复加载。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bidding_qa_chatbot.config import Settings, get_settings, validate_runtime
from bidding_qa_chatbot.log import redact, setup_logging
from bidding_qa_chatbot.retrieval import bm25 as bm25_mod
from bidding_qa_chatbot.retrieval.qdrant_store import build_client, collection_meta

logger = logging.getLogger(__name__)

#: 图谱启动探活的独立超时。驱动的 connection_timeout 是 60s，直接 await 会让
#: Neo4j 不可用时启动白等一分钟；探活失败只记日志、不阻止启动。
GRAPH_PROBE_TIMEOUT = 3.0


def _describe_db_url(url: str) -> str:
    """把连接串转成可安全打日志的形态（**不含凭据**）。"""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    scheme = (parsed.scheme or "").split("+")[0]  # postgresql+psycopg → postgresql
    host = parsed.hostname or "?"
    port = f":{parsed.port}" if parsed.port else ""
    db = (parsed.path or "").lstrip("/") or "?"
    return f"{scheme}://{host}{port}/{db}"


def build_http_client(settings: Settings) -> httpx.AsyncClient:
    """构造共享的异步 HTTP 客户端。

    单独抽成函数是为了可测：`httpx.Timeout` 要求给默认值、或四个参数全给，
    只写 `connect=` + `read=` 会在**运行时**抛 ValueError。
    这个错误只在启动时才暴露，单元测试直接构造一次最省事。
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(settings.llm_read_timeout, connect=settings.llm_connect_timeout),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        follow_redirects=True,
        # 与 Qdrant 客户端保持一致：不走系统代理（本机代理连不通部分目标）。
        trust_env=settings.http_trust_env,
    )


def build_agent(
    settings: Settings,
    chain: Any,
    qdrant: Any,
    encoder: Any,
    reranker: Any,
    bm25_artifact: Any,
    collection_vocab_sha1: str | None,
    *,
    graph: Any = None,
) -> Any:
    """构造 Agent（未启用时返回 None）。

    **刻意不检查 `encoder` 是否为 None。** Agent 路径与「本地模型是否就绪」
    是两件正交的事：前者决定由谁判断要不要检索，后者只影响工具内部能不能跑。

    模型缺失时正确的做法是让工具返回一条明确的失败信息（模型据此如实告知用户），
    而不是切到固定检索路径——那条路径在缺 encoder 时是**直接 503** 的，
    切过去只会把「部分可用」变成「完全不可用」。

    `graph`（GraphService）只在 ENABLE_NEO4J=true 时非 None；为 None 时
    不注册图谱工具（功能关闭时模型不该看到一个永远不可用的工具）。

    抽成独立函数是为了可测：这条逻辑曾经写反过，且只在运行时才暴露。
    """
    if not settings.agent_enabled:
        return None

    from bidding_qa_chatbot.agent import RagAgent, build_default_registry
    from bidding_qa_chatbot.retrieval.service import Retriever

    # 检索器无请求态，启动时构造一次复用即可
    retriever = Retriever(
        qdrant,
        encoder,
        reranker,
        settings,
        bm25_artifact=bm25_artifact,
        collection_vocab_sha1=collection_vocab_sha1,
    )
    agent = RagAgent(chain, build_default_registry(retriever, settings, graph=graph), settings)
    logger.info("Agent 已启用，工具: %s", agent.registry.names)
    if not retriever.available:
        logger.warning(
            "本地模型未就绪：Agent 仍可作答非检索类问题，但知识库检索工具会返回明确的不可用提示"
        )
    return agent


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    problems = validate_runtime(settings)
    for problem in problems:
        logger.warning("配置问题: %s", problem)

    app.state.http = build_http_client(settings)
    app.state.qdrant = build_client(settings)

    # ---- 关系数据库：账号与对话历史 ----
    from bidding_qa_chatbot.db.session import Database

    # psycopg 的异步模式**无法**在 Windows 的 ProactorEventLoop 上运行。
    # 这里主动检测并给出可操作的提示，而不是让它在第一次查询时抛一句
    # "Psycopg cannot use the 'ProactorEventLoop'"——那种报错很难定位到启动方式上。
    loop_name = type(asyncio.get_running_loop()).__name__
    if sys.platform == "win32" and "Proactor" in loop_name:
        logger.error(
            "当前事件循环是 %s，PostgreSQL 异步驱动无法在此循环上工作，"
            "账号与对话历史将不可用。请用 `uv run bidding-qa-chatbot serve` 启动"
            "（它会在 Windows 上显式使用 SelectorEventLoop）。",
            loop_name,
        )

    app.state.db = Database(settings.database_url, settings)
    db_ok, db_err = await app.state.db.healthcheck()
    if db_ok:
        # 打印**实际**连接目标，而不是 .env 里的 POSTGRES_*：DATABASE_URL 覆盖
        # 生效时两者不同，照着错的日志排查会走很远弯路。
        logger.info("数据库已连接: %s", _describe_db_url(settings.database_url))
    else:
        # 连不上不阻止服务启动：健康检查会如实报告，登录/历史相关接口自然失败，
        # 但检索与健康检查仍然可用，便于排查问题。
        logger.error("数据库连接失败（登录与对话历史将不可用）: %s", redact(str(db_err)))

    # ---- 知识图谱（Neo4j）：仅在 ENABLE_NEO4J=true 时接入 ----
    app.state.graph = None
    if settings.enable_neo4j:
        from bidding_qa_chatbot.graph import GraphService, build_driver, probe_with_timeout

        app.state.graph = GraphService(build_driver(settings), settings)
        graph_ok, graph_err = await probe_with_timeout(app.state.graph, GRAPH_PROBE_TIMEOUT)
        if graph_ok:
            # 连接串本身不含凭据（build_driver 走 auth=），可安全打日志
            logger.info(
                "知识图谱已连接: %s（数据库 %s）", settings.neo4j_uri, settings.neo4j_database
            )
        else:
            # 与数据库同样的处理：连不上不阻止启动。工具仍然注册——它会给模型一条
            # 可转述的失败信息；不注册的话模型会转而凭记忆编造交易数据。
            logger.error(
                "知识图谱连接失败（查询工具会返回明确的不可用提示）: %s", redact(str(graph_err))
            )
    else:
        logger.info("ENABLE_NEO4J 未开启：知识图谱查询工具不注册")

    # BM25 工件与集合元数据：用于判断稀疏一路能否安全使用。
    app.state.bm25 = bm25_mod.load_artifact(settings.extracted_dir / "bm25.json")
    app.state.collection_vocab_sha1 = None
    try:
        meta = await collection_meta(app.state.qdrant, settings)
        app.state.collection_vocab_sha1 = meta.get("bm25_vocab_sha1")
    except Exception as exc:  # noqa: BLE001 — 集合可能还不存在
        logger.warning("读取集合元数据失败（集合可能尚未创建）: %s", redact(str(exc)))

    # 模型加载耗时且吃内存；失败不应阻止服务启动（检索会退化，但健康检查仍可用）。
    try:
        from bidding_qa_chatbot.embeddings.dense import DenseEncoder
        from bidding_qa_chatbot.embeddings.reranker import Reranker

        app.state.encoder = DenseEncoder(settings)
        app.state.encoder.warmup()
        app.state.reranker = Reranker(settings) if settings.rerank_enabled else None
        if app.state.reranker is not None:
            app.state.reranker.warmup()
    except Exception as exc:  # noqa: BLE001
        logger.error("本地模型加载失败，检索能力将不可用: %s", redact(str(exc)))
        app.state.encoder = None
        app.state.reranker = None

    from bidding_qa_chatbot.llm import build_chain

    app.state.chain = build_chain(settings)

    # ---- Agent：把 RAG 检索注册成工具，由模型自行决定何时调用 ----
    app.state.agent = build_agent(
        settings,
        app.state.chain,
        app.state.qdrant,
        app.state.encoder,
        app.state.reranker,
        app.state.bm25,
        app.state.collection_vocab_sha1,
        graph=app.state.graph,
    )

    app.state.settings = settings

    logger.info("服务就绪：provider 链 %s", " -> ".join(settings.fallback_chain))
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.qdrant.close()
        with contextlib.suppress(Exception):
            await app.state.db.dispose()
        # Neo4j 驱动关闭失败同样不该影响进程退出
        graph = getattr(app.state, "graph", None)
        if graph is not None:
            with contextlib.suppress(Exception):
                await graph.close()
        # provider 关闭失败不该影响进程退出
        with contextlib.suppress(Exception):
            await app.state.chain.aclose()


def create_app(settings: Settings | None = None) -> FastAPI:
    setup_logging()
    app = FastAPI(
        title="招投标智能问答机器人",
        description="RAG（混合检索 + 精排）+ Agent 工具调用（知识库检索 / Neo4j 知识图谱查询）",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings or get_settings()

    # 开发期前端由 Next 代理转发，同源，不需要 CORS；
    # 保留一份宽松配置以便直接用 curl / 其他端口的前端调试。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from bidding_qa_chatbot.api.routes_auth import router as auth_router
    from bidding_qa_chatbot.api.routes_chat import router as chat_router
    from bidding_qa_chatbot.api.routes_conversations import router as conversations_router
    from bidding_qa_chatbot.api.routes_health import router as health_router
    from bidding_qa_chatbot.api.routes_sources import router as sources_router

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(conversations_router)
    app.include_router(sources_router)
    app.include_router(chat_router)
    return app


app = create_app()
