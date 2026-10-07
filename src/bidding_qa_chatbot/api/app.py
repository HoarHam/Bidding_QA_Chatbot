"""FastAPI 应用工厂与生命周期。

生命周期里做三件重活，且**只在启动时做一次**：
1. 构建共享的 `httpx.AsyncClient`（连接池 + 统一超时）；
2. 创建 Qdrant 客户端；
3. 加载嵌入与精排模型（2-5 秒），并把 BM25 工件读进内存。

放在 lifespan 里而不是每个请求里，是因为模型加载如果按请求触发，
第一个用户会等好几秒，且并发请求会重复加载。
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bidding_qa_chatbot.config import Settings, get_settings, validate_runtime
from bidding_qa_chatbot.log import redact, setup_logging
from bidding_qa_chatbot.retrieval import bm25 as bm25_mod
from bidding_qa_chatbot.retrieval.qdrant_store import build_client, collection_meta

logger = logging.getLogger(__name__)


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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    problems = validate_runtime(settings)
    for problem in problems:
        logger.warning("配置问题: %s", problem)

    app.state.http = build_http_client(settings)
    app.state.qdrant = build_client(settings)

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
    app.state.settings = settings

    logger.info("服务就绪：provider 链 %s", " -> ".join(settings.fallback_chain))
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.qdrant.close()
        # provider 关闭失败不该影响进程退出
        with contextlib.suppress(Exception):
            await app.state.chain.aclose()


def create_app(settings: Settings | None = None) -> FastAPI:
    setup_logging()
    app = FastAPI(
        title="招投标智能问答机器人",
        description="第一阶段：RAG（混合检索 + 精排 + 可切换 LLM）",
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

    from bidding_qa_chatbot.api.routes_chat import router as chat_router
    from bidding_qa_chatbot.api.routes_health import router as health_router
    from bidding_qa_chatbot.api.routes_sources import router as sources_router

    app.include_router(health_router)
    app.include_router(sources_router)
    app.include_router(chat_router)
    return app


app = create_app()
