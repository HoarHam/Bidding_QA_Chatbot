"""共享测试夹具。

**测试绝不读取仓库里的 `.env`**：它被 git 忽略，CI 里根本不存在，
而且里面是真实可用的付费密钥。所有配置都用 `Settings(_env_file=None, **overrides)`
显式构造，`config.py` 的 `model_config` 已经设了 `env_file=None`，
所以不存在第二条解析路径。

注意：这里刻意**不**加全局的「禁止网络」autouse fixture。
RESPX 的全局 mock 会与那些自己管理 router 的测试互相打架；
网络隔离改为在需要它的测试里显式声明。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bidding_qa_chatbot.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_XLSX = REPO_ROOT / "data" / "processed" / "问题答案样例数据0113.xlsx"

requires_corpus = pytest.mark.skipif(
    not CORPUS_XLSX.exists(), reason="语料文件不存在（CI 中 /data 被忽略时会出现）"
)


#: 会被 `config.load_dotenv()` 写进 os.environ 的键。
#: 测试开始前必须清掉，否则测试会读到 .env 里**真实的付费密钥**——
#: 既让断言依赖本机环境，也存在真的打到线上 API 的风险。
_ENV_KEYS = (
    "DEEPSEEK_API_KEY",
    "ZHIPU_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
    "QDRANT_VECTOR_SIZE",
    "QDRANT_DISTANCE",
    "LLM_PROVIDER",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_THINKING_MODEL",
    "ZHIPU_MODEL",
    "ZHIPU_THINKING_MODEL",
    "HF_TOKEN",
    "VLLM_API_KEY",
    "VLLM_MODEL",
    "ENABLE_INGEST_ENDPOINT",
    # Neo4j：`load_dotenv` 会把 .env 里的真实地址/密码写进 os.environ，
    # 而 `Settings(_env_file=None)` 仍会读到；不清掉的话每个测试都会
    # 静默拿到 enable_neo4j=True 与实际凭据。
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
    "NEO4J_MAX_CONNECTION_LIFETIME",
    "NEO4J_MAX_CONNECTION_POOL_SIZE",
    "NEO4J_CONNECTION_TIMEOUT",
    "NEO4J_QUERY_TIMEOUT",
    "NEO4J_CONNECTION_ACQUISITION_TIMEOUT",
    "NEO4J_MAX_ROWS",
    "ENABLE_NEO4J",
    # 联网搜索：.env 里有真实 Tavily/Exa 密钥与开关，必须从测试环境里清掉
    "TAVILY_API_KEY",
    "EXA_API_KEY",
    "ENABLE_WEB_SEARCH",
    # Ollama / 附件 OCR：.env 的值会泄漏进 Settings，测试必须使用明确的假配置，
    # 绝不能误连本机 Ollama。
    "OLLAMA_BASE_URL",
    "OLLAMA_API_KEY",
    "OLLAMA_MODEL",
    "OLLAMA_OCR_BASE_URL",
    "OLLAMA_OCR_MODEL",
    "OLLAMA_OCR_TIMEOUT",
    "OLLAMA_OCR_KEEP_ALIVE",
    "OLLAMA_OCR_NUM_CTX",
    "OLLAMA_OCR_NUM_PREDICT",
    "OLLAMA_OCR_MAX_CONCURRENCY",
    "MAX_UPLOAD_BYTES",
    "MAX_ATTACHMENT_CHARS",
    # RAG 候选缓存：.env 若配置会影响测试的请求计数与命中率断言
    "RETRIEVAL_CACHE_ENABLED",
    "RETRIEVAL_CACHE_MAXSIZE",
    "RETRIEVAL_CACHE_TTL",
    "MAX_PRIOR_ATTACHMENT_CHARS",
    "ATTACHMENT_PDF_MAX_PAGES",
    "ATTACHMENT_DOCX_MAX_IMAGES",
    "ATTACHMENT_RECENT_LIMIT",
    "UPLOADS_DIR_NAME",
    "UPLOADS_RETENTION_DAYS",
    # Langfuse：.env 里有真实密钥与云地址。SDK 会读取自己的环境变量，
    # 不清掉会让「未配置应静默禁用」的测试真的连到线上。
    "LANGFUSE_ENABLED",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_HOST",
    "LANGFUSE_CAPTURE_CONTENT",
    "LANGFUSE_TIMEOUT",
    "LANGFUSE_SAMPLE_RATE",
    "LANGFUSE_ENVIRONMENT",
    "LANGFUSE_TRACING_ENABLED",
    "LANGFUSE_TRACING_ENVIRONMENT",
    "LANGFUSE_DEBUG",
    "OTEL_SDK_DISABLED",
    # 综合评测：.env 里 EVALUATE_ENABLED=true 且配置了判分模型；测试必须使用
    # 显式关闭的假配置，绝不能真的去调用 Ragas/智谱。
    "EVALUATE_ENABLED",
    "EVALUATE_JUDGE_PROVIDER",
    "EVALUATE_JUDGE_MODEL",
    "EVALUATE_JUDGE_MAX_TOKENS",
    "EVALUATE_JUDGE_TIMEOUT",
    "EVALUATE_HISTORY_LIMIT",
    "EVALUATE_ONLINE_LIMIT",
    "EVALUATE_RAGAS_CACHE_DIR",
    "EVALUATE_RESULTS_DIR_NAME",
    "EVALUATE_CASES_FILENAME",
    "RAGAS_DO_NOT_TRACK",
    "ZHIPU_BASE_URL",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """让每个测试都在干净的环境变量下运行。

    `config.py` 在导入时执行 `load_dotenv(override=False)`，这是为了让
    `huggingface_hub` 能读到 `HF_ENDPOINT`——但它同时把 `.env` 里的
    所有键写进了 `os.environ`。若不清理，`Settings(_env_file=None)` 依然会
    从 `os.environ` 读到真实密钥，测试就不再隔离了。
    """
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture
def settings_factory():
    """构造 Settings，默认不读 .env，也不访问网络。"""

    def _make(**overrides) -> Settings:
        base = {
            "qdrant_url": "http://localhost:6333",
            "qdrant_api_key": "test-key",
            "deepseek_api_key": "sk-test",
            "zhipu_api_key": "test.id",
        }
        base.update(overrides)
        return Settings(_env_file=None, **base)

    return _make


@pytest.fixture
def settings(settings_factory) -> Settings:
    return settings_factory()


class FakeEncoder:
    """确定性假编码器：同样的文本永远得到同样的向量，便于断言。

    用字符哈希播种，保证结果可复现，且不同文本向量不同。
    """

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension
        self.calls: list[str] = []

    def _vector(self, text: str) -> list[float]:
        import hashlib
        import math

        raw = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [
            ((b / 255.0) - 0.5) for b in (raw * (self.dimension // len(raw) + 1))[: self.dimension]
        ]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return [self._vector(t) for t in texts]

    def encode_query(self, query: str) -> list[float]:
        return self._vector(query)


# ------------------------------------------------------------------ 数据库
@pytest.fixture
def auth_settings(settings_factory) -> Settings:
    """带 JWT 密钥的配置。真实环境必须显式配 JWT_SECRET，测试里给个固定值。"""
    return settings_factory(jwt_secret="test-secret-" + "x" * 40)


@pytest.fixture
async def db(auth_settings):
    """内存 SQLite 数据库（已建表）。

    CI 里没有 Postgres，模型刻意只用可移植类型（`sa.Uuid` / `sa.JSON`），
    所以 ORM 与仓储逻辑能在 SQLite 上完整验证。
    真正需要 Postgres 的行为（JSONB、并发、索引）另用 `RUN_PG_TESTS=1` 跑。
    """
    from bidding_qa_chatbot.db.session import Database

    database = Database("sqlite+aiosqlite:///:memory:", auth_settings)
    await database.create_all()
    yield database
    await database.dispose()


@pytest.fixture
async def api_client(db, auth_settings):
    """注入了测试数据库的 HTTP 客户端（不启动 lifespan，不连任何外部服务）。"""
    import httpx

    from bidding_qa_chatbot.api.app import create_app

    app = create_app(auth_settings)
    app.state.settings = auth_settings
    app.state.db = db
    # 聊天相关依赖置空——这些用例只验证账号与历史，不碰检索与模型
    app.state.chain = None
    app.state.agent = None
    app.state.encoder = None
    app.state.reranker = None
    app.state.bm25 = None
    app.state.collection_vocab_sha1 = None

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        client.app = app  # type: ignore[attr-defined] — 测试里直接注入/检查 app.state
        yield client


async def register_user(client, *, username="alice", email=None, password="pw-12345678"):
    """注册一个用户并返回 (token, user_dict)。"""
    resp = await client.post(
        "/api/auth/register",
        json={
            "username": username,
            "email": email or f"{username}@example.com",
            "password": password,
        },
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    return data["access_token"], data["user"]


def auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class FakeReranker:
    """按查询词在文本中出现的次数打分，排序可预期。"""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    def rerank(self, query: str, candidates: list[str], top_k: int | None = None):
        from bidding_qa_chatbot.embeddings.reranker import RankedItem

        scored = []
        for i, text in enumerate(candidates):
            self.seen.append((query, text))
            overlap = sum(1 for ch in set(query) if ch in text)
            scored.append(RankedItem(index=i, score=float(overlap)))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: (top_k if top_k is not None else len(scored))]
