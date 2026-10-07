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
