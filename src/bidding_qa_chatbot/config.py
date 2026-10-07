"""应用配置。

两条重要约束，改这个文件前务必先读：

1. **`load_dotenv()` 必须在本模块导入时立即执行**，且早于任何 `huggingface_hub` /
   `sentence_transformers` 的导入。`huggingface_hub` 在 import 时读取 `HF_ENDPOINT`
   环境变量，而本机无法直连 huggingface.co——如果顺序错了，模型下载会先卡住几分钟
   再报一个难以定位的网络错误。因此 `embeddings/*` 里的模型导入一律延迟到函数内部。

2. **`override=False`**：真实环境变量 > `.env` > 代码默认值。CI 里没有 `.env` 文件，
   靠环境变量注入，所以测试必须显式构造 `Settings(_env_file=None, ...)`。

字段名与 `.env` 中的 key 一一对应（`case_sensitive=False`，故 `llm_provider`
对应 `LLM_PROVIDER`）。少数 `.env` 里没有的 key（如 base_url、视觉模型名）给默认值，
保证现有 `.env` 无需改动即可工作。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR / ".env"

# 必须在构造 Settings 之前执行：第三方库（huggingface_hub 等）直接读 os.environ，
# 它们看不到 pydantic-settings 解析出的对象。
load_dotenv(dotenv_path=ENV_FILE, override=False)


class Settings(BaseSettings):
    # env_file=None：.env 只由上面那次 load_dotenv 解析一次，
    # 避免 pydantic-settings 再解析一遍导致两套优先级互相打架。
    model_config = SettingsConfigDict(env_file=None, extra="ignore", case_sensitive=False)

    # ---------- LLM (需求 4/5) ----------
    llm_provider: Literal["deepseek", "zhipu", "vllm", "ollama"] = "deepseek"
    deepseek_api_key: SecretStr | None = None
    zhipu_api_key: SecretStr | None = None
    deepseek_model: str = "deepseek-flash"
    deepseek_thinking_model: str = "deepseek-flash"
    zhipu_model: str = "glm-4.7-flashx"
    zhipu_thinking_model: str = "glm-4.7-flashx"

    # 以下 key 在 .env 中不存在，给默认值以免强迫用户改 .env。
    deepseek_base_url: str = "https://api.deepseek.com"
    # zai-sdk 默认指向国际站 api.z.ai，而 .env 里的 key 是国内站 bigmodel.cn 的
    # {id}.{secret} 格式（SDK 内部按 '.' 切分并签 HS256 JWT）。必须显式覆盖。
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    zhipu_vision_model: str = "glm-4.6v-flashx"  # 需求 6：图片解析
    llm_fallback_chain: str = "deepseek,zhipu,vllm"
    llm_connect_timeout: float = 5.0
    # 首 token 超时是「快速失败切换」的关键：比读超时短得多，
    # 才能在 provider 无响应时立刻换下一个，而不是干等 60 秒。
    llm_first_token_timeout: float = 15.0
    llm_read_timeout: float = 60.0
    llm_breaker_failures: int = 3
    llm_breaker_open_seconds: int = 60
    stream_restart_on_failure: bool = False

    # ---------- Qdrant (需求 8) ----------
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "bid_qa"
    qdrant_vector_size: int = 512
    qdrant_distance: str = "cosine"
    qdrant_timeout: int = 30
    # 是否让 httpx 读取系统代理（Windows 注册表 / 环境变量）。
    # 默认 False：本机开启了系统代理（127.0.0.1:10808），httpx 默认会走它，
    # 而该代理连不通 Qdrant Cloud，表现为 ConnectError；直连反而正常（实测）。
    # 若你的网络确实必须经代理，设为 true。
    http_trust_env: bool = False

    # ---------- 检索调参 ----------
    retrieval_prefetch_dense: int = 20
    retrieval_prefetch_sparse: int = 20
    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    rerank_enabled: bool = True
    # 默认不做分数截断：sentence-transformers 对单标签 CrossEncoder 会套一层
    # activation，predict() 返回的是 (0,1) 而非原始 logit，网上流传的 logit 阈值
    # 在这里不适用。先观察真实分布再调。
    rerank_min_score: float = 0.0
    max_context_chars: int = 6000
    sparse_strict: bool = False

    # ---------- 嵌入模型 (需求 6) ----------
    embed_model_name: str = "BAAI/bge-small-zh-v1.5"
    rerank_model_name: str = "BAAI/bge-reranker-base"
    # bge-zh 系列要求 query 侧加指令前缀、passage 侧不加。
    # 索引与查询必须用同一个值，否则分数会无声退化——所以放在配置里而非散在代码中。
    embed_query_instruction: str = "为这个句子生成表示以用于检索相关文章："

    # ---------- HuggingFace / 本地推理 ----------
    hf_token: SecretStr | None = None
    hf_endpoint: str = "https://hf-mirror.com"
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: SecretStr | None = None
    # 这是 vLLM 服务端的文件系统路径，必须原样透传。
    vllm_model: str = ""
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_api_key: SecretStr | None = None
    ollama_model: str = "deepseek-r1:8b"

    # ---------- API 服务 ----------
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    enable_ingest_endpoint: bool = False
    ingest_token: SecretStr | None = None

    # ---------- 第二阶段：仅配置与可达性探测，不实现业务逻辑 ----------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: SecretStr | None = None
    neo4j_database: str = "BiddingRelations"
    neo4j_max_connection_lifetime: int = 3600
    neo4j_max_connection_pool_size: int = 50
    neo4j_connection_timeout: int = 60
    postgres_user: str | None = None
    postgres_password: SecretStr | None = None
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str | None = None
    tavily_api_key: SecretStr | None = None
    exa_api_key: SecretStr | None = None
    enable_neo4j: bool = False
    enable_postgres: bool = False
    enable_web_search: bool = False

    # ---------- 路径 ----------
    data_dir: Path = Field(default=ROOT_DIR / "data")
    extracted_dir_name: str = "extracted"

    @model_validator(mode="after")
    def _export_hf_endpoint(self) -> Settings:
        """把 HF_ENDPOINT 写回 os.environ。

        子进程（如 `huggingface-cli`）与部分库只认环境变量，不认 Settings 对象。
        本机直连 huggingface.co 会超时，因此这一步是模型能否下载的前提。
        """
        if self.hf_endpoint and not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = self.hf_endpoint
        if self.hf_token is not None and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = self.hf_token.get_secret_value()
        return self

    # ---------- 派生属性 ----------
    @property
    def extracted_dir(self) -> Path:
        return self.data_dir / self.extracted_dir_name

    @property
    def images_dir(self) -> Path:
        return self.extracted_dir / "images"

    @property
    def corpus_cache_path(self) -> Path:
        return self.extracted_dir / "corpus.jsonl"

    @property
    def vision_cache_path(self) -> Path:
        return self.extracted_dir / "vision_cache.json"

    @property
    def fallback_chain(self) -> list[str]:
        """主 provider 置顶，其余按配置顺序跟随，去重且保持稳定。"""
        others = [p.strip() for p in self.llm_fallback_chain.split(",") if p.strip()]
        chain = [self.llm_provider] + [p for p in others if p != self.llm_provider]
        seen: set[str] = set()
        return [p for p in chain if not (p in seen or seen.add(p))]

    def model_for(self, provider: str, mode: str) -> str:
        """按 provider + 模式取模型名。

        注意 `DEEPSEEK_MODEL` 与 `DEEPSEEK_THINKING_MODEL` 当前都是 `deepseek-flash`，
        这是有意为之：同一个模型，靠请求参数切换思考模式，而不是换模型名。
        保留两个 key 是为了将来真拆成两个模型时改配置即可。
        """
        thinking = mode == "thinking"
        if provider == "deepseek":
            return self.deepseek_thinking_model if thinking else self.deepseek_model
        if provider == "zhipu":
            return self.zhipu_thinking_model if thinking else self.zhipu_model
        if provider == "vllm":
            return self.vllm_model
        if provider == "ollama":
            return self.ollama_model
        raise ValueError(f"未知 provider: {provider}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def validate_runtime(settings: Settings) -> list[str]:
    """返回配置问题列表（空列表代表通过）。

    一次性汇总所有问题而非抛第一个异常——启动时把缺什么、错什么一次说清。
    """
    problems: list[str] = []
    if not settings.qdrant_url:
        problems.append("QDRANT_URL 未配置")
    if settings.qdrant_api_key is None:
        problems.append("QDRANT_API_KEY 未配置")
    if settings.qdrant_distance.lower() not in {"cosine", "dot", "euclid", "manhattan"}:
        problems.append(f"QDRANT_DISTANCE 取值非法: {settings.qdrant_distance}")
    if settings.qdrant_vector_size != 512:
        problems.append(
            f"QDRANT_VECTOR_SIZE={settings.qdrant_vector_size}，"
            f"但 {settings.embed_model_name} 输出 512 维，两者必须一致"
        )
    for provider in settings.fallback_chain:
        if provider == "deepseek" and settings.deepseek_api_key is None:
            problems.append("DEEPSEEK_API_KEY 未配置")
        if provider == "zhipu" and settings.zhipu_api_key is None:
            problems.append("ZHIPU_API_KEY 未配置")
    return problems
