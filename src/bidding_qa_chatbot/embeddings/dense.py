"""Dense 语义编码：BAAI/bge-small-zh-v1.5（512 维）。

注意本文件**没有**顶层 `import sentence_transformers`——原因见包 docstring。

bge-zh v1.5 是非对称检索模型：query 侧要加指令前缀，passage 侧不加。
两边用错会导致分数无声退化，所以前缀值放在 `Settings.embed_query_instruction`，
索引与查询共用同一份配置。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from bidding_qa_chatbot.config import Settings, get_settings

if TYPE_CHECKING:  # 仅类型提示，运行时绝不导入
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    # 延迟导入：必须等到 config 已把 HF_ENDPOINT 写进 os.environ 之后。
    import os

    import torch
    from sentence_transformers import SentenceTransformer

    from bidding_qa_chatbot.embeddings import load_with_cache_fallback

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Windows 上 torch 的默认 intra-op 线程数有时是 1，会让编码慢好几倍。
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    logger.info("加载嵌入模型 %s (device=%s)", model_name, device)
    return load_with_cache_fallback(
        model_name,
        lambda local_only: SentenceTransformer(
            model_name, device=device, local_files_only=local_only
        ),
    )


class DenseEncoder:
    """把文本编码为 512 维归一化向量。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def dimension(self) -> int:
        return self.settings.qdrant_vector_size

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """编码入库文本（passage 侧，不加指令前缀）。"""
        return self._encode(texts, is_query=False)

    def encode_query(self, query: str) -> list[float]:
        """编码查询（query 侧，加指令前缀）。"""
        prefixed = f"{self.settings.embed_query_instruction}{query}"
        return self._encode([prefixed], is_query=True)[0]

    def _encode(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        if not texts:
            return []
        model = _load_model(self.settings.embed_model_name)
        vectors = model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,  # 与 Qdrant 的 Cosine 距离配套
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.shape[1] != self.dimension:
            raise ValueError(
                f"{self.settings.embed_model_name} 输出 {arr.shape[1]} 维，"
                f"但 QDRANT_VECTOR_SIZE={self.dimension}，两者必须一致"
            )
        return arr.tolist()

    def warmup(self) -> None:
        """启动时预加载，避免第一个请求承担模型加载的 2-5 秒。"""
        _load_model(self.settings.embed_model_name)


def clear_model_cache() -> Any:
    """测试用：释放已加载的模型。"""
    _load_model.cache_clear()
