"""RAG 检索候选缓存：query embedding → Qdrant 返回的候选文档。

在 `Retriever.retrieve` 里，缓存的是 **Qdrant 混合查询的候选点**（文档 id +
payload + RRF 分数），不是最终精排结果。原因：

- 同一查询向量重复出现时可以直接跳过 `query_points`，省掉一次 Qdrant API 调用；
- 精排分数依赖当前问题的原文，必须每次重算，缓存它会把旧问题的排序带过来；
- Qdrant 的查询响应本来就同时返回 payload，缓存它可以避免为了取正文再发一次
  retrieve 请求——只缓存 id 仍然要额外一次 Qdrant 调用，省不下来。

缓存键包含稠密向量、稀疏向量、集合名、top_k、预取数量、模型与词表指纹。
只要任一项不同就不会误命中。TTL 用来兜底「服务运行期间重新入库」的情况。
"""

from __future__ import annotations

from array import array
from collections import OrderedDict
from collections.abc import Callable
from hashlib import blake2b
from threading import Lock
from time import monotonic
from typing import Any

from bidding_qa_chatbot.retrieval.qdrant_store import RetrievedPoint

CACHE_KEY_VERSION = "rag-cache-v1"

#: 缓存值保留 Qdrant 返回的完整候选点（含 payload），供精排/上下文组装直接使用。
CacheValue = tuple[RetrievedPoint, ...]


def build_cache_key(
    *,
    dense_vector: list[float],
    sparse_vector: Any | None,
    collection: str,
    limit: int,
    prefetch_dense: int,
    prefetch_sparse: int,
    embed_model: str,
    query_instruction: str,
    collection_vocab_sha1: str | None,
    distance: str,
) -> str:
    """生成稳定键。

    稠密向量按 float32 打包——编码器输出的就是 float32，用 float64 打包只会
    放大平台差异却不增加区分度。稀疏向量同时纳入 indices 与 values：
    同一个稠密向量配不同查询词时，BM25 一路的候选不同，不能共用缓存。
    """
    digest = blake2b(digest_size=20)
    parts = (
        CACHE_KEY_VERSION,
        collection,
        str(limit),
        str(prefetch_dense),
        str(prefetch_sparse),
        embed_model,
        query_instruction,
        collection_vocab_sha1 or "",
        distance,
    )
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x00")

    digest.update(array("f", dense_vector).tobytes())
    digest.update(b"\x00")
    indices = getattr(sparse_vector, "indices", None) if sparse_vector is not None else None
    values = getattr(sparse_vector, "values", None) if sparse_vector is not None else None
    if indices and values:
        digest.update(b"s")
        digest.update(array("I", (int(i) for i in indices)).tobytes())
        digest.update(array("f", (float(v) for v in values)).tobytes())
    else:
        digest.update(b"d")
    return digest.hexdigest()


class RetrievalCache:
    """线程安全的小型 LRU（任务在 asyncio 单线程里跑，但同步锁成本可忽略）。

    `maxsize=0` 表示关闭；此时所有 get 都 miss、put 都丢弃，调用方无需分支。
    """

    def __init__(
        self,
        maxsize: int = 128,
        ttl_seconds: float = 600.0,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.maxsize = max(0, int(maxsize))
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self._clock = clock or monotonic
        self._entries: OrderedDict[str, tuple[float, CacheValue]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    @property
    def enabled(self) -> bool:
        return self.maxsize > 0

    def get(self, key: str) -> list[RetrievedPoint] | None:
        if not self.enabled:
            return None
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            stored_at, points = entry
            if self.ttl_seconds > 0 and now - stored_at >= self.ttl_seconds:
                del self._entries[key]
                self._expirations += 1
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            # 浅拷贝列表：调用方可能替换列表内容，但不允许改到缓存中的列表本身。
            return list(points)

    def put(self, key: str, points: list[RetrievedPoint]) -> None:
        if not self.enabled:
            return
        now = self._clock()
        with self._lock:
            self._entries[key] = (now, tuple(points))
            self._entries.move_to_end(key)
            while len(self._entries) > self.maxsize:
                self._entries.popitem(last=False)
                self._evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "maxsize": self.maxsize,
                "ttl_seconds": self.ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "expirations": self._expirations,
            }
