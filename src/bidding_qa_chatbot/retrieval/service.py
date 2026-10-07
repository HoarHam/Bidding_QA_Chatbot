"""检索门面：混合召回 → 精排 → 组装 RAG 上下文。

依赖注入是刻意的：`DenseEncoder` / `Reranker` / BM25 工件 / Qdrant 客户端都可替换，
测试因此能在 `QdrantClient(":memory:")` + 假编码器上跑完整链路，不碰任何外部服务。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from qdrant_client import models as qm

from bidding_qa_chatbot.config import Settings, get_settings
from bidding_qa_chatbot.retrieval import bm25 as bm25_mod
from bidding_qa_chatbot.retrieval.qdrant_store import RetrievedPoint, hybrid_search
from bidding_qa_chatbot.schemas import SourceImage, SourceRef

logger = logging.getLogger(__name__)

#: 引用卡片里展示的答案摘要长度
SNIPPET_CHARS = 160


class SupportsEncodeQuery(Protocol):
    def encode_query(self, query: str) -> list[float]: ...


class SupportsRerank(Protocol):
    def rerank(self, query: str, candidates: list[str], top_k: int | None = None) -> list[Any]: ...


@dataclass(slots=True)
class RetrievedSource:
    """检索结果，携带完整正文供 prompt 组装。"""

    chunk_id: str
    source_label: str
    question: str
    answer: str
    text: str
    excel_row: int
    rrf_score: float
    rerank_score: float | None
    images: list[dict[str, Any]] = field(default_factory=list)

    def to_ref(self) -> SourceRef:
        snippet = (
            self.answer if len(self.answer) <= SNIPPET_CHARS else self.answer[:SNIPPET_CHARS] + "…"
        )
        return SourceRef(
            chunk_id=self.chunk_id,
            source_label=self.source_label,
            question=self.question,
            snippet=snippet,
            excel_row=self.excel_row,
            rrf_score=self.rrf_score,
            rerank_score=self.rerank_score,
            images=[
                SourceImage(
                    index=int(img.get("index", 0)),
                    url=str(img.get("url", "")),
                    description=img.get("description"),
                )
                for img in self.images
            ],
        )


@dataclass(slots=True)
class RetrievalResult:
    sources: list[RetrievedSource]
    elapsed_ms: int
    degraded: bool = False
    reason: str | None = None


class Retriever:
    def __init__(
        self,
        qdrant_client: Any,
        encoder: SupportsEncodeQuery,
        reranker: SupportsRerank | None,
        settings: Settings | None = None,
        bm25_artifact: bm25_mod.BM25Artifact | None = None,
        collection_vocab_sha1: str | None = None,
    ) -> None:
        self.client = qdrant_client
        self.encoder = encoder
        self.reranker = reranker
        self.settings = settings or get_settings()
        self.bm25 = bm25_artifact
        self.collection_vocab_sha1 = collection_vocab_sha1

    # ------------------------------------------------------------ 内部
    def _sparse_available(self) -> tuple[bool, str | None]:
        """判断稀疏一路能否安全使用。"""
        if self.bm25 is None:
            return False, "BM25 词表不存在，请先运行 ingest"
        if self.collection_vocab_sha1 and self.collection_vocab_sha1 != self.bm25.vocab_sha1:
            # 词表整体平移后，所有已入库的稀疏向量都失去意义，
            # 但 dense 一路照常工作——所以必须显式告警，不能静默出结果。
            return False, "BM25 词表与集合不一致，请重新运行 ingest"
        return True, None

    def _encode_sparse(self, query: str) -> qm.SparseVector | None:
        ok, reason = self._sparse_available()
        if not ok:
            logger.warning("稀疏检索已禁用：%s", reason)
            return None
        assert self.bm25 is not None
        return bm25_mod.encode_query(query, self.bm25)

    # ------------------------------------------------------------ 对外
    async def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        started = time.perf_counter()

        dense_vector = self.encoder.encode_query(query)
        sparse_vector = self._encode_sparse(query)

        points: list[RetrievedPoint] = await hybrid_search(
            self.client, dense_vector, sparse_vector, self.settings, top_k=top_k
        )

        degraded = sparse_vector is None
        reason = None if not degraded else self._sparse_available()[1]

        if not points:
            return RetrievalResult(
                sources=[], elapsed_ms=_ms(started), degraded=degraded, reason=reason
            )

        sources = self._rerank(query, points, degraded=degraded, reason=reason)
        return RetrievalResult(
            sources=sources,
            elapsed_ms=_ms(started),
            degraded=degraded,
            reason=reason,
        )

    def _rerank(
        self,
        query: str,
        points: list[RetrievedPoint],
        *,
        degraded: bool,
        reason: str | None,
    ) -> list[RetrievedSource]:
        candidates = [_source_from_point(p, rerank_score=None) for p in points]

        if self.reranker is None or not self.settings.rerank_enabled:
            # 不精排时按 RRF 顺序截断即可。
            return candidates[: self.settings.rerank_top_k]

        # 精排输入必须与入库文本一致，否则会出现无法解释的排序翻转。
        texts = [c.text for c in candidates]
        ranked = self.reranker.rerank(query, texts, top_k=self.settings.rerank_top_k)

        out: list[RetrievedSource] = []
        for item in ranked:
            source = candidates[item.index]
            source.rerank_score = item.score
            out.append(source)
        return out


def _source_from_point(point: RetrievedPoint, rerank_score: float | None) -> RetrievedSource:
    payload = point.payload
    return RetrievedSource(
        chunk_id=point.chunk_id,
        source_label=str(payload.get("source_label", point.chunk_id)),
        question=str(payload.get("question", "")),
        answer=str(payload.get("answer", "")),
        text=str(payload.get("text", "")),
        excel_row=int(payload.get("excel_row", 0)),
        rrf_score=point.score,
        rerank_score=rerank_score,
        images=list(payload.get("images") or []),
    )


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
