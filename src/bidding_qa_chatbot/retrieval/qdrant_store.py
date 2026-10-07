"""Qdrant Named Vectors 混合检索（需求 8）。

集合结构：
    dense  : 512 维 float，Cosine          —— bge-small-zh-v1.5 语义召回
    sparse : SparseVector(indices, values) —— jieba+BM25 词汇精确匹配

查询用 `Prefetch` 两路各取候选，再由 `FusionQuery(Fusion.RRF)` 做排名融合。

## 为什么不静默删集合

`QDRANT_VECTOR_SIZE` 是环境变量驱动的。如果某次改动把它从 512 改成 768，
而建集合逻辑选择「不匹配就删了重建」，就会在没有任何提示的情况下
清空整个集合。这里改成：schema 不匹配时**抛错**并提示加 `--recreate`，
把破坏性操作交还给操作者显式决定。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from bidding_qa_chatbot.config import Settings, get_settings

logger = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

_DISTANCE_MAP = {
    "cosine": qm.Distance.COSINE,
    "dot": qm.Distance.DOT,
    "euclid": qm.Distance.EUCLID,
    "manhattan": qm.Distance.MANHATTAN,
}


class CollectionSchemaMismatch(RuntimeError):
    """既有集合结构与当前配置不一致。绝不自动删除，交由人工决定。"""


@dataclass(slots=True)
class RetrievedPoint:
    chunk_id: str
    score: float
    payload: dict[str, Any]


def build_client(settings: Settings | None = None, **kwargs: Any) -> AsyncQdrantClient:
    """构造 Qdrant 客户端。

    支持注入 `location=":memory:"`（测试用），此时不读远程配置。
    """
    settings = settings or get_settings()
    if "location" in kwargs or "path" in kwargs:
        return AsyncQdrantClient(**kwargs)
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
        timeout=settings.qdrant_timeout,
        # trust_env 默认 True 会让 httpx 走 Windows 系统代理。
        # 本机代理连不通 Qdrant Cloud（ConnectError），直连正常，故默认关闭。
        trust_env=settings.http_trust_env,
        **kwargs,
    )


def _dense_params(settings: Settings) -> qm.VectorParams:
    distance = _DISTANCE_MAP.get(settings.qdrant_distance.lower())
    if distance is None:
        raise ValueError(
            f"QDRANT_DISTANCE 取值非法: {settings.qdrant_distance}；可选 {sorted(_DISTANCE_MAP)}"
        )
    return qm.VectorParams(size=settings.qdrant_vector_size, distance=distance)


async def ensure_collection(
    client: AsyncQdrantClient,
    settings: Settings | None = None,
    *,
    recreate: bool = False,
) -> None:
    """确保集合存在且结构符合预期。"""
    settings = settings or get_settings()
    name = settings.qdrant_collection

    exists = await client.collection_exists(name)
    if exists and recreate:
        logger.warning("按 --recreate 删除集合 %s", name)
        await client.delete_collection(name)
        exists = False

    if not exists:
        await client.create_collection(
            collection_name=name,
            vectors_config={DENSE_VECTOR: _dense_params(settings)},
            sparse_vectors_config={SPARSE_VECTOR: qm.SparseVectorParams()},
        )
        # payload 索引让按 chunk_id / dataset 过滤不必全表扫。
        for field_name in ("chunk_id", "dataset"):
            await client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        logger.info("已创建集合 %s", name)
        return

    info = await client.get_collection(name)
    problems: list[str] = []

    vectors = info.config.params.vectors
    if not isinstance(vectors, dict) or DENSE_VECTOR not in vectors:
        problems.append(f"缺少命名向量 {DENSE_VECTOR!r}")
    else:
        actual_size = vectors[DENSE_VECTOR].size
        if actual_size != settings.qdrant_vector_size:
            problems.append(
                f"{DENSE_VECTOR} 维度为 {actual_size}，配置要求 {settings.qdrant_vector_size}"
            )

    sparse = info.config.params.sparse_vectors or {}
    if SPARSE_VECTOR not in sparse:
        problems.append(f"缺少稀疏向量 {SPARSE_VECTOR!r}")

    if problems:
        raise CollectionSchemaMismatch(
            f"集合 {name} 结构与当前配置不一致：{'; '.join(problems)}。"
            f"确认无误后请加 --recreate 显式重建（该操作会清空集合）。"
        )


async def upsert_chunks(
    client: AsyncQdrantClient,
    points: list[qm.PointStruct],
    settings: Settings | None = None,
) -> None:
    settings = settings or get_settings()
    if not points:
        return
    await client.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)


async def delete_stale(
    client: AsyncQdrantClient,
    keep_chunk_ids: set[str],
    settings: Settings | None = None,
) -> int:
    """删除已不在语料中的点（例如源表删了行）。返回删除数量。"""
    settings = settings or get_settings()
    name = settings.qdrant_collection

    existing: list[str] = []
    offset = None
    while True:
        records, offset = await client.scroll(
            collection_name=name,
            limit=256,
            offset=offset,
            with_payload=["chunk_id"],
            with_vectors=False,
        )
        existing.extend(str(r.payload.get("chunk_id")) for r in records if r.payload)
        if offset is None:
            break

    stale = [cid for cid in existing if cid not in keep_chunk_ids]
    if not stale:
        return 0

    from bidding_qa_chatbot.ingestion.ids import make_point_id

    await client.delete(
        collection_name=name,
        points_selector=qm.PointIdsList(points=[make_point_id(cid) for cid in stale]),
        wait=True,
    )
    logger.info("清理 %d 个陈旧分块", len(stale))
    return len(stale)


async def hybrid_search(
    client: AsyncQdrantClient,
    dense_vector: list[float],
    sparse_vector: qm.SparseVector | None,
    settings: Settings | None = None,
    *,
    top_k: int | None = None,
) -> list[RetrievedPoint]:
    """两路预取 + RRF 融合。

    `sparse_vector` 为空（词表缺失或查询词全部未命中）时自动退化为纯 dense 检索，
    而不是报错——用户仍然能拿到语义检索的结果。
    """
    settings = settings or get_settings()
    limit = top_k or settings.retrieval_top_k

    prefetch: list[qm.Prefetch] = [
        qm.Prefetch(
            query=dense_vector,
            using=DENSE_VECTOR,
            limit=settings.retrieval_prefetch_dense,
        )
    ]
    if sparse_vector is not None and len(sparse_vector.indices) > 0:
        prefetch.append(
            qm.Prefetch(
                query=sparse_vector,
                using=SPARSE_VECTOR,
                limit=settings.retrieval_prefetch_sparse,
            )
        )

    response = await client.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=prefetch,
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    return [
        RetrievedPoint(
            chunk_id=str((p.payload or {}).get("chunk_id", p.id)),
            score=float(p.score),
            payload=dict(p.payload or {}),
        )
        for p in response.points
    ]


async def collection_meta(
    client: AsyncQdrantClient, settings: Settings | None = None
) -> dict[str, Any]:
    settings = settings or get_settings()
    info = await client.get_collection(settings.qdrant_collection)
    return dict(info.config.metadata or {})


async def set_collection_meta(
    client: AsyncQdrantClient, meta: dict[str, Any], settings: Settings | None = None
) -> None:
    settings = settings or get_settings()
    await client.update_collection(collection_name=settings.qdrant_collection, metadata=meta)


async def count_points(client: AsyncQdrantClient, settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    result = await client.count(collection_name=settings.qdrant_collection, exact=True)
    return int(result.count)
