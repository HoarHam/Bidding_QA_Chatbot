"""混合检索端到端测试——全程离线。

用 `QdrantClient(location=":memory:")` + 确定性假编码器 + 真实 BM25 编码，
完整跑通「dense + sparse → Prefetch → RRF 融合 → 精排」这条链路，
不需要 Qdrant Cloud，也不需要任何 API key。

**断言只校验顺序与成员，不校验分数数值。** 实测 qdrant-client 本地模式返回的
RRF 分数是归一化过的，与服务端原始 `1/(k+rank)` 不同；若断言具体分数，
测试会在切到真实服务端时莫名其妙地挂掉。
"""

from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from bidding_qa_chatbot.ingestion.ids import make_point_id
from bidding_qa_chatbot.retrieval import bm25 as bm25_mod
from bidding_qa_chatbot.retrieval.qdrant_store import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    ensure_collection,
    hybrid_search,
)
from bidding_qa_chatbot.retrieval.service import Retriever

DOCS = [
    (
        "qa-0000",
        "供应商针对单一来源异议被驳回，可以再次异议吗？",
        "对同一采购程序的异议只能提出一次，驳回后不能再次异议。",
    ),
    (
        "qa-0001",
        "投标保证金的退还期限是多久？",
        "采购人应当在成交结果公告后5个工作日内退还投标保证金。",
    ),
    (
        "qa-0002",
        "竞争性磋商文件的澄清期限如何规定？",
        "磋商文件的澄清或修改应当在提交首次响应文件截止时间5日前发出。",
    ),
]


@pytest.fixture
async def memory_store():
    client = AsyncQdrantClient(location=":memory:")
    yield client
    await client.close()


@pytest.fixture
def bm25_artifact():
    return bm25_mod.build_artifact([f"{q}\n{a}" for _, q, a in DOCS], "test-corpus")


async def _seed(client, artifact, encoder):
    from bidding_qa_chatbot.config import Settings

    settings = Settings(
        _env_file=None, qdrant_url="http://x", qdrant_api_key="k", qdrant_collection="t"
    )
    await ensure_collection(client, settings)

    vectors = encoder.encode_documents([f"{q}\n{a}" for _, q, a in DOCS])
    points = [
        qm.PointStruct(
            id=make_point_id(cid),
            vector={
                DENSE_VECTOR: vectors[i],
                SPARSE_VECTOR: bm25_mod.encode_document(f"{q}\n{a}", artifact),
            },
            payload={
                "chunk_id": cid,
                "question": q,
                "answer": a,
                "text": f"{q}\n{a}",
                "source_label": f"测试数据 · 第{i + 2}行",
                "excel_row": i + 2,
                "images": [],
            },
        )
        for i, (cid, q, a) in enumerate(DOCS)
    ]
    await client.upsert("t", points=points, wait=True)
    return settings


async def test_sparse_and_dense_both_contribute(memory_store):
    """两路预取都要有候选，融合结果非空。"""
    from tests.conftest import FakeEncoder

    encoder = FakeEncoder()
    artifact = bm25_mod.build_artifact([f"{q}\n{a}" for _, q, a in DOCS], "c")
    settings = await _seed(memory_store, artifact, encoder)

    dense = encoder.encode_query("投标保证金退还")
    sparse = bm25_mod.encode_query("投标保证金退还", artifact)
    assert len(sparse.indices) > 0, "查询词应命中词表"

    hits = await hybrid_search(memory_store, dense, sparse, settings, top_k=3)
    assert hits, "融合结果不应为空"
    assert {h.chunk_id for h in hits} <= {cid for cid, _, _ in DOCS}


async def test_keyword_query_hits_correct_document_via_sparse(memory_store):
    """词汇完全匹配时，BM25 一路应把目标文档排到很靠前。"""
    from tests.conftest import FakeEncoder

    encoder = FakeEncoder()
    artifact = bm25_mod.build_artifact([f"{q}\n{a}" for _, q, a in DOCS], "c")
    settings = await _seed(memory_store, artifact, encoder)

    sparse = bm25_mod.encode_query("竞争性磋商文件的澄清期限", artifact)
    hits = await hybrid_search(memory_store, encoder.encode_query("x"), sparse, settings, top_k=3)
    assert hits[0].chunk_id == "qa-0002"


async def test_sparse_absent_degrades_to_dense_only(memory_store):
    """稀疏向量为 None 时必须退化为纯 dense，而不是报错。"""
    from tests.conftest import FakeEncoder

    encoder = FakeEncoder()
    artifact = bm25_mod.build_artifact([f"{q}\n{a}" for _, q, a in DOCS], "c")
    settings = await _seed(memory_store, artifact, encoder)

    hits = await hybrid_search(
        memory_store, encoder.encode_query("任意查询"), None, settings, top_k=3
    )
    assert len(hits) == 3


async def test_collection_schema_mismatch_raises_rather_than_wipes(memory_store):
    """维度不匹配必须抛错，绝不能静默删集合。"""
    from bidding_qa_chatbot.config import Settings
    from bidding_qa_chatbot.retrieval.qdrant_store import CollectionSchemaMismatch

    await ensure_collection(
        memory_store,
        Settings(
            _env_file=None,
            qdrant_url="http://x",
            qdrant_api_key="k",
            qdrant_collection="m",
            qdrant_vector_size=4,
        ),
    )
    with pytest.raises(CollectionSchemaMismatch, match="不一致"):
        await ensure_collection(
            memory_store,
            Settings(
                _env_file=None,
                qdrant_url="http://x",
                qdrant_api_key="k",
                qdrant_collection="m",
                qdrant_vector_size=512,
            ),
        )


async def test_retriever_end_to_end_with_rerank(memory_store):
    """完整门面：召回 → 精排 → 按精排分降序，且精排输入与入库文本一致。"""
    from tests.conftest import FakeEncoder, FakeReranker

    encoder = FakeEncoder()
    reranker = FakeReranker()
    artifact = bm25_mod.build_artifact([f"{q}\n{a}" for _, q, a in DOCS], "c")
    settings = await _seed(memory_store, artifact, encoder)

    retriever = Retriever(memory_store, encoder, reranker, settings, bm25_artifact=artifact)
    result = await retriever.retrieve("投标保证金退还", top_k=3)

    assert result.sources
    assert not result.degraded
    scores = [s.rerank_score for s in result.sources]
    assert scores == sorted(scores, reverse=True)
    # 精排收到的文本必须与入库文本一致，否则排序会莫名其妙地翻转
    assert all(text in [f"{q}\n{a}" for _, q, a in DOCS] for _, text in reranker.seen)


async def test_retriever_degrades_when_vocab_stale(memory_store):
    """词表与集合不一致时必须降级并给出原因，而不是静默返回可疑结果。"""
    from tests.conftest import FakeEncoder

    encoder = FakeEncoder()
    artifact = bm25_mod.build_artifact([f"{q}\n{a}" for _, q, a in DOCS], "c")
    settings = await _seed(memory_store, artifact, encoder)

    retriever = Retriever(
        memory_store,
        encoder,
        None,
        settings,
        bm25_artifact=artifact,
        collection_vocab_sha1="完全不匹配的指纹",
    )
    result = await retriever.retrieve("投标保证金", top_k=3)
    assert result.degraded
    assert result.reason and "不一致" in result.reason
    assert result.sources, "降级后仍应通过 dense 返回结果"


async def test_retriever_without_artifact_is_degraded(memory_store):
    from tests.conftest import FakeEncoder

    encoder = FakeEncoder()
    artifact = bm25_mod.build_artifact([f"{q}\n{a}" for _, q, a in DOCS], "c")
    settings = await _seed(memory_store, artifact, encoder)

    retriever = Retriever(memory_store, encoder, None, settings, bm25_artifact=None)
    result = await retriever.retrieve("任意", top_k=3)
    assert result.degraded
    assert "词表不存在" in (result.reason or "")
