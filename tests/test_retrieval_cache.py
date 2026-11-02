"""RAG 候选缓存：命中、键隔离、LRU、TTL 与关闭开关。

不碰真实 Qdrant：把 `service.hybrid_search` 换成计数假实现，
断言「同一查询向量只调用一次」以及不同 key 不会误命中。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.retrieval import service as service_mod
from bidding_qa_chatbot.retrieval.cache import RetrievalCache, build_cache_key
from bidding_qa_chatbot.retrieval.qdrant_store import RetrievedPoint
from bidding_qa_chatbot.retrieval.service import Retriever
from tests.conftest import FakeEncoder


def _point(chunk_id: str = "qa-0001") -> RetrievedPoint:
    return RetrievedPoint(
        chunk_id=chunk_id,
        score=0.42,
        payload={
            "chunk_id": chunk_id,
            "source_label": "测试数据 · 第2行",
            "question": "投标保证金什么时候退？",
            "answer": "成交结果公告后5个工作日内退还。",
            "text": "问\n答",
            "excel_row": 2,
            "images": [],
        },
    )


def _settings(settings_factory, **overrides) -> Settings:
    return settings_factory(qdrant_collection="cache_test", **overrides)


@pytest.fixture
def counted_search(monkeypatch):
    calls: list[dict] = []

    async def fake_hybrid(client, dense_vector, sparse_vector, settings, *, top_k=None):
        calls.append({"dense": tuple(dense_vector), "sparse": sparse_vector, "top_k": top_k})
        return [_point()]

    monkeypatch.setattr(service_mod, "hybrid_search", fake_hybrid)
    return calls


async def test_shared_cache_skips_qdrant_for_repeated_embedding(settings_factory, counted_search):
    settings = _settings(settings_factory)
    encoder = FakeEncoder()
    cache = RetrievalCache(maxsize=8)

    # 两个 Retriever 实例模拟「Agent 工具」与「固定检索路径」共享 app.state 缓存
    first = Retriever(object(), encoder, None, settings, cache=cache)
    second = Retriever(object(), encoder, None, settings, cache=cache)

    one = await first.retrieve("投标保证金退还期限")
    two = await second.retrieve("投标保证金退还期限")

    assert len(counted_search) == 1, "相同查询向量不应重复调用 Qdrant"
    assert one.cache_hit is False
    assert two.cache_hit is True
    assert [s.chunk_id for s in one.sources] == [s.chunk_id for s in two.sources]
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1


async def test_route_retrieve_uses_the_shared_app_state_cache(settings_factory, counted_search):
    """固定检索路径每次请求都会新建 Retriever，缓存必须来自 app.state。"""
    from bidding_qa_chatbot.api.routes_chat import _retrieve

    settings = _settings(settings_factory)
    state = SimpleNamespace(
        settings=settings,
        encoder=FakeEncoder(),
        qdrant=object(),
        reranker=None,
        bm25=None,
        collection_vocab_sha1=None,
        retrieval_cache=RetrievalCache(maxsize=8),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    body = SimpleNamespace(query="投标保证金退还期限", top_k=None)

    first = await _retrieve(request, body)
    second = await _retrieve(request, body)

    assert len(counted_search) == 1
    assert first.cache_hit is False and second.cache_hit is True


async def test_different_query_or_top_k_is_a_different_cache_key(settings_factory, counted_search):
    settings = _settings(settings_factory)
    retriever = Retriever(object(), FakeEncoder(), None, settings, cache=RetrievalCache(8))

    await retriever.retrieve("投标保证金", top_k=3)
    await retriever.retrieve("投标保证金", top_k=3)
    await retriever.retrieve("投标保证金", top_k=5)
    await retriever.retrieve("竞争性磋商", top_k=3)

    assert len(counted_search) == 3
    assert [c["top_k"] for c in counted_search] == [3, 5, 3]


async def test_empty_result_is_cached_too(settings_factory, monkeypatch):
    settings = _settings(settings_factory)
    calls = []

    async def empty_hybrid(*args, **kwargs):
        calls.append(1)
        return []

    monkeypatch.setattr(service_mod, "hybrid_search", empty_hybrid)
    retriever = Retriever(object(), FakeEncoder(), None, settings, cache=RetrievalCache(8))

    first = await retriever.retrieve("没有结果的问题")
    second = await retriever.retrieve("没有结果的问题")

    assert len(calls) == 1
    assert first.cache_hit is False and second.cache_hit is True
    assert second.sources == []


async def test_lru_eviction(settings_factory, counted_search):
    settings = _settings(settings_factory)
    cache = RetrievalCache(maxsize=1, ttl_seconds=0)
    retriever = Retriever(object(), FakeEncoder(), None, settings, cache=cache)

    await retriever.retrieve("第一个问题")
    await retriever.retrieve("第二个问题")
    await retriever.retrieve("第一个问题")

    assert len(counted_search) == 3, "maxsize=1 时第一个条目应已被淘汰"
    assert cache.stats()["evictions"] == 2


async def test_ttl_expiry(settings_factory, counted_search):
    settings = _settings(settings_factory)
    now = [100.0]
    cache = RetrievalCache(maxsize=8, ttl_seconds=10, clock=lambda: now[0])
    retriever = Retriever(object(), FakeEncoder(), None, settings, cache=cache)

    await retriever.retrieve("投标保证金")
    now[0] += 9.9
    assert (await retriever.retrieve("投标保证金")).cache_hit is True
    assert len(counted_search) == 1

    now[0] += 0.2  # 超过 10 秒
    third = await retriever.retrieve("投标保证金")
    assert third.cache_hit is False
    assert len(counted_search) == 2
    assert cache.stats()["expirations"] == 1


async def test_cache_can_be_disabled(settings_factory, counted_search):
    settings = _settings(settings_factory, retrieval_cache_enabled=False)
    retriever = Retriever(object(), FakeEncoder(), None, settings)

    first = await retriever.retrieve("投标保证金")
    second = await retriever.retrieve("投标保证金")

    assert retriever.cache.enabled is False
    assert len(counted_search) == 2
    assert first.cache_hit is False and second.cache_hit is False


def test_cache_key_includes_sparse_vector_and_search_params():
    common = {
        "dense_vector": [0.1, 0.2, 0.3],
        "collection": "bid_qa",
        "limit": 10,
        "prefetch_dense": 20,
        "prefetch_sparse": 20,
        "embed_model": "bge",
        "query_instruction": "指令",
        "collection_vocab_sha1": "abc",
        "distance": "cosine",
    }
    dense_only = build_cache_key(sparse_vector=None, **common)
    sparse_a = build_cache_key(
        sparse_vector=SimpleNamespace(indices=[1, 2], values=[0.5, 0.25]), **common
    )
    sparse_b = build_cache_key(
        sparse_vector=SimpleNamespace(indices=[1, 3], values=[0.5, 0.25]), **common
    )

    assert len({dense_only, sparse_a, sparse_b}) == 3
    assert build_cache_key(sparse_vector=None, **{**common, "limit": 5}) != dense_only
    assert build_cache_key(sparse_vector=None, **{**common, "collection": "other"}) != dense_only


def test_cache_stats_and_clear():
    cache = RetrievalCache(maxsize=2, ttl_seconds=0)
    assert cache.get("missing") is None
    cache.put("a", [_point()])
    assert cache.get("a") is not None
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1
    assert len(cache) == 1

    cache.clear()
    assert len(cache) == 0
    assert cache.stats()["hits"] == 0
