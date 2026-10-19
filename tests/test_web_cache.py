"""进程内 TTL 缓存：命中、过期、淘汰、清空。"""

import pytest

import bidding_qa_chatbot.web.cache as cache_mod
from bidding_qa_chatbot.web.cache import TTLCache


async def test_set_and_get():
    cache = TTLCache(ttl=60, maxsize=4)
    await cache.set("a", [1, 2])
    assert await cache.get("a") == [1, 2]
    assert await cache.get("missing") is None
    assert len(cache) == 1


async def test_ttl_expiry(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(cache_mod.time, "monotonic", lambda: now["t"])
    cache = TTLCache(ttl=10, maxsize=4)
    await cache.set("a", "v")

    now["t"] = 1009.0
    assert await cache.get("a") == "v"

    now["t"] = 1010.0
    assert await cache.get("a") is None
    assert len(cache) == 0


async def test_expired_entries_are_purged_on_set(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(cache_mod.time, "monotonic", lambda: now["t"])
    cache = TTLCache(ttl=5, maxsize=4)
    await cache.set("a", 1)
    now["t"] = 1006.0
    await cache.set("b", 2)
    assert await cache.get("a") is None
    assert await cache.get("b") == 2


async def test_maxsize_evicts_oldest():
    cache = TTLCache(ttl=60, maxsize=2)
    await cache.set("a", 1)
    await cache.set("b", 2)
    await cache.set("c", 3)

    assert await cache.get("a") is None
    assert await cache.get("b") == 2
    assert await cache.get("c") == 3


async def test_clear():
    cache = TTLCache(ttl=60, maxsize=4)
    await cache.set("a", 1)
    await cache.clear()
    assert await cache.get("a") is None
    assert len(cache) == 0


def test_maxsize_is_at_least_one():
    assert TTLCache(ttl=1, maxsize=0).maxsize == 1


@pytest.mark.parametrize("key", ["a", "A", "a "])
async def test_keys_are_not_normalized_by_cache(key):
    """归一化是调用方的职责（service 会拼归一化后的键）。"""
    cache = TTLCache(ttl=60, maxsize=4)
    await cache.set("a", 1)
    assert await cache.get(key) == (1 if key == "a" else None)
