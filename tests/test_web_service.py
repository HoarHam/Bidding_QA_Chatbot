"""WebSearchService：引擎可用性、Tavily 进程内缓存、条数钳制、错误传播。"""

import pytest

from bidding_qa_chatbot.schemas import WebSource
from bidding_qa_chatbot.web.errors import WebSearchError
from bidding_qa_chatbot.web.service import WebSearchService


def _source(url: str = "https://example.com/a") -> WebSource:
    return WebSource(url=url, title="T", snippet="S", score=0.9, engine="tavily")


class FakeTavily:
    def __init__(self, sources=None, error: Exception | None = None) -> None:
        self.sources = [_source()] if sources is None else sources
        self.error = error
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, max_results: int) -> list[WebSource]:
        self.calls.append((query, max_results))
        if self.error is not None:
            raise self.error
        return list(self.sources)


class FakeExa:
    def __init__(self, sources=None, error: Exception | None = None) -> None:
        self.sources = [_source("https://example.com/exa")] if sources is None else sources
        self.error = error
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, max_results: int) -> list[WebSource]:
        self.calls.append((query, max_results))
        if self.error is not None:
            raise self.error
        return list(self.sources)


def test_availability_flags(settings_factory):
    none = WebSearchService(settings_factory(), tavily=FakeTavily(), exa=FakeExa())
    assert none.keyword_available is False
    assert none.semantic_available is False
    assert none.available is False

    # 语义搜索主路径是 Exa HTTP API：有 key 即可用（npx 只影响 MCP 回退）
    semantic_only = WebSearchService(
        settings_factory(exa_api_key="e"), tavily=FakeTavily(), exa=FakeExa()
    )
    assert semantic_only.keyword_available is False
    assert semantic_only.semantic_available is True
    assert semantic_only.available is True

    both = WebSearchService(
        settings_factory(tavily_api_key="t", exa_api_key="e"),
        tavily=FakeTavily(),
        exa=FakeExa(),
    )
    assert both.keyword_available is True
    assert both.semantic_available is True
    assert both.available is True


async def test_keyword_search_caches_normalized_query(settings_factory):
    fake = FakeTavily()
    service = WebSearchService(settings_factory(tavily_api_key="t"), tavily=fake)

    first = await service.keyword_search("  政府采购  ")
    second = await service.keyword_search("政府采购")

    assert first.cached is False
    assert second.cached is True
    assert fake.calls == [("  政府采购  ", 10)]  # 只有第一次真正调用；查询原样传给引擎
    assert second.sources == first.sources


async def test_different_max_results_bypasses_cache(settings_factory):
    fake = FakeTavily()
    service = WebSearchService(settings_factory(tavily_api_key="t"), tavily=fake)

    await service.keyword_search("q", 3)
    await service.keyword_search("q", 5)

    assert fake.calls == [("q", 3), ("q", 5)]


async def test_max_results_is_clamped(settings_factory):
    fake = FakeTavily()
    service = WebSearchService(
        settings_factory(tavily_api_key="t", web_search_max_results=8), tavily=fake
    )

    await service.keyword_search("big", 999)
    await service.keyword_search("small", 0)

    assert fake.calls == [("big", 8), ("small", 1)]


async def test_keyword_unavailable_raises(settings_factory):
    service = WebSearchService(settings_factory(tavily_api_key=None), tavily=FakeTavily())
    with pytest.raises(WebSearchError) as excinfo:
        await service.keyword_search("q")
    assert excinfo.value.code == "web_search_unavailable"


async def test_semantic_search_uses_exa(settings_factory):
    fake = FakeExa()
    service = WebSearchService(settings_factory(exa_api_key="e"), tavily=FakeTavily(), exa=fake)

    result = await service.semantic_search("单一来源采购 条件", 4)

    assert result.engine == "exa"
    assert fake.calls == [("单一来源采购 条件", 4)]
    assert result.sources[0].url == "https://example.com/exa"
    assert result.cached is False


async def test_semantic_unavailable_without_key(settings_factory):
    service = WebSearchService(
        settings_factory(exa_api_key=None), tavily=FakeTavily(), exa=FakeExa()
    )

    with pytest.raises(WebSearchError) as excinfo:
        await service.semantic_search("q")
    assert excinfo.value.code == "web_search_unavailable"


async def test_engine_errors_propagate_untouched(settings_factory):
    error = WebSearchError("web_search_timeout", "Tavily 联网搜索超时")
    service = WebSearchService(settings_factory(tavily_api_key="t"), tavily=FakeTavily(error=error))

    with pytest.raises(WebSearchError) as excinfo:
        await service.keyword_search("q")
    assert excinfo.value.code == "web_search_timeout"
