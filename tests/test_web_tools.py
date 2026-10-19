"""联网搜索工具：spec、严格签名、可用性、空结果/成功 artifacts、错误包含。

`artifacts` 的形状是硬约束：联网结果必须走 `web_sources`，不能混进 SourceRef 通道。
"""

import inspect

import pytest

from bidding_qa_chatbot.agent.base import ToolRegistry
from bidding_qa_chatbot.agent.web_tools import (
    MAX_RESULTS,
    WebSearchKeywordTool,
    WebSearchSemanticTool,
)
from bidding_qa_chatbot.schemas import WebSource
from bidding_qa_chatbot.web import WebSearchError, WebSearchResult


def _source(url: str = "https://example.com/a", score: float | None = 0.87) -> WebSource:
    return WebSource(url=url, title="示例标题", snippet="摘要内容", score=score, engine="tavily")


class FakeService:
    def __init__(
        self,
        *,
        keyword_available: bool = True,
        semantic_available: bool = True,
        result: WebSearchResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.keyword_available = keyword_available
        self.semantic_available = semantic_available
        self.result = result or WebSearchResult(
            engine="tavily", query="q", sources=[_source()], elapsed_ms=12
        )
        self.error = error
        self.calls: list[tuple[str, str, int]] = []

    async def keyword_search(self, query: str, max_results: int) -> WebSearchResult:
        self.calls.append(("keyword", query, max_results))
        if self.error is not None:
            raise self.error
        return self.result

    async def semantic_search(self, query: str, max_results: int) -> WebSearchResult:
        self.calls.append(("semantic", query, max_results))
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.parametrize(
    ("cls", "name", "engine_label"),
    [
        (WebSearchKeywordTool, "search_web_keyword", "Tavily"),
        (WebSearchSemanticTool, "search_web_semantic", "Exa"),
    ],
)
def test_specs_are_well_formed(cls, name, engine_label):
    spec = cls.spec
    assert spec.name == name
    assert spec.parameters["required"] == ["query"]
    assert spec.parameters["properties"]["max_results"]["maximum"] == MAX_RESULTS
    assert "联网" in spec.description
    assert engine_label in spec.description
    assert "URL" in spec.description


@pytest.mark.parametrize("cls", [WebSearchKeywordTool, WebSearchSemanticTool])
def test_strict_signature(cls):
    signature = inspect.signature(cls.run)
    kinds = {p.kind for p in signature.parameters.values()}
    assert inspect.Parameter.VAR_KEYWORD not in kinds
    assert set(signature.parameters) == {"self", "query", "max_results"}


async def test_keyword_success_artifacts():
    service = FakeService()
    result = await WebSearchKeywordTool(service).run(query="政府采购 新规", max_results=3)

    assert result.ok
    assert service.calls == [("keyword", "政府采购 新规", 3)]
    assert "https://example.com/a" in result.content
    assert "0.87" in result.content
    assert result.artifacts["summary"] == "联网搜索到 1 条结果"
    assert result.artifacts["web_sources"][0]["url"] == "https://example.com/a"
    assert "sources" not in result.artifacts  # 绝不能走 SourceRef 通道
    assert result.artifacts["web"]["engine"] == "tavily"


async def test_semantic_uses_semantic_engine():
    service = FakeService(result=WebSearchResult("exa", "q", [_source()], 5))
    result = await WebSearchSemanticTool(service).run(query="完整问句？")

    assert service.calls == [("semantic", "完整问句？", 5)]
    assert result.artifacts["web"]["engine"] == "exa"


async def test_unavailable_engine():
    service = FakeService(keyword_available=False)
    result = await WebSearchKeywordTool(service).run(query="q")

    assert result.ok is False
    assert result.error == "web_search_unavailable"
    assert "不要" in result.content
    assert service.calls == []

    assert (await WebSearchKeywordTool(None).run(query="q")).error == "web_search_unavailable"


async def test_empty_query():
    result = await WebSearchKeywordTool(FakeService()).run(query="   ")
    assert result.ok is False
    assert result.error == "empty_query"


async def test_bad_max_results():
    result = await WebSearchKeywordTool(FakeService()).run(query="q", max_results="abc")
    assert result.ok is False
    assert result.error == "bad_arguments"


async def test_max_results_clamped():
    service = FakeService()
    await WebSearchKeywordTool(service).run(query="q", max_results=999)
    await WebSearchKeywordTool(service).run(query="q", max_results=0)
    assert service.calls == [("keyword", "q", MAX_RESULTS), ("keyword", "q", 1)]


async def test_empty_results():
    service = FakeService(result=WebSearchResult("tavily", "q", [], 3))
    result = await WebSearchKeywordTool(service).run(query="q")

    assert result.ok is True
    assert "没有找到" in result.content
    assert result.artifacts["summary"] == "未检索到联网结果"
    assert "web_sources" not in result.artifacts


async def test_web_search_error_is_transcribable():
    service = FakeService(error=WebSearchError("web_search_timeout", "Tavily 联网搜索超时"))
    result = await WebSearchKeywordTool(service).run(query="q")

    assert result.ok is False
    assert result.error == "web_search_timeout"
    assert "超时" in result.content
    assert "不要" in result.content


async def test_unexpected_exception_is_contained():
    service = FakeService(error=RuntimeError("内部爆炸"))
    result = await WebSearchKeywordTool(service).run(query="q")

    assert result.ok is False
    assert result.error == "web_search_failed"
    assert "内部爆炸" in result.content


async def test_registry_reports_bad_arguments():
    registry = ToolRegistry()
    registry.register(WebSearchKeywordTool(FakeService()))
    result = await registry.execute("search_web_keyword", {"q": "x"})
    assert result.ok is False
    assert result.error == "bad_arguments"
