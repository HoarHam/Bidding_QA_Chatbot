"""Exa 语义搜索：HTTP API 主路径（真实 score）+ MCP 回退（无评分）。

不拉起 npx：回退调用通过 `mcp_caller` 注入缝替换；API 请求用 respx 拦截
`https://api.exa.ai/search`。
"""

import asyncio
import json

import httpx
import pytest
import respx

import bidding_qa_chatbot.web.exa_client as exa_mod
from bidding_qa_chatbot.web.errors import WebSearchError
from bidding_qa_chatbot.web.exa_client import (
    ExaSearchClient,
    _parse_advanced,
    _parse_basic,
)

API_URL = "https://api.exa.ai/search"


def _client(settings_factory, *, mcp_caller=None, http=None, **settings_kw) -> ExaSearchClient:
    return ExaSearchClient(
        settings_factory(exa_api_key="exa-key", **settings_kw),
        mcp_caller=mcp_caller,
        http=http,
    )


def _api_payload() -> dict:
    return {
        "requestId": "r1",
        "results": [
            {
                "url": "https://example.com/a",
                "title": "单一来源采购适用情形",
                "score": 0.93,
                "text": "这是 API 返回的正文摘要。",
                "publishedDate": "2026-02-03T00:00:00.000Z",
            },
            {"title": "无 URL 条目", "score": 0.4},
        ],
    }


@respx.mock
async def test_api_path_returns_real_scores(settings_factory):
    route = respx.post(API_URL).mock(return_value=httpx.Response(200, json=_api_payload()))

    async def mcp_caller(query, max_results):  # pragma: no cover — 不应被调用
        raise AssertionError("API 成功时不应走 MCP 回退")

    async with httpx.AsyncClient() as http:
        sources = await _client(settings_factory, mcp_caller=mcp_caller, http=http).search("q", 5)

    assert route.called
    assert len(sources) == 1
    src = sources[0]
    assert src.url == "https://example.com/a"
    assert src.title == "单一来源采购适用情形"
    assert src.snippet == "这是 API 返回的正文摘要。"
    assert src.score == 0.93
    assert src.engine == "exa"
    assert src.published_date.startswith("2026-02-03")

    body = route.calls[0].request.content.decode().replace(" ", "")
    assert '"type":"neural"' in body  # 只有 neural 响应带 score
    assert '"maxcharacters"' in body.lower()  # 请求了正文用于摘要


@respx.mock
async def test_api_failure_falls_back_to_mcp_without_scores(settings_factory, monkeypatch):
    monkeypatch.setattr(exa_mod, "npx_available", lambda: True)
    respx.post(API_URL).mock(return_value=httpx.Response(500))

    async def mcp_caller(query, max_results):
        return (
            "web_search_advanced_exa",
            json.dumps({"results": [{"url": "https://example.com/mcp", "title": "MCP 结果"}]}),
        )

    async with httpx.AsyncClient() as http:
        sources = await _client(settings_factory, mcp_caller=mcp_caller, http=http).search("q", 5)

    assert [s.url for s in sources] == ["https://example.com/mcp"]
    assert sources[0].score is None  # MCP 回退没有评分


@respx.mock
async def test_api_failure_without_npx_raises_api_error(settings_factory, monkeypatch):
    monkeypatch.setattr(exa_mod, "npx_available", lambda: False)

    async def mcp_caller(query, max_results):  # pragma: no cover — 不应被调用
        raise AssertionError("没有 npx 时不应尝试 MCP")

    async with httpx.AsyncClient() as http:
        with pytest.raises(WebSearchError) as excinfo:
            await _client(settings_factory, mcp_caller=mcp_caller, http=http).search("q", 5)

    assert excinfo.value.code == "web_search_failed"


@respx.mock
async def test_api_timeout_maps_to_web_search_timeout(settings_factory, monkeypatch):
    monkeypatch.setattr(exa_mod, "npx_available", lambda: False)
    respx.post(API_URL).mock(side_effect=httpx.ReadTimeout("slow"))

    async with httpx.AsyncClient() as http:
        with pytest.raises(WebSearchError) as excinfo:
            await _client(settings_factory, http=http).search("q", 5)

    assert excinfo.value.code == "web_search_timeout"
    assert "超时" in str(excinfo.value)


@respx.mock
async def test_mcp_fallback_timeout(settings_factory, monkeypatch):
    monkeypatch.setattr(exa_mod, "npx_available", lambda: True)
    respx.post(API_URL).mock(return_value=httpx.Response(500))

    async def mcp_caller(query, max_results):
        await asyncio.sleep(0.3)
        return ("web_search_exa", "")

    async with httpx.AsyncClient() as http:
        client = _client(settings_factory, mcp_caller=mcp_caller, http=http, exa_mcp_timeout=0.05)
        with pytest.raises(WebSearchError) as excinfo:
            await client.search("q", 5)

    assert excinfo.value.code == "web_search_timeout"


async def test_unavailable_without_key(settings_factory):
    async def mcp_caller(query, max_results):  # pragma: no cover
        raise AssertionError("不可用时不应发起搜索")

    client = ExaSearchClient(settings_factory(exa_api_key=None), mcp_caller=mcp_caller)
    with pytest.raises(WebSearchError) as excinfo:
        await client.search("q", 5)

    assert excinfo.value.code == "web_search_unavailable"


def test_available_does_not_require_npx(settings_factory, monkeypatch):
    monkeypatch.setattr(exa_mod, "npx_available", lambda: False)
    client = _client(settings_factory)
    assert client.available is True  # 主路径是 API
    assert client.fallback_available is False


def test_parse_advanced_accepts_results_dict_and_list():
    payload = {"results": [{"url": "https://a", "title": "A", "score": 0.5}]}
    assert _parse_advanced(json.dumps(payload), 5)[0].score == 0.5
    assert _parse_advanced(json.dumps([{"url": "https://b"}]), 5)[0].url == "https://b"
    with pytest.raises(ValueError):
        _parse_advanced(json.dumps({"nope": 1}), 5)


def test_parse_basic_blocks():
    text = (
        "Title: 第一条\nURL: https://example.com/1\nPublished: 2026-01-01\n"
        "Highlights:\n摘要一\n\n---\n\n"
        "Title: 第二条\nURL: https://example.com/2\nHighlights:\n摘要二"
    )
    sources = _parse_basic(text, 10)
    assert [s.url for s in sources] == ["https://example.com/1", "https://example.com/2"]
    assert sources[0].snippet == "摘要一"
    assert sources[0].published_date == "2026-01-01"
    assert all(s.score is None and s.engine == "exa" for s in sources)


@respx.mock
async def test_mcp_fallback_bad_json_falls_back_to_text(settings_factory, monkeypatch):
    monkeypatch.setattr(exa_mod, "npx_available", lambda: True)
    respx.post(API_URL).mock(return_value=httpx.Response(503))

    async def mcp_caller(query, max_results):
        return (
            "web_search_advanced_exa",
            "Title: 回退条目\nURL: https://example.com/fallback\nHighlights:\n文本摘要",
        )

    async with httpx.AsyncClient() as http:
        sources = await _client(settings_factory, mcp_caller=mcp_caller, http=http).search("q", 5)

    assert [s.url for s in sources] == ["https://example.com/fallback"]
    assert sources[0].score is None
