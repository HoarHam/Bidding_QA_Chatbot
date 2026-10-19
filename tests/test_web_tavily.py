"""Tavily 关键词搜索：结果映射、URL 过滤、超时/失败映射。

HTTP 层用 respx 拦截 `https://api.tavily.com/search`（AsyncTavilyClient 注入
我们自己的 httpx.AsyncClient）。不发起任何真实请求。
"""

import httpx
import pytest
import respx

from bidding_qa_chatbot.web.errors import WebSearchError
from bidding_qa_chatbot.web.tavily_client import TavilySearchClient


def _client(settings_factory, http: httpx.AsyncClient) -> TavilySearchClient:
    return TavilySearchClient(settings_factory(tavily_api_key="tvly-test"), http=http)


@respx.mock
async def test_maps_results(settings_factory):
    route = respx.post("https://api.tavily.com/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "政府采购新规解读",
                        "url": "https://example.com/a",
                        "content": "这是摘要内容。",
                        "score": 0.87,
                        "published_date": "2026-09-01",
                    },
                    {"title": "无链接条目", "content": "应被过滤", "score": 0.5},
                ]
            },
        )
    )

    async with httpx.AsyncClient() as http:
        sources = await _client(settings_factory, http).search("政府采购 新规", 5)

    assert route.called
    assert len(sources) == 1
    src = sources[0]
    assert src.url == "https://example.com/a"
    assert src.title == "政府采购新规解读"
    assert src.snippet == "这是摘要内容。"
    assert src.score == 0.87
    assert src.engine == "tavily"
    assert src.published_date == "2026-09-01"

    # 请求体携带 basic 深度
    body = route.calls[0].request.content.decode("utf-8")
    assert '"search_depth":"basic"' in body.replace(" ", "")


@respx.mock
async def test_timeout_maps_to_web_search_timeout(settings_factory):
    respx.post("https://api.tavily.com/search").mock(side_effect=httpx.ReadTimeout("slow"))

    async with httpx.AsyncClient() as http:
        with pytest.raises(WebSearchError) as excinfo:
            await _client(settings_factory, http).search("q", 5)

    assert excinfo.value.code == "web_search_timeout"
    assert "超时" in str(excinfo.value)


@respx.mock
async def test_http_error_maps_to_web_search_failed(settings_factory):
    respx.post("https://api.tavily.com/search").mock(
        return_value=httpx.Response(401, json={"detail": "unauthorized"})
    )

    async with httpx.AsyncClient() as http:
        with pytest.raises(WebSearchError) as excinfo:
            await _client(settings_factory, http).search("q", 5)

    assert excinfo.value.code == "web_search_failed"
    assert "Tavily 联网搜索失败" in str(excinfo.value)


@respx.mock
async def test_empty_results(settings_factory):
    respx.post("https://api.tavily.com/search").mock(
        return_value=httpx.Response(200, json={"results": []})
    )

    async with httpx.AsyncClient() as http:
        sources = await _client(settings_factory, http).search("q", 5)

    assert sources == []


@respx.mock
async def test_respects_max_results(settings_factory):
    respx.post("https://api.tavily.com/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": f"T{i}",
                        "url": f"https://example.com/{i}",
                        "content": "",
                        "score": 0.1,
                    }
                    for i in range(5)
                ]
            },
        )
    )

    async with httpx.AsyncClient() as http:
        sources = await _client(settings_factory, http).search("q", 2)

    assert len(sources) == 2
