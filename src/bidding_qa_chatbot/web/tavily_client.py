"""Tavily 关键词搜索客户端（异步）。

用 `AsyncTavilyClient` 并注入共享的 `app.state.http`：
- 继承项目的 `trust_env=False`（本机系统代理连不通部分目标）；
- 连接池与超时统一；
- 测试可用 respx 拦截，或注入 fake client。

缓存不在这里做：由 `WebSearchService` 统一负责（键含引擎/条数/查询）。
"""

from __future__ import annotations

from typing import Any

import httpx
from tavily.errors import TimeoutError as TavilyTimeoutError

from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.log import redact
from bidding_qa_chatbot.schemas import WebSource
from bidding_qa_chatbot.web.errors import WebSearchError

#: 回灌给模型的单条摘要最大长度
SNIPPET_CHARS = 400


def _map_results(response: Any, max_results: int) -> list[WebSource]:
    results = response.get("results") if isinstance(response, dict) else None
    sources: list[WebSource] = []
    for item in (results or [])[:max_results]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue  # 没有 URL 的结果对「可跳转来源」没有意义
        title = str(item.get("title") or url).strip()
        content = str(item.get("content") or "").strip()
        raw_score = item.get("score")
        score = float(raw_score) if isinstance(raw_score, int | float) else None
        published = item.get("published_date")
        sources.append(
            WebSource(
                url=url,
                title=title,
                snippet=content[:SNIPPET_CHARS],
                score=score,
                engine="tavily",
                published_date=str(published) if published else None,
            )
        )
    return sources


class TavilySearchClient:
    def __init__(self, settings: Settings, http: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.http = http
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from tavily import AsyncTavilyClient

            api_key = (
                self.settings.tavily_api_key.get_secret_value()
                if self.settings.tavily_api_key
                else ""
            )
            self._client = AsyncTavilyClient(api_key=api_key, client=self.http)
        return self._client

    async def search(self, query: str, max_results: int) -> list[WebSource]:
        """执行一次 basic 深度搜索。失败抛 `WebSearchError`。"""
        try:
            response = await self._get_client().search(
                query,
                search_depth="basic",
                max_results=max_results,
                include_answer=False,
                include_raw_content=False,
                timeout=self.settings.web_search_timeout,
            )
        # SDK 会把 httpx 的超时包装成 tavily.errors.TimeoutError（不是内置 TimeoutError）
        except (TimeoutError, httpx.TimeoutException, TavilyTimeoutError) as exc:
            raise WebSearchError(
                "web_search_timeout",
                f"Tavily 联网搜索超时（超过 {self.settings.web_search_timeout:.0f} 秒）",
            ) from exc
        except Exception as exc:  # noqa: BLE001 — 统一转成可转述错误
            raise WebSearchError(
                "web_search_failed",
                f"Tavily 联网搜索失败：{redact(f'{type(exc).__name__}: {exc}')[:200]}",
            ) from exc
        return _map_results(response, max_results)
