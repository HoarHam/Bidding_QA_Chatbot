"""联网搜索服务：把 Tavily（关键词）与 Exa MCP（语义）统一成可注入测试的门面。

- 只缓存 Tavily 关键词搜索（需求明确要求进程内缓存；Exa 每次都是实时语义检索）。
- 引擎可用性是**静态**能力位（密钥存在 / npx 可用），与 `Retriever.available`
  同一语义：运行期是否真的连得上由 `WebSearchError` 表达。
- 所有错误都转成带中文消息的 `WebSearchError`，绝不向上抛第三方异常。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from bidding_qa_chatbot.config import Settings, get_settings
from bidding_qa_chatbot.schemas import WebSource
from bidding_qa_chatbot.web.cache import TTLCache
from bidding_qa_chatbot.web.errors import WebSearchError
from bidding_qa_chatbot.web.exa_client import ExaSearchClient
from bidding_qa_chatbot.web.tavily_client import TavilySearchClient

__all__ = ["WebSearchError", "WebSearchResult", "WebSearchService"]


@dataclass(slots=True)
class WebSearchResult:
    engine: str
    query: str
    sources: list[WebSource]
    elapsed_ms: int
    cached: bool = False


class WebSearchService:
    def __init__(
        self,
        settings: Settings | None = None,
        http: httpx.AsyncClient | None = None,
        *,
        tavily: Any = None,
        exa: Any = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.http = http
        self._tavily = tavily  # 注入缝：测试传 fake client
        self._exa = exa
        self._cache = TTLCache(
            self.settings.web_search_cache_ttl, self.settings.web_search_cache_maxsize
        )

    @property
    def keyword_available(self) -> bool:
        return self.settings.tavily_api_key is not None

    @property
    def semantic_available(self) -> bool:
        # 主路径是 Exa HTTP API（只有它返回真实 score）；npx MCP 只是回退，
        # 不影响可用性判断。
        return self.settings.exa_api_key is not None

    @property
    def available(self) -> bool:
        return self.keyword_available or self.semantic_available

    def _tavily_client(self) -> Any:
        if self._tavily is None:
            self._tavily = TavilySearchClient(self.settings, self.http)
        return self._tavily

    def _exa_client(self) -> Any:
        if self._exa is None:
            # 传共享 http（trust_env=False、连接池统一）；API 失败时客户端内部回退到 MCP
            self._exa = ExaSearchClient(self.settings, http=self.http)
        return self._exa

    def _clamp(self, max_results: int | None) -> int:
        cap = max_results if max_results is not None else self.settings.web_search_max_results
        return min(max(1, int(cap)), self.settings.web_search_max_results)

    async def keyword_search(self, query: str, max_results: int | None = None) -> WebSearchResult:
        if not self.keyword_available:
            raise WebSearchError(
                "web_search_unavailable", "Tavily 关键词搜索不可用：TAVILY_API_KEY 未配置"
            )
        cap = self._clamp(max_results)
        key = f"tavily|{cap}|{query.strip().lower()}"
        cached = await self._cache.get(key)
        if cached is not None:
            return WebSearchResult("tavily", query, list(cached), 0, cached=True)

        started = time.perf_counter()
        sources = await self._tavily_client().search(query, cap)
        await self._cache.set(key, sources)
        elapsed = int((time.perf_counter() - started) * 1000)
        return WebSearchResult("tavily", query, sources, elapsed)

    async def semantic_search(self, query: str, max_results: int | None = None) -> WebSearchResult:
        if not self.semantic_available:
            raise WebSearchError(
                "web_search_unavailable",
                "Exa 语义搜索不可用：EXA_API_KEY 未配置或未找到 npx（需要 Node.js 20+）",
            )
        cap = self._clamp(max_results)
        started = time.perf_counter()
        sources = await self._exa_client().search(query, cap)
        elapsed = int((time.perf_counter() - started) * 1000)
        return WebSearchResult("exa", query, sources, elapsed)
