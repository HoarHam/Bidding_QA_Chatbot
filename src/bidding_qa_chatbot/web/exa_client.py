"""Exa 语义搜索客户端：HTTP API 为主，MCP（npx）为回退。

## 为什么主路径是 API 而不是 MCP

实测（2026-10，exa-mcp-server 3.4.2）：
- MCP 的 `web_search_advanced_exa` 响应会被裁剪掉 `score`，拿不到相关性评分；
- 另发一次 Exa API 查询再按 URL 合并行不通——MCP（auto）与 API（neural）的
  结果集实测完全不重叠；
- Exa `/search` 只有 `type="neural"` 的响应带真实 `score`。

因此语义搜索以 **Exa HTTP API（neural）** 为主路径（结果与评分同源），
API 不可用且本机有 npx 时回退到 `npx -y exa-mcp-server`（MCP stdio，
`web_search_advanced_exa`；该回退路径没有评分）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.log import redact
from bidding_qa_chatbot.schemas import WebSource
from bidding_qa_chatbot.web.errors import WebSearchError

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 400
ADVANCED_TOOL = "web_search_advanced_exa"
BASIC_TOOL = "web_search_exa"

EXA_SEARCH_URL = "https://api.exa.ai/search"

#: 注入缝：MCP 回退调用，返回 (tool_name, text_content)。
McpCaller = Callable[[str, int], Awaitable[tuple[str, str]]]

_FIELD_RE = re.compile(r"^(Title|URL|Published|Author|Highlights)\s*:\s*(.*)$", re.I)


def npx_available() -> bool:
    """MCP 回退是否可用。Windows 上 `shutil.which("npx")` 会命中 npx.cmd。"""
    return shutil.which("npx") is not None


def _snippet_from(item: dict[str, Any]) -> str:
    highlights = item.get("highlights")
    if isinstance(highlights, list):
        text = " ".join(str(h) for h in highlights if h)
    elif isinstance(highlights, str):
        text = highlights
    else:
        text = ""
    if not text:
        text = str(item.get("summary") or item.get("text") or "")
    return " ".join(text.split())[:SNIPPET_CHARS]


def _map_api_results(data: Any, limit: int) -> list[WebSource]:
    results = data.get("results") if isinstance(data, dict) else None
    sources: list[WebSource] = []
    for item in results or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        raw_score = item.get("score")
        published = item.get("publishedDate") or item.get("published_date")
        sources.append(
            WebSource(
                url=url,
                title=str(item.get("title") or url).strip(),
                snippet=_snippet_from(item),
                score=float(raw_score) if isinstance(raw_score, int | float) else None,
                engine="exa",
                published_date=str(published) if published else None,
            )
        )
    return sources[:limit]


def _parse_advanced(text: str, limit: int) -> list[WebSource]:
    """解析 MCP `web_search_advanced_exa` 的 JSON 响应。结构异常时抛 ValueError。"""
    data = json.loads(text)
    results = data.get("results") if isinstance(data, dict) else data
    if not isinstance(results, list):
        raise ValueError("Exa 响应缺少 results 列表")
    return _map_api_results({"results": results}, limit)


def _parse_basic(text: str, limit: int) -> list[WebSource]:
    """解析 MCP `web_search_exa` 的纯文本响应（Title/URL/Published/Author/Highlights）。"""
    blocks = [block.strip() for block in text.split("\n\n---\n\n") if block.strip()]
    sources: list[WebSource] = []
    for block in blocks[:limit]:
        fields: dict[str, str] = {}
        current: str | None = None
        for line in block.splitlines():
            match = _FIELD_RE.match(line)
            if match:
                current = match.group(1).lower()
                fields[current] = match.group(2).strip()
            elif current:
                fields[current] += "\n" + line
        url = fields.get("url", "").strip()
        if not url:
            continue
        sources.append(
            WebSource(
                url=url,
                title=fields.get("title", "").strip() or url,
                snippet=fields.get("highlights", "").strip()[:SNIPPET_CHARS],
                score=None,  # basic 工具不返回相关度
                engine="exa",
                published_date=fields.get("published", "").strip() or None,
            )
        )
    return sources


def _parse_mcp_output(tool_name: str, text: str, limit: int) -> list[WebSource]:
    if tool_name == ADVANCED_TOOL:
        try:
            return _parse_advanced(text, limit)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass  # 结构变了就回退按纯文本解析
    return _parse_basic(text, limit)


class ExaSearchClient:
    def __init__(
        self,
        settings: Settings,
        *,
        mcp_caller: McpCaller | None = None,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self._mcp_caller = mcp_caller
        self.http = http
        self._semaphore = asyncio.Semaphore(max(1, settings.exa_mcp_max_concurrency))

    @property
    def available(self) -> bool:
        # 主路径是 HTTP API，只要有 key 就可用；npx 只影响回退能力。
        return self.settings.exa_api_key is not None

    @property
    def fallback_available(self) -> bool:
        return npx_available()

    async def search(self, query: str, max_results: int) -> list[WebSource]:
        if not self.available:
            raise WebSearchError("web_search_unavailable", "Exa 语义搜索不可用：EXA_API_KEY 未配置")

        api_error: WebSearchError
        try:
            return await asyncio.wait_for(
                self._api_search(query, max_results),
                timeout=self.settings.web_search_timeout,
            )
        except (TimeoutError, httpx.TimeoutException):
            api_error = WebSearchError(
                "web_search_timeout",
                f"Exa API 搜索超时（超过 {self.settings.web_search_timeout:.0f} 秒）",
            )
        except WebSearchError as exc:
            api_error = exc
        except Exception as exc:  # noqa: BLE001 — 统一转成可转述错误
            api_error = WebSearchError(
                "web_search_failed",
                f"Exa API 搜索失败：{redact(f'{type(exc).__name__}: {exc}')[:200]}",
            )

        if not self.fallback_available:
            raise api_error
        logger.warning("Exa API 搜索失败，回退到 MCP：%s", api_error)
        return await self._mcp_search(query, max_results)

    async def _api_search(self, query: str, max_results: int) -> list[WebSource]:
        api_key = self.settings.exa_api_key.get_secret_value() if self.settings.exa_api_key else ""
        headers = {"x-api-key": api_key, "content-type": "application/json"}
        payload = {
            "query": query,
            "numResults": max_results,
            # 只有 neural 响应带 score；auto/keyword/fast 都没有。
            "type": "neural",
            "contents": {"text": {"maxCharacters": SNIPPET_CHARS * 3}},
        }
        timeout = self.settings.web_search_timeout
        if self.http is not None:
            response = await self.http.post(
                EXA_SEARCH_URL, headers=headers, json=payload, timeout=timeout
            )
        else:
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.post(
                    EXA_SEARCH_URL, headers=headers, json=payload, timeout=timeout
                )
        response.raise_for_status()
        return _map_api_results(response.json(), max_results)

    async def _mcp_search(self, query: str, max_results: int) -> list[WebSource]:
        timeout = self.settings.exa_mcp_timeout
        try:
            async with self._semaphore:
                tool_name, text = await asyncio.wait_for(
                    (self._mcp_caller or self._call_mcp)(query, max_results), timeout=timeout
                )
        except TimeoutError as exc:
            raise WebSearchError(
                "web_search_timeout",
                f"Exa MCP 搜索超时（超过 {timeout:.0f} 秒；首次使用需要下载 npm 包）",
            ) from exc
        except WebSearchError:
            raise
        except Exception as exc:  # noqa: BLE001 — 子进程/协议错误统一转述
            raise WebSearchError(
                "web_search_failed",
                f"Exa MCP 搜索失败：{redact(f'{type(exc).__name__}: {exc}')[:200]}",
            ) from exc
        return _parse_mcp_output(tool_name, text, max_results)

    async def _call_mcp(self, query: str, max_results: int) -> tuple[str, str]:
        """回退实现：拉起 `npx -y exa-mcp-server` 并完成一次 tools/call。"""
        from mcp import ClientSession, StdioServerParameters, stdio_client

        api_key = self.settings.exa_api_key.get_secret_value() if self.settings.exa_api_key else ""
        params = StdioServerParameters(
            command="npx",
            args=["-y", "exa-mcp-server"],
            env={
                "EXA_API_KEY": api_key,
                "ENABLED_TOOLS": f"{ADVANCED_TOOL},{BASIC_TOOL}",
            },
        )
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            if ADVANCED_TOOL in names:
                tool_name = ADVANCED_TOOL
            elif BASIC_TOOL in names:
                tool_name = BASIC_TOOL
            else:
                raise WebSearchError(
                    "web_search_failed",
                    "Exa MCP 未提供可用的搜索工具（期望 web_search_advanced_exa / web_search_exa）",
                )
            result = await session.call_tool(tool_name, {"query": query, "numResults": max_results})
            if getattr(result, "isError", False):
                raise WebSearchError("web_search_failed", "Exa MCP 搜索返回错误")
            text = "\n".join(
                item.text for item in result.content if getattr(item, "type", "") == "text"
            )
            return tool_name, text
