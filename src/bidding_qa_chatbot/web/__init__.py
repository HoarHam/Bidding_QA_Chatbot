"""联网搜索：Tavily（关键词）+ Exa MCP（语义）。

- `tavily_client.py`  关键词搜索（AsyncTavilyClient，注入共享 httpx 客户端）
- `exa_client.py`     语义搜索（Exa HTTP API 为主，npx MCP 为回退）
- `cache.py`          进程内 TTL 缓存（只用于 Tavily）
- `service.py`        统一门面 + 引擎可用性 + 错误映射

门控：`settings.enable_web_search`（ENABLE_WEB_SEARCH）；单次请求再由
`ChatRequest.web_search` 决定是否把工具提供给模型。工具层见 `agent/web_tools.py`。
"""

from bidding_qa_chatbot.web.errors import WebSearchError
from bidding_qa_chatbot.web.service import WebSearchResult, WebSearchService

__all__ = ["WebSearchError", "WebSearchResult", "WebSearchService"]
