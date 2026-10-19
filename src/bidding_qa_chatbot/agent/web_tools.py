"""把联网搜索注册成两个工具：关键词（Tavily）与语义（Exa）。

为什么是两个工具而不是一个带 `engine` 参数的：让模型直接按工具描述选择引擎，
「关键词型 / 概念型」的分工更清晰；代价只是多一份 schema（描述控制在短句内）。

可用性语义与 `RagSearchTool` 一致：**已注册但引擎暂时不可用时**，工具返回一条
可转述的失败信息，而不是消失——否则模型会转而凭记忆编造外部信息。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from bidding_qa_chatbot.agent.base import ToolResult, ToolSpec
from bidding_qa_chatbot.config import Settings, get_settings
from bidding_qa_chatbot.log import redact
from bidding_qa_chatbot.schemas import WebSource
from bidding_qa_chatbot.web import WebSearchError, WebSearchResult, WebSearchService

logger = logging.getLogger(__name__)

DEFAULT_RESULTS = 5
MAX_RESULTS = 10
_SNIPPET_CHARS = 240

_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "检索语句：关键词模式写简洁的关键词组合；语义模式写一句完整的自然语言问题。"
            ),
        },
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_RESULTS,
            "description": f"返回条数，默认 {DEFAULT_RESULTS}。",
        },
    },
    "required": ["query"],
}


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_sources(sources: list[WebSource]) -> str:
    blocks: list[str] = []
    for i, src in enumerate(sources, start=1):
        score = f"{src.score:.2f}" if src.score is not None else "—"
        snippet = (src.snippet or "（无摘要）")[:_SNIPPET_CHARS]
        blocks.append(f"【结果{i}】{src.title}\nURL：{src.url}\n相关度：{score}\n摘要：{snippet}")
    return "\n\n".join(blocks)


class _WebToolBase:
    """两个工具共用的执行流程。子类只声明 spec 与引擎标识。"""

    engine_label = ""
    available_attr = ""

    spec: ToolSpec

    def __init__(self, service: WebSearchService | None, settings: Settings | None = None) -> None:
        self.service = service
        self.settings = settings or get_settings()

    def _available(self) -> bool:
        return bool(self.service is not None and getattr(self.service, self.available_attr, False))

    async def run(self, query: str = "", max_results: int | None = None) -> ToolResult:
        started = time.perf_counter()
        if not self._available():
            return ToolResult(
                name=self.spec.name,
                ok=False,
                error="web_search_unavailable",
                content=(
                    f"{self.engine_label}当前不可用（服务端未配置相应搜索服务）。"
                    "请如实告知用户无法进行联网搜索，不要凭自身记忆编造外部信息；"
                    "可以建议用户改用本地知识库能覆盖的问题。"
                ),
            )

        query = (query or "").strip()
        if not query:
            return ToolResult(
                name=self.spec.name,
                ok=False,
                error="empty_query",
                content="错误：query 不能为空，请提供要检索的内容。",
            )

        value = _as_int(max_results)
        if max_results is not None and value is None:
            return ToolResult(
                name=self.spec.name,
                ok=False,
                error="bad_arguments",
                content=f"错误：max_results 必须是 1-{MAX_RESULTS} 的整数，收到：{max_results!r}。",
            )
        limit = min(max(1, value if value is not None else DEFAULT_RESULTS), MAX_RESULTS)

        try:
            result = await self._search(query, limit)
        except WebSearchError as exc:
            return ToolResult(
                name=self.spec.name,
                ok=False,
                error=exc.code,
                content=(
                    f"联网搜索失败：{exc}。请如实告知用户联网搜索暂时不可用，"
                    "不要凭记忆编造外部信息，也不要假装查到了结果。"
                ),
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:  # noqa: BLE001 — 工具失败绝不打断对话
            logger.exception("联网搜索工具 %s 异常", self.spec.name)
            return ToolResult(
                name=self.spec.name,
                ok=False,
                error="web_search_failed",
                content=(
                    f"联网搜索失败：{redact(str(exc))[:200]}。"
                    "请如实告知用户联网搜索暂时不可用，不要编造外部信息。"
                ),
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

        elapsed = int((time.perf_counter() - started) * 1000)
        if not result.sources:
            return ToolResult(
                name=self.spec.name,
                ok=True,
                content=(
                    "联网搜索没有找到与该问题相关的结果。"
                    "请如实告知用户「联网搜索没有找到结果」，不要用记忆编造外部信息；"
                    "可以建议换个问法，或说明本地知识库里是否有相关内容。"
                ),
                artifacts={"summary": "未检索到联网结果"},
                elapsed_ms=elapsed,
            )

        content = (
            f"【联网搜索结果·{self.engine_label}】查询：{query}\n\n"
            + _format_sources(result.sources)
            + "\n\n（请只依据以上结果作答；引用时必须给出上面的 URL，"
            "不要编造结果中没有的信息。）"
        )
        return ToolResult(
            name=self.spec.name,
            ok=True,
            content=content,
            artifacts={
                "summary": f"联网搜索到 {len(result.sources)} 条结果",
                "web": {
                    "engine": result.engine,
                    "query": query,
                    "cached": result.cached,
                    "elapsed_ms": result.elapsed_ms,
                },
                # 注意：绝不能放进 artifacts["sources"]——那是 SourceRef 的通道，
                # routes_chat 会对每项做 SourceRef.model_validate（联网结构必然校验失败）。
                "web_sources": [s.model_dump(mode="json") for s in result.sources],
            },
            elapsed_ms=elapsed,
        )

    async def _search(self, query: str, limit: int) -> WebSearchResult:
        raise NotImplementedError


class WebSearchKeywordTool(_WebToolBase):
    """Tavily 关键词联网搜索。"""

    engine_label = "Tavily 关键词搜索"
    available_attr = "keyword_available"

    spec = ToolSpec(
        name="search_web_keyword",
        description=(
            "联网关键词搜索（Tavily）。用于核实**时效性强的外部公开信息**：最新政策/新闻/公告、"
            "具体机构或项目的公开报道、需要外部资料佐证的具体名词与数字。\n"
            "仅在用户开启「联网搜索」时可用。优先使用本地知识库与知识图谱；"
            "本地信息不足或明显过时再用本工具。\n"
            "查询用简洁的关键词组合（如「2026 政府采购 节能产品 政策」）。\n"
            "返回结果含标题、URL 与相关度；引用时必须给出 URL，不要编造结果中没有的信息。"
        ),
        parameters=_PARAMETERS,
    )

    async def _search(self, query: str, limit: int) -> WebSearchResult:
        return await self.service.keyword_search(query, limit)


class WebSearchSemanticTool(_WebToolBase):
    """Exa 语义联网搜索。"""

    engine_label = "Exa 语义搜索"
    available_attr = "semantic_available"

    spec = ToolSpec(
        name="search_web_semantic",
        description=(
            "联网语义搜索（Exa）。用于**概念性、开放式、多角度**的问题：需要理解意图并从多个"
            "公开来源综合信息（如「政府采购支持中小企业的最新做法有哪些」）。\n"
            "仅在用户开启「联网搜索」时可用。优先使用本地知识库与知识图谱。\n"
            "查询写成一句完整的自然语言问题。\n"
            "返回结果含标题、URL 与相关度；引用时必须给出 URL，不要编造结果中没有的信息。"
        ),
        parameters=_PARAMETERS,
    )

    async def _search(self, query: str, limit: int) -> WebSearchResult:
        return await self.service.semantic_search(query, limit)
