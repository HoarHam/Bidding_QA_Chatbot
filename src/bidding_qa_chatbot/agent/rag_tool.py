"""把 RAG 检索注册成工具。

`description` 写得比较具体，因为它是模型判断「要不要调用这个工具」的**唯一依据**。
招投标是垂直领域、模型自身记忆里的法规时效和金额档位极不可靠，
所以这里明确要求：涉及法规、时限、金额、资格条件的问题都必须先查知识库。
"""

from __future__ import annotations

import time
from typing import Any

from bidding_qa_chatbot.agent.base import ToolResult, ToolSpec
from bidding_qa_chatbot.config import Settings, get_settings
from bidding_qa_chatbot.retrieval.service import RetrievedSource, Retriever

#: 回灌给模型的单条资料最大长度。太长会挤占上下文，也拖慢生成。
MAX_SNIPPET_CHARS = 1200


class RagSearchTool:
    """语义 + 关键词混合检索招投标知识库。"""

    spec = ToolSpec(
        name="search_bidding_knowledge",
        description=(
            "检索招投标与政府采购知识库，返回与问题最相关的问答条目及出处。\n"
            "凡涉及以下内容，必须先调用本工具，不要凭自身记忆回答：\n"
            "- 法律法规条款、政策依据（如《政府采购法》《招标投标法》）；\n"
            "- 各类期限与时限（质疑期、投诉期、公告期、退还保证金期限等）；\n"
            "- 金额门槛、比例、资格条件、评审办法；\n"
            "- 具体业务操作流程与责任主体。\n"
            "如果检索结果不足以回答，应如实说明，而不是编造。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "检索用的查询语句。把用户的问题改写成检索友好的表述，"
                        "保留关键术语（如「单一来源采购」「投标保证金」）。"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回条数，默认 5。问题较宽泛时可适当调大。",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    )

    def __init__(self, retriever: Retriever, settings: Settings | None = None) -> None:
        self.retriever = retriever
        self.settings = settings or get_settings()

    async def run(self, query: str = "", top_k: int | None = None, **_: Any) -> ToolResult:
        started = time.perf_counter()

        # 编码器缺失时给一条**可转述**的失败信息。
        # 之所以仍注册这个工具而不是干脆不给模型：不注册的话模型会转而
        # 凭自身记忆回答招投标问题——本领域里那比「答不出来」危险得多。
        if not self.retriever.available:
            return ToolResult(
                name=self.spec.name,
                content=(
                    "知识库检索当前不可用：本地嵌入模型未就绪，无法进行语义编码。"
                    "请如实告知用户知识库暂时无法访问，不要凭自身记忆回答法规、"
                    "期限或金额类问题。"
                ),
                ok=False,
                error="retriever_unavailable",
            )

        query = (query or "").strip()
        if not query:
            return ToolResult(
                name=self.spec.name,
                content="错误：query 不能为空，请提供要检索的问题。",
                ok=False,
                error="empty_query",
            )

        result = await self.retriever.retrieve(query, top_k=top_k)
        elapsed = int((time.perf_counter() - started) * 1000)

        if not result.sources:
            return ToolResult(
                name=self.spec.name,
                content=(
                    "知识库中没有检索到与该问题相关的内容。"
                    "请如实告知用户「根据知识库无法确定」，不要凭记忆作答。"
                ),
                ok=True,
                artifacts={
                    "sources": [],
                    "degraded": result.degraded,
                    "cache_hit": result.cache_hit,
                },
                elapsed_ms=elapsed,
            )

        content = self._format(result.sources)
        if result.degraded:
            # 稀疏检索不可用时召回质量会下降，提醒模型更保守一些。
            content = (
                "（注意：本次仅使用了语义检索，关键词检索不可用，结果可能不完整）\n\n" + content
            )

        return ToolResult(
            name=self.spec.name,
            content=content,
            ok=True,
            artifacts={
                "sources": [s.to_ref().model_dump(mode="json") for s in result.sources],
                "retrieved": len(result.sources),
                "degraded": result.degraded,
                "degrade_reason": result.reason,
                "cache_hit": result.cache_hit,
            },
            elapsed_ms=elapsed,
        )

    @staticmethod
    def _format(sources: list[RetrievedSource]) -> str:
        blocks: list[str] = []
        for i, src in enumerate(sources, start=1):
            body = src.answer
            if len(body) > MAX_SNIPPET_CHARS:
                body = body[:MAX_SNIPPET_CHARS] + "…"
            block = f"【资料{i}】出处：{src.source_label}\n问：{src.question}\n答：{body}"
            described = [img for img in src.images if img.get("description")]
            if described:
                block += "\n图片内容：\n" + "\n\n".join(
                    str(img["description"]) for img in described
                )
            blocks.append(block)
        return "\n\n".join(blocks)
