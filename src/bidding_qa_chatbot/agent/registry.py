"""默认工具集的注册点。

**新增工具在这里注册即可，Agent 循环无需改动**（docs/agent.md 的约定）。
单独成模块而不是留在 `rag_tool.py` 里：检索工具不该负责注册图谱工具。
"""

from __future__ import annotations

import logging
from typing import Any

from bidding_qa_chatbot.agent.base import ToolRegistry
from bidding_qa_chatbot.agent.graph_tool import GraphQueryTool
from bidding_qa_chatbot.agent.rag_tool import RagSearchTool
from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.retrieval.service import Retriever

logger = logging.getLogger(__name__)


def build_default_registry(
    retriever: Retriever,
    settings: Settings | None = None,
    *,
    graph: Any | None = None,
) -> ToolRegistry:
    """构造默认工具集。

    `graph`（`GraphService`）为 None 时不注册图谱工具——功能关闭（ENABLE_NEO4J=false）
    时模型不该看到一个永远不可用的工具。已注册但服务器宕机是另一回事：
    工具保留并返回明确的不可用提示（不注册的话模型会转而凭记忆编造交易数据，
    与 rag_tool 对「编码器缺失」的处理同一理由）。
    """
    registry = ToolRegistry()
    registry.register(RagSearchTool(retriever, settings))
    if graph is not None:
        registry.register(GraphQueryTool(graph, settings))
    logger.info("已注册工具: %s", registry.names)
    return registry
