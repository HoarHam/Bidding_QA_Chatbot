"""Agent 层：把能力注册成工具，由模型自行决定何时调用。

当前有两个工具：RAG 知识库检索（`rag_tool`）与 Neo4j 知识图谱查询（`graph_tool`），
抽象按「多工具」设计——加新工具只需实现 `Tool` 协议并在 `registry.py` 注册，
Agent 循环无需改动。
"""

from bidding_qa_chatbot.agent.agent import (
    AgentDone,
    AgentItem,
    RagAgent,
    ToolCallNotice,
    ToolResultNotice,
)
from bidding_qa_chatbot.agent.base import Tool, ToolRegistry, ToolResult, ToolSpec
from bidding_qa_chatbot.agent.graph_tool import GraphQueryTool
from bidding_qa_chatbot.agent.rag_tool import RagSearchTool
from bidding_qa_chatbot.agent.registry import build_default_registry

__all__ = [
    "AgentDone",
    "AgentItem",
    "GraphQueryTool",
    "RagAgent",
    "RagSearchTool",
    "Tool",
    "ToolCallNotice",
    "ToolRegistry",
    "ToolResult",
    "ToolResultNotice",
    "ToolSpec",
    "build_default_registry",
]
