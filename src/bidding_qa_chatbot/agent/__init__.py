"""Agent 层：把能力注册成工具，由模型自行决定何时调用。

当前只有一个工具（RAG 知识库检索），但抽象是按「多工具」设计的——
加新工具只需实现 `Tool` 协议并注册，Agent 循环无需改动。
"""

from bidding_qa_chatbot.agent.agent import (
    AgentDone,
    AgentItem,
    RagAgent,
    ToolCallNotice,
    ToolResultNotice,
)
from bidding_qa_chatbot.agent.base import Tool, ToolRegistry, ToolResult, ToolSpec
from bidding_qa_chatbot.agent.rag_tool import RagSearchTool, build_default_registry

__all__ = [
    "AgentDone",
    "AgentItem",
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
