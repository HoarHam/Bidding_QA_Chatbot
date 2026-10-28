"""LLM 调用与 Agent 行为可观测性（Langfuse）。

对外只暴露三个东西：构造入口、trace 句柄、payload 转换函数。
Langfuse SDK 只在 `service.py` 内部导入，未启用时整个包不加载 OTel。
"""

from bidding_qa_chatbot.observability.payloads import (
    messages_payload,
    tool_output_payload,
    usage_payload,
)
from bidding_qa_chatbot.observability.service import (
    LangfuseObservability,
    NoopObservability,
    TraceHandle,
    build_observability,
)

__all__ = [
    "LangfuseObservability",
    "NoopObservability",
    "TraceHandle",
    "build_observability",
    "messages_payload",
    "tool_output_payload",
    "usage_payload",
]
