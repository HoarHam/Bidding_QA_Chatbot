"""把内部对象转成适合上报给 Langfuse 的普通结构。

单独放一层的原因：这些函数既要在 Agent 循环里用（`ChatMessage`），
也要在读历史摘要、一次性回答路径里用。上报失败不能影响业务，因此这里只做
纯数据转换，不做任何网络或异常处理。
"""

from __future__ import annotations

from typing import Any


def messages_payload(messages: list[Any]) -> list[dict[str, Any]]:
    """序列化 LLM 输入消息。

    保留 role / content / tool_calls / reasoning；不保留 provider 私有字段。
    工具调用只保留名字与原始参数串——参数本身可能包含 Cypher / 查询词，
    对排查 Agent 行为是必要信息。
    """
    payload: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {"role": getattr(message, "role", "unknown")}
        content = getattr(message, "content", "") or ""
        if content:
            item["content"] = content

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            item["tool_calls"] = [
                {
                    "name": getattr(call, "name", ""),
                    "arguments": getattr(call, "arguments", ""),
                }
                for call in tool_calls
            ]

        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            item["tool_call_id"] = tool_call_id

        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            item["reasoning"] = reasoning

        payload.append(item)
    return payload


def usage_payload(usage: Any | None) -> dict[str, int] | None:
    """把 `Usage`（dataclass 或 pydantic 模型）转成 Langfuse 的 usage_details。

    Langfuse 识别的键是 input / output / total。缺哪项就省略哪项，
    不要把 None 写成 0——0 会被当成真实用量参与成本统计。
    """
    if usage is None:
        return None

    mapping = (
        ("input", "prompt_tokens"),
        ("output", "completion_tokens"),
        ("total", "total_tokens"),
    )
    details: dict[str, int] = {}
    for target, source in mapping:
        value = getattr(usage, source, None)
        if value is not None:
            details[target] = int(value)
    return details or None


def tool_output_payload(result: Any) -> dict[str, Any]:
    """工具结果的紧凑视图（正文由 TraceHandle 按内容开关决定是否截断/省略）。"""
    artifacts = getattr(result, "artifacts", None) or {}
    return {
        "content": getattr(result, "content", "") or "",
        "error": getattr(result, "error", None),
        "sources": len(artifacts.get("sources") or []),
        "web_sources": len(artifacts.get("web_sources") or []),
        "degraded": bool(artifacts.get("degraded")),
    }
