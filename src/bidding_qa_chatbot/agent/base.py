"""工具的抽象。

工具对模型只暴露三件事：名字、用途说明、参数 JSON Schema。
`description` 是模型决定「要不要调用、调用哪个」的**唯一依据**，
所以它要写清楚「什么时候该用」，而不只是「这是什么」。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from bidding_qa_chatbot.log import redact

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_schema(self) -> dict[str, Any]:
        """OpenAI / 智谱通用的 function calling 结构。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class ToolResult:
    """工具执行结果。

    `content` 是回灌给模型的文本；`ok=False` 时也照样回灌，
    让模型知道工具失败了并据此调整（而不是假装成功）。
    `artifacts` 放结构化附加数据（如引用来源），供 API 层推给前端，
    模型看不到。
    """

    name: str
    content: str
    ok: bool = True
    error: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0


@runtime_checkable
class Tool(Protocol):
    spec: ToolSpec

    async def run(self, **kwargs: Any) -> ToolResult: ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """传给模型的工具声明。空注册表返回空列表，调用方据此决定是否带 tools 参数。"""
        return [t.spec.to_openai_schema() for t in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """执行工具。**任何异常都不向上抛**——工具失败是要告诉模型的信息，
        不是要打断整个对话的错误。"""
        tool = self._tools.get(name)
        if tool is None:
            logger.warning("模型请求了未注册的工具: %s", name)
            return ToolResult(
                name=name,
                content=f"错误：不存在名为 {name} 的工具。可用工具：{self.names}",
                ok=False,
                error="unknown_tool",
            )
        try:
            return await tool.run(**arguments)
        except TypeError as exc:
            # 多半是模型给的参数不对（缺参数/多了参数/名字写错）
            logger.warning("工具 %s 参数不匹配: %s", name, exc)
            return ToolResult(
                name=name,
                content=f"错误：工具 {name} 的参数不正确（{exc}）。请检查参数后重试。",
                ok=False,
                error="bad_arguments",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("工具 %s 执行失败", name)
            return ToolResult(
                name=name,
                content=f"错误：工具 {name} 执行失败：{redact(str(exc))[:200]}",
                ok=False,
                error=type(exc).__name__,
            )
