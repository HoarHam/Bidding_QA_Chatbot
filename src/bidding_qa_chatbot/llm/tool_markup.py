"""智谱 GLM 工具调用文本标记的流式过滤与还原。

GLM 在部分场景（尤其 thinking 模式、回退到智谱时）会把工具调用**当正文**返回：

    <tool_call>search_bidding_knowledge
    <arg_key>query</arg_key><arg_value>2023年 采购人 交易</arg_value>
    </tool_call>

原样透传会让用户看到内部标记，工具也不会执行。本模块：

1. 跨 chunk 暂存可能的标记片段，绝不外泄半个标记；
2. 把 `<tool_call>` + 多组 `<arg_key>/<arg_value>` 还原成 `ToolCall`
   （值尝试 JSON 解析，失败则保留字符串）；
3. 未闭合/残缺标记直接丢弃。

只处理模型输出，不执行任何东西。
"""

from __future__ import annotations

import json
import re
from typing import Any

from bidding_qa_chatbot.llm.base import ToolCall

_TAG = re.compile(
    r"<\s*(?P<close>/?)\s*(?P<kind>tool_call|arg_key|arg_value)\s*>",
    re.IGNORECASE,
)


def _coerce(value: str) -> Any:
    text = value.strip()
    try:
        return json.loads(text)
    except ValueError:
        return text


class ZhipuToolCallFilter:
    """按 chunk 消费模型输出，返回可见文本与新解析出的工具调用。"""

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False
        self._name_parts: list[str] = []
        self._name_done = False
        self._field: str | None = None
        self._key: str | None = None
        self._args: dict[str, Any] = {}
        self._calls: list[ToolCall] = []
        self._call_seq = 0

    def feed(self, text: str) -> tuple[str, list[ToolCall]]:
        self._buffer += text
        return self._drain(final=False)

    def finish(self) -> tuple[str, list[ToolCall]]:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> tuple[str, list[ToolCall]]:
        visible: list[str] = []
        while True:
            match = _TAG.search(self._buffer)
            if match is None:
                if final:
                    if self._inside or _looks_like_prefix(self._buffer):
                        self._buffer = ""
                    else:
                        visible.append(self._buffer)
                        self._buffer = ""
                elif self._inside:
                    # 标记内部（工具名或参数值）：必须等闭合标签
                    break
                else:
                    cut = self._safe_cut()
                    if cut > 0:
                        visible.append(self._buffer[:cut])
                        self._buffer = self._buffer[cut:]
                break

            pre = self._buffer[: match.start()]
            if not self._inside:
                visible.append(pre)

            closing = bool(match.group("close"))
            kind = (match.group("kind") or "").lower()
            self._handle_tag(kind, closing, pre)
            self._buffer = self._buffer[match.end() :]

        calls, self._calls = self._calls, []
        return "".join(visible), calls

    def _safe_cut(self) -> int:
        index = self._buffer.rfind("<")
        if index == -1:
            return len(self._buffer)
        if ">" not in self._buffer[index:]:
            return index
        return len(self._buffer)

    def _handle_tag(self, kind: str, closing: bool, pre: str) -> None:
        if kind == "tool_call":
            if closing:
                self._finish_call()
                self._inside = False
            else:
                self._finish_call()
                self._reset_call_state()
                self._inside = True
            return

        if not self._inside:
            return

        if kind == "arg_key":
            if closing:
                self._key = pre.strip()
                self._field = None
            else:
                if not self._name_done:
                    self._name_parts.append(pre)
                    self._name_done = True
                self._field = "arg_key"
            return

        if kind == "arg_value":
            if closing:
                if self._key is not None:
                    self._args[self._key] = _coerce(pre)
                    self._key = None
                self._field = None
            else:
                if not self._name_done:
                    self._name_parts.append(pre)
                    self._name_done = True
                self._field = "arg_value"

    def _reset_call_state(self) -> None:
        self._name_parts = []
        self._name_done = False
        self._field = None
        self._key = None
        self._args = {}

    def _finish_call(self) -> None:
        if not self._inside:
            self._reset_call_state()
            return
        name = "".join(self._name_parts).strip().strip('"').strip()
        args = self._args
        self._reset_call_state()
        if not name:
            return
        self._call_seq += 1
        self._calls.append(
            ToolCall(
                id=f"call_zhipu_{self._call_seq}",
                name=name,
                arguments=json.dumps(args, ensure_ascii=False),
            )
        )


def _looks_like_prefix(text: str) -> bool:
    if not text or "<" not in text:
        return False
    tail = text[text.rfind("<") :].lower()
    return "tool_call" in tail or "arg_key" in tail or "arg_value" in tail or "<arg" in tail


def strip_zhipu_tool_calls(text: str) -> str:
    """非流式场景：去掉文本中的 GLM 工具调用标记（不关心还原出的调用）。"""
    if not text:
        return text
    stream_filter = ZhipuToolCallFilter()
    visible, _ = stream_filter.feed(text)
    tail, _ = stream_filter.finish()
    return visible + tail
