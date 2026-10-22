"""DeepSeek DSML 工具调用标记的流式过滤与还原。

DeepSeek 在部分场景（尤其 thinking 模式、长上下文）会把工具调用以 DSML 标记
**当正文/思维内容流式返回**，例如：

    <｜｜DSML｜｜ calls>
    <｜｜DSML｜｜ invoke name="query_bidding_graph">
    <｜｜DSML｜｜ parameter name="cypher" string="true">MATCH ...</｜｜DSML｜｜ parameter>
    </｜｜DSML｜｜ invoke>
    </｜｜DSML｜｜ calls>

如果原样透传，用户会看到这些内部标记，工具也不会被执行。本模块负责：

1. **流式过滤**：标记可以跨 chunk 拆分，过滤器会保留可能构成标记的尾部片段，
   绝不把半个标记发出去；
2. **还原工具调用**：识别 invoke/parameter 结构，生成 `ToolCall`（参数按
   `string="false"` 尝试 JSON 解析，否则按字符串）；
3. **容错**：兼容单/双全角竖线、ASCII 竖线、`DSML` 后带空格、缺少外层
   `calls` 包裹、未闭合的残片（残片直接丢弃，不泄露）。

安全边界：这里只处理**模型输出**，不执行任何东西；调用是否合法仍由 Agent 的
工具注册表决定。
"""

from __future__ import annotations

import json
import re
from typing import Any

from bidding_qa_chatbot.llm.base import ToolCall

_DSML_TAG = re.compile(
    r"<\s*(?P<close>/?)\s*[｜|]{1,2}\s*DSML\s*[｜|]{1,2}\s*"
    r"(?P<kind>[A-Za-z_]+)?(?P<attrs>[^>]*)>",
    re.IGNORECASE,
)
_WRAPPERS = {"calls", "tool_calls", "tool_call", "function_calls"}
_NAME_ATTR = re.compile(r'name\s*=\s*"([^"]*)"', re.IGNORECASE)
_STRING_ATTR = re.compile(r'string\s*=\s*"([^"]*)"', re.IGNORECASE)


def _convert_value(raw: str, string_flag: str | None) -> Any:
    text = raw.strip()
    if string_flag is None or string_flag.lower() == "true":
        return raw
    try:
        return json.loads(text)
    except ValueError:
        return raw


class DSMLStreamFilter:
    """按 chunk 消费模型输出，返回可见文本与新解析出的工具调用。"""

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False
        self._wrapped = False
        self._current: dict[str, Any] | None = None
        self._calls: list[ToolCall] = []
        self._call_seq = 0

    def feed(self, text: str) -> tuple[str, list[ToolCall]]:
        self._buffer += text
        return self._drain(final=False)

    def finish(self) -> tuple[str, list[ToolCall]]:
        return self._drain(final=True)

    # ------------------------------------------------------------ 内部
    def _drain(self, *, final: bool) -> tuple[str, list[ToolCall]]:
        visible: list[str] = []
        while True:
            match = _DSML_TAG.search(self._buffer)
            if match is None:
                if final:
                    # 未闭合的 DSML 块：整段丢弃；否则按普通文本输出
                    if self._inside or _looks_like_dsml_prefix(self._buffer):
                        self._buffer = ""
                    else:
                        visible.append(self._buffer)
                        self._buffer = ""
                elif self._inside:
                    # 标记内部：文本可能是参数值，必须等闭合标签，不能当正文发出去
                    break
                else:
                    cut = self._safe_cut()
                    if cut > 0:
                        visible.append(self._buffer[:cut])
                        self._buffer = self._buffer[cut:]
                break

            pre = self._buffer[: match.start()]
            # 标记内部（wrapper/invoke/parameter 之间）的文本不展示
            if not self._inside:
                visible.append(pre)
            self._handle_tag(match, pre)
            self._buffer = self._buffer[match.end() :]

        calls, self._calls = self._calls, []
        return "".join(visible), calls

    def _safe_cut(self) -> int:
        """保留可能构成标记的尾部：从最后一个未闭合的 `<` 起暂存。"""
        index = self._buffer.rfind("<")
        if index == -1:
            return len(self._buffer)
        if ">" not in self._buffer[index:]:
            return index
        return len(self._buffer)

    def _handle_tag(self, match: re.Match[str], pre: str) -> None:
        closing = bool(match.group("close"))
        kind = (match.group("kind") or "").lower()
        attrs = match.group("attrs") or ""

        if kind in _WRAPPERS:
            if closing:
                self._finish_invoke()
                self._inside = False
                self._wrapped = False
            else:
                # 外层包裹开始；如果此前没有 wrapper，invoke 也会自行开启
                self._inside = True
                self._wrapped = True
                self._current = None
            return

        if kind == "invoke":
            if closing:
                self._finish_invoke(pre)
                if not self._wrapped:
                    self._inside = False
            else:
                self._finish_invoke()
                name_match = _NAME_ATTR.search(attrs)
                self._wrapped = self._inside and self._wrapped
                self._inside = True
                self._current = {
                    "name": name_match.group(1).strip() if name_match else "",
                    "params": {},
                    "param": None,
                    "body": "",
                }
            return

        if kind == "parameter":
            if self._current is None:
                return
            if closing:
                param = self._current.get("param")
                if param:
                    self._current["params"][param["name"]] = _convert_value(pre, param["string"])
                    self._current["param"] = None
            else:
                name_match = _NAME_ATTR.search(attrs)
                string_match = _STRING_ATTR.search(attrs)
                self._current["param"] = {
                    "name": name_match.group(1).strip() if name_match else "",
                    "string": string_match.group(1) if string_match else None,
                }
            return

        # 其他 DSML 标签：标记内部就丢掉；若在标记外出现，也不展示
        if closing and not self._wrapped:
            self._inside = False

    def _finish_invoke(self, body: str = "") -> None:
        current = self._current
        self._current = None
        if not current or not current["name"]:
            return
        params = current["params"]
        if not params and body.strip():
            try:
                decoded = json.loads(body.strip())
            except ValueError:
                decoded = None
            if isinstance(decoded, dict):
                params = decoded
        if not params and not body.strip():
            # 无参数的 invoke 也允许（工具参数全可选）
            params = {}
        self._call_seq += 1
        self._calls.append(
            ToolCall(
                id=f"call_dsml_{self._call_seq}",
                name=current["name"],
                arguments=json.dumps(params, ensure_ascii=False),
            )
        )


def _looks_like_dsml_prefix(text: str) -> bool:
    if not text or "<" not in text:
        return False
    index = text.rfind("<")
    tail = text[index:]
    upper = tail.upper()
    return "DSML" in upper or "｜" in tail or "|" in tail


def strip_dsml(text: str) -> str:
    """非流式场景：去掉文本中的 DSML 标记（不关心还原出的调用）。"""
    if not text:
        return text
    stream_filter = DSMLStreamFilter()
    visible, _ = stream_filter.feed(text)
    tail, _ = stream_filter.finish()
    return visible + tail
