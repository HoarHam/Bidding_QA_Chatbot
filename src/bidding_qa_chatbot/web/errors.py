"""联网搜索的错误类型（工具层会把它转成可转述的 ToolResult）。"""

from __future__ import annotations


class WebSearchError(RuntimeError):
    """搜索失败。消息已脱敏，可直接回灌给模型。`code` 供前端错误标签使用。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
