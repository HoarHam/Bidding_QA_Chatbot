"""招投标智能问答机器人。

`main()` 是 `pyproject.toml` 里 `[project.scripts]` 的目标，**必须保留在本模块**。
真正的实现委托给 `cli.main`，此处用函数内延迟导入，避免 `import bidding_qa_chatbot`
就把 fastapi / torch 等重依赖全部拉起来。
"""

from __future__ import annotations

__version__ = "0.1.0"


def main() -> int:
    from bidding_qa_chatbot.cli import main as _main

    return _main()
