"""支持 `uv run python -m bidding_qa_chatbot ...`。"""

from __future__ import annotations

import sys

from bidding_qa_chatbot.cli import main

if __name__ == "__main__":
    sys.exit(main())
