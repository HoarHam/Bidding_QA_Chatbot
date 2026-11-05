"""Agent 综合评测（Ragas）：离线数据集 + 在线真实对话。

- `dataset.py`：加载 `evaluate_cases.json`、工具名归一化、工具选择判定；
- `ragas_runner.py`：判分 LLM / 本地嵌入 / Ragas 指标适配；
- `runner.py`：后台运行、进度、指标汇总、结果持久化；
- `store.py`：`data/evaluations/*.json` 读写；
- `page.py`：8001 端口上的 `/evaluate` 单页面板。
"""

from bidding_qa_chatbot.evaluation.runner import EvaluationService

__all__ = ["EvaluationService"]
