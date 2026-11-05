"""离线评测数据集：加载与工具选择归一化。

数据集里的 `expected_tool` 是早期设计稿的名字（`search_knowledge_graph` /
`search_postgresql` / `ask_clarification`），而当前实际工具叫
`query_bidding_graph`，也没有 SQL 工具。这里做显式别名映射，避免评测结果
因为名字不同而全部判错；`ask_clarification` 表示「不该调用任何工具」。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: 数据集里的旧名字 → 当前实际工具名。None 表示期望不调用工具。
_EXPECTED_TOOL_ALIASES: dict[str, str | None] = {
    "search_bidding_knowledge": "search_bidding_knowledge",
    "search_knowledge_graph": "query_bidding_graph",
    "query_bidding_graph": "query_bidding_graph",
    # 早期设计里有 PostgreSQL 聚合工具；现在由知识图谱承担同类结构化查询。
    "search_postgresql": "query_bidding_graph",
    "ask_clarification": None,
}

#: 当前 Agent 的联网工具。离线评测默认关闭，避免外部搜索让指标不可复现。
WEB_TOOLS = {"search_web_keyword", "search_web_semantic"}


class EvaluationDatasetError(ValueError):
    """数据集文件不存在 / 结构不合法。"""


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    index: int
    question: str
    reference: str
    #: 归一化后的实际工具名；`expected_no_tool=True` 时为 None。
    expected_tool: str | None
    #: 数据集原始标签，仅用于展示与排查。
    original_expected_tool: str
    #: ask_clarification：期望模型向用户澄清，不调用工具。
    expected_no_tool: bool = False


def normalize_expected_tool(raw: str) -> tuple[str | None, bool]:
    """返回 `(实际工具名, 是否期望不调用工具)`。"""
    value = raw.strip()
    if value in _EXPECTED_TOOL_ALIASES:
        mapped = _EXPECTED_TOOL_ALIASES[value]
        return mapped, mapped is None
    # 未知标签原样保留：如果模型刚好调用了同名工具，仍应算对；
    # 否则它会自然判错，不会因为拼写不认识而被静默忽略。
    return value or None, False


def load_cases(path: Path) -> list[EvaluationCase]:
    if not path.exists():
        raise EvaluationDatasetError(f"评测数据集不存在: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationDatasetError(f"读取评测数据集失败: {path} ({exc})") from exc
    if not isinstance(raw, list) or not raw:
        raise EvaluationDatasetError("评测数据集必须是非空 JSON 数组")

    cases: list[EvaluationCase] = []
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise EvaluationDatasetError(f"第 {index + 1} 条用例不是对象")
        question = str(row.get("question") or "").strip()
        if not question:
            raise EvaluationDatasetError(f"第 {index + 1} 条用例缺少 question")
        expected_raw = str(row.get("expected_tool") or "").strip()
        expected_tool, expected_no_tool = normalize_expected_tool(expected_raw)
        cases.append(
            EvaluationCase(
                index=index,
                question=question,
                reference=str(row.get("reference") or "").strip(),
                expected_tool=expected_tool,
                original_expected_tool=expected_raw,
                expected_no_tool=expected_no_tool,
            )
        )
    return cases


def tool_selection_matches(case: EvaluationCase, actual_tools: list[str]) -> bool:
    """工具选择是否正确。

    - `ask_clarification`：没有调用任何工具才算对；
    - 其他：实际调用序列中出现了期望工具即算对（允许多查一次补充资料）。
    """
    if case.expected_no_tool:
        return not actual_tools
    if case.expected_tool is None:
        return False
    return case.expected_tool in actual_tools
