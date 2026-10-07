"""读取问答 Excel。

语料结构：单 sheet `Sheet1`，两列 `问` / `答`，80 行有效数据。
表格实际有 82 行——表头 1 行 + 数据 80 行 + 末尾 1 行全空，必须显式丢弃，
否则会多出 1 个内容为空的分块。

保留 pandas 原始索引到 Excel 行号的映射（`excel_row = idx + 2`），
它是「参考来源」里 `第N行` 标签的来源，也是图片锚点定位的基准。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

QUESTION_COL = "问"
ANSWER_COL = "答"
#: 表头占第 1 行，故数据行号 = 0 基索引 + 2
_HEADER_ROWS = 1


@dataclass(frozen=True, slots=True)
class QARow:
    """一条问答，`data_index` 是 0 基序号，`excel_row` 是给用户看的 1 基行号。"""

    data_index: int
    excel_row: int
    question: str
    answer: str


def read_qa_rows(xlsx_path: Path, sheet_name: str = "Sheet1") -> list[QARow]:
    """读取并清洗问答对。

    `dtype=str` 是刻意的：否则 pandas 会把纯数字的答案推断成数值类型，
    再转字符串时可能带上 `.0` 后缀，污染入库文本。
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, dtype=str)

    missing = {QUESTION_COL, ANSWER_COL} - set(df.columns)
    if missing:
        raise ValueError(
            f"{xlsx_path.name} 缺少必需列: {sorted(missing)}；实际列为 {list(df.columns)}"
        )

    # 先丢全空行（末尾那行），再丢关键列缺失的行。
    df = df.dropna(how="all").dropna(subset=[QUESTION_COL, ANSWER_COL])

    rows: list[QARow] = []
    for idx, record in df.iterrows():
        question = str(record[QUESTION_COL]).strip()
        answer = str(record[ANSWER_COL]).strip()
        if not question or not answer:
            continue
        rows.append(
            QARow(
                data_index=len(rows),
                excel_row=int(idx) + _HEADER_ROWS + 1,  # type: ignore[arg-type]
                question=question,
                answer=answer,
            )
        )
    return rows
