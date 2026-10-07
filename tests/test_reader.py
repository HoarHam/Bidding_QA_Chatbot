"""问答表读取测试。

语料 `data/` 已被 `.gitignore` 忽略，CI 里不存在，因此：
- 真实语料的断言挂在 `requires_corpus` 上（本地跑，CI 跳过）；
- 结构类断言用 openpyxl 现造一个临时 xlsx（纯文本，不需要 Pillow）。
"""

from __future__ import annotations

import openpyxl
import pytest

from bidding_qa_chatbot.ingestion.reader import read_qa_rows
from tests.conftest import CORPUS_XLSX, requires_corpus


def _make_xlsx(path, rows, header=("问", "答")):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(list(header))
    for q, a in rows:
        ws.append([q, a])
    wb.save(path)
    return path


def test_reads_rows_and_maps_excel_rows(tmp_path):
    xlsx = _make_xlsx(
        tmp_path / "t.xlsx",
        [("问题一", "答案一"), ("问题二", "答案二")],
    )
    rows = read_qa_rows(xlsx)

    assert len(rows) == 2
    # 表头占第 1 行，故第一条数据在第 2 行
    assert [r.excel_row for r in rows] == [2, 3]
    assert [r.data_index for r in rows] == [0, 1]
    assert rows[0].question == "问题一"


def test_drops_trailing_all_empty_row(tmp_path):
    """真实语料末尾有一个全空行，若不丢弃会多出 1 个空分块。"""
    xlsx = tmp_path / "t.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["问", "答"])
    ws.append(["问题", "答案"])
    ws.append([None, None])  # 模拟末尾空行
    wb.save(xlsx)

    rows = read_qa_rows(xlsx)
    assert len(rows) == 1
    assert rows[0].question == "问题"


def test_drops_rows_missing_answer(tmp_path):
    xlsx = _make_xlsx(tmp_path / "t.xlsx", [("有问题", None), ("完整", "完整答案")])
    rows = read_qa_rows(xlsx)
    assert len(rows) == 1
    assert rows[0].question == "完整"


def test_strips_whitespace(tmp_path):
    xlsx = _make_xlsx(tmp_path / "t.xlsx", [("  前后有空格  ", "  答案  ")])
    rows = read_qa_rows(xlsx)
    assert rows[0].question == "前后有空格"
    assert rows[0].answer == "答案"


def test_missing_column_raises(tmp_path):
    xlsx = _make_xlsx(tmp_path / "t.xlsx", [("a", "b")], header=("题目", "答案"))
    with pytest.raises(ValueError, match="缺少必需列"):
        read_qa_rows(xlsx)


@requires_corpus
def test_real_corpus_shape():
    rows = read_qa_rows(CORPUS_XLSX)
    assert len(rows) == 80
    assert rows[0].excel_row == 2
    assert rows[-1].excel_row == 81
    assert all(r.question and r.answer for r in rows)


@requires_corpus
def test_row_52_is_the_image_only_answer():
    """第 52 行的正文只是一句引导语，真正的答案在 3 张图里。

    这条断言锁住的是「视觉解析为什么是必需的」这个事实：
    如果哪天语料变了，这里会第一时间失败。
    """
    rows = read_qa_rows(CORPUS_XLSX)
    row52 = next(r for r in rows if r.excel_row == 52)
    assert row52.data_index == 50
    assert len(row52.answer) < 40
    assert "招采情况如下" in row52.answer
