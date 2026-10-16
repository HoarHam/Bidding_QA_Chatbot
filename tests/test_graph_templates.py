"""Cypher 模板（graph/cypher.py）。

两条硬性断言贯穿全文：
1. **锚点值只进 params，绝不出现于查询文本**（防注入的结构性保证）；
2. 关系类型/方向与 data/batch/to_neo4j.py 里的图模型一致。
"""

import pytest

from bidding_qa_chatbot.graph.cypher import (
    ASPECTS,
    candidate_query,
    default_aspect,
    render,
)

#: 带引号/关键字/注释符的“恶意”值：只要它出现在查询文本里就说明发生了拼接
EVIL = "x' DELETE n // /* DROP"

ALL_ANCHORS = {
    "subject_matter": EVIL,
    "supplier": EVIL,
    "purchaser": EVIL,
    "agency": EVIL,
    "location": EVIL,
}


@pytest.mark.parametrize("aspect", ASPECTS)
@pytest.mark.parametrize("anchors", [{}, ALL_ANCHORS])
def test_values_never_interpolated(aspect, anchors):
    cypher, params = render(aspect, anchors, limit=7)
    assert EVIL not in cypher
    assert "'" not in cypher, "查询文本里不应出现任何单引号字面量"
    assert params["limit"] == 7
    if anchors:
        for key, value in anchors.items():
            assert params[key] == value.lower()
            assert f"${key}" in cypher


@pytest.mark.parametrize("aspect", ASPECTS)
def test_unknown_anchor_combinations_are_still_safe(aspect):
    # 多锚点组合语义一致：都变成 s 上的 EXISTS 过滤条件
    cypher, params = render(aspect, ALL_ANCHORS, limit=5)
    for key in ("supplier", "purchaser", "agency", "location"):
        assert f"${key}" in cypher
    assert "  AND " in cypher


def test_default_aspect_branches():
    assert default_aspect("空调") == "overview"
    assert default_aspect("  ") == "subject_matters"
    assert default_aspect(None) == "subject_matters"


def test_unknown_aspect_raises():
    with pytest.raises(ValueError, match="未知 aspect"):
        render("drop_everything", {}, limit=5)


@pytest.mark.parametrize("aspect", ["subject_matters", "overview"])
def test_subject_anchored_templates(aspect):
    cypher, _ = render(aspect, {"subject_matter": "空调"}, limit=5)
    assert cypher.startswith("MATCH (s:SubjectMatter)")
    assert "toLower(s.name) CONTAINS $subject_matter" in cypher
    assert "coalesce(s.tradeFrequency, 0) AS trade_frequency" in cypher
    assert "ORDER BY trade_frequency DESC" in cypher
    assert "LIMIT $limit" in cypher


def test_overview_counts_and_amounts():
    cypher, _ = render("overview", {}, limit=5)
    assert "OPTIONAL MATCH (s)-[:HAS_TRANSACTION]->(t:Transaction)" in cypher
    assert "count(t) AS tx_count" in cypher
    for rel in ("SUPPLIED_BY", "PURCHASED_BY", "AGENT_BY", "LOCATED_AT"):
        assert f"COUNT {{ MATCH (s)-[:{rel}]" in cypher
    assert "avg(t.amount) AS avg_amount" in cypher
    assert "max(t.amount) AS max_amount" in cypher


def test_transactions_template():
    cypher, _ = render("transactions", {"subject_matter": "空调"}, limit=5)
    assert "MATCH (s:SubjectMatter)-[:HAS_TRANSACTION]->(t:Transaction)" in cypher
    assert "[(s)-[:SUPPLIED_BY]->(su:Supplier) | su.name][0..3] AS matter_suppliers" in cypher
    # 空金额排最后
    assert "ORDER BY (t.amount IS NULL), t.amount DESC" in cypher


@pytest.mark.parametrize(
    ("aspect", "rel", "label"),
    [
        ("suppliers", "SUPPLIED_BY", "Supplier"),
        ("purchasers", "PURCHASED_BY", "Purchaser"),
        ("agencies", "AGENT_BY", "Agency"),
        ("locations", "LOCATED_AT", "Location"),
    ],
)
def test_entity_aspects(aspect, rel, label):
    cypher, _ = render(aspect, {}, limit=5)
    assert f"MATCH (s:SubjectMatter)-[:{rel}]->(e:{label})" in cypher
    assert "count(DISTINCT s) AS subject_matter_count" in cypher
    assert "sum(coalesce(s.tradeFrequency, 0)) AS total_trade_frequency" in cypher


def test_entity_anchor_becomes_exists_filter():
    cypher, params = render("subject_matters", {"supplier": "合肥"}, limit=5)
    assert "EXISTS { MATCH (s)-[:SUPPLIED_BY]->(x_supplier:Supplier)" in cypher
    assert params["supplier"] == "合肥"


def test_no_anchors_no_where():
    cypher, _ = render("subject_matters", {}, limit=5)
    assert "WHERE" not in cypher


def test_candidate_query_subject_matter():
    cypher, params = candidate_query("subject_matter", " 空调 ")
    assert "MATCH (s:SubjectMatter)" in cypher
    assert "toLower(s.name) CONTAINS $value" in cypher
    assert params["value"] == "空调"  # strip + lower 已归一


@pytest.mark.parametrize(
    ("kind", "rel", "label"),
    [
        ("supplier", "SUPPLIED_BY", "Supplier"),
        ("purchaser", "PURCHASED_BY", "Purchaser"),
        ("agency", "AGENT_BY", "Agency"),
        ("location", "LOCATED_AT", "Location"),
    ],
)
def test_candidate_query_entity_kinds(kind, rel, label):
    cypher, params = candidate_query(kind, "合肥")
    assert f"MATCH (s:SubjectMatter)-[:{rel}]->(e:{label})" in cypher
    assert "count(DISTINCT s) AS subject_matter_count" in cypher
    assert params["value"] == "合肥"
