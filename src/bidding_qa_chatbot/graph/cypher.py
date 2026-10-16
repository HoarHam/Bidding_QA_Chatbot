"""Cypher 模板：结构化参数 → 参数化查询 + 绑定参数。

模板把**图 schema 知识**（关系名、方向、属性名 camelCase）集中在这里，
工具层只负责把 JSON 参数翻译成 `anchors` 字典。约定：

- 锚点一律是**过滤条件**：`subject_matter` 过滤 `s`，其余锚点用 `EXISTS` 过滤
  「与该实体有关系的标的物」。任何锚点组合语义一致，不会出现「多给了锚点被静默忽略」。
- **用户输入只进 `params`，绝不拼接进查询文本**（已在测试中按此断言）。
- `tradeFrequency` 可能为 null（`LOAD_TX_CYPHER` 建节点时不带该属性），一律 `coalesce(..., 0)`。
"""

from __future__ import annotations

from typing import Any

#: 支持的查询方面。`suppliers` 等 4 个实体方面的实现共用一套模板。
ASPECTS = (
    "overview",
    "subject_matters",
    "transactions",
    "suppliers",
    "purchasers",
    "agencies",
    "locations",
)

#: 锚点 → (关系类型, 目标标签)。键即工具参数名。
_ENTITY_ANCHORS: dict[str, tuple[str, str]] = {
    "supplier": ("SUPPLIED_BY", "Supplier"),
    "purchaser": ("PURCHASED_BY", "Purchaser"),
    "agency": ("AGENT_BY", "Agency"),
    "location": ("LOCATED_AT", "Location"),
}

#: 实体方面的 aspect → (关系类型, 目标标签)
_ENTITY_ASPECTS: dict[str, tuple[str, str]] = {
    "suppliers": ("SUPPLIED_BY", "Supplier"),
    "purchasers": ("PURCHASED_BY", "Purchaser"),
    "agencies": ("AGENT_BY", "Agency"),
    "locations": ("LOCATED_AT", "Location"),
}


def default_aspect(subject_matter: str | None) -> str:
    """省略 aspect 时的默认值：给了标的物锚点 → 概览；否则 → 标的物排行。"""
    return "overview" if (subject_matter or "").strip() else "subject_matters"


def _conditions(anchors: dict[str, str], params: dict[str, Any]) -> list[str]:
    """把锚点翻译成 WHERE 条件（全部为绑定参数）。"""
    conditions: list[str] = []
    subject_matter = (anchors.get("subject_matter") or "").strip().lower()
    if subject_matter:
        params["subject_matter"] = subject_matter
        conditions.append("toLower(s.name) CONTAINS $subject_matter")
    for key, (rel, label) in _ENTITY_ANCHORS.items():
        value = (anchors.get(key) or "").strip().lower()
        if not value:
            continue
        params[key] = value
        # 变量名加锚点后缀：多个 EXISTS 子查询里的模式变量互不干扰
        conditions.append(
            f"EXISTS {{ MATCH (s)-[:{rel}]->(x_{key}:{label}) "
            f"WHERE toLower(x_{key}.name) CONTAINS ${key} }}"
        )
    return conditions


def render(aspect: str, anchors: dict[str, str], limit: int) -> tuple[str, dict[str, Any]]:
    """渲染模板。返回 `(cypher, params)`；`anchors` 里只应包含非空值。

    `aspect` 必须已在 `ASPECTS` 中（工具层先行校验并给出中文错误）。
    """
    if aspect not in ASPECTS:
        raise ValueError(f"未知 aspect: {aspect}")

    params: dict[str, Any] = {"limit": limit}
    conditions = _conditions(anchors, params)
    where = "WHERE " + "\n  AND ".join(conditions) if conditions else ""

    if aspect == "subject_matters":
        match = "MATCH (s:SubjectMatter)"
        tail = (
            "RETURN s.name AS name, coalesce(s.tradeFrequency, 0) AS trade_frequency\n"
            "ORDER BY trade_frequency DESC, name\n"
            "LIMIT $limit"
        )
    elif aspect == "overview":
        match = "MATCH (s:SubjectMatter)"
        tail = (
            # avg()/max() 自动忽略 null；COUNT{} 子查询不会与 OPTIONAL MATCH 产生笛卡尔积
            "OPTIONAL MATCH (s)-[:HAS_TRANSACTION]->(t:Transaction)\n"
            "WITH s, count(t) AS tx_count, avg(t.amount) AS avg_amount, "
            "max(t.amount) AS max_amount\n"
            "RETURN s.name AS name,\n"
            "  coalesce(s.tradeFrequency, 0) AS trade_frequency,\n"
            "  tx_count,\n"
            "  COUNT { MATCH (s)-[:SUPPLIED_BY]->(:Supplier) } AS supplier_count,\n"
            "  COUNT { MATCH (s)-[:PURCHASED_BY]->(:Purchaser) } AS purchaser_count,\n"
            "  COUNT { MATCH (s)-[:AGENT_BY]->(:Agency) } AS agency_count,\n"
            "  COUNT { MATCH (s)-[:LOCATED_AT]->(:Location) } AS location_count,\n"
            "  avg_amount, max_amount\n"
            "ORDER BY trade_frequency DESC, tx_count DESC, name\n"
            "LIMIT $limit"
        )
    elif aspect == "transactions":
        match = "MATCH (s:SubjectMatter)-[:HAS_TRANSACTION]->(t:Transaction)"
        tail = (
            # 供应商为标的物级关联（参考值），不能当作本笔交易的中标人——
            # 输出层会在表格下附提示语。
            "RETURN s.name AS name, t.amount AS amount,\n"
            "  [(s)-[:SUPPLIED_BY]->(su:Supplier) | su.name][0..3] AS matter_suppliers\n"
            "ORDER BY (t.amount IS NULL), t.amount DESC\n"
            "LIMIT $limit"
        )
    else:
        rel, label = _ENTITY_ASPECTS[aspect]
        match = f"MATCH (s:SubjectMatter)-[:{rel}]->(e:{label})"
        tail = (
            "RETURN e.name AS name,\n"
            "  count(DISTINCT s) AS subject_matter_count,\n"
            "  sum(coalesce(s.tradeFrequency, 0)) AS total_trade_frequency\n"
            "ORDER BY subject_matter_count DESC, total_trade_frequency DESC, name\n"
            "LIMIT $limit"
        )

    cypher = "\n".join(part for part in (match, where, tail) if part)
    return cypher, params


def candidate_query(kind: str, value: str, limit: int = 10) -> tuple[str, dict[str, Any]]:
    """列出与 `value` 部分匹配（不区分大小写）的候选名称，用于消歧。

    `kind` ∈ `{"subject_matter"} ∪ _ENTITY_ANCHORS.keys()`。
    """
    params: dict[str, Any] = {"value": value.strip().lower(), "limit": limit}
    if kind == "subject_matter":
        return (
            "MATCH (s:SubjectMatter)\n"
            "WHERE toLower(s.name) CONTAINS $value\n"
            "RETURN DISTINCT s.name AS name, coalesce(s.tradeFrequency, 0) AS trade_frequency\n"
            "ORDER BY trade_frequency DESC, name\n"
            "LIMIT $limit",
            params,
        )
    rel, label = _ENTITY_ANCHORS[kind]
    return (
        f"MATCH (s:SubjectMatter)-[:{rel}]->(e:{label})\n"
        "WHERE toLower(e.name) CONTAINS $value\n"
        "RETURN e.name AS name, count(DISTINCT s) AS subject_matter_count\n"
        "ORDER BY subject_matter_count DESC, name\n"
        "LIMIT $limit",
        params,
    )
