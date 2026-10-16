"""GraphQueryTool：工具契约、严格签名、消歧、防护、格式化与 artifacts。

全部离线：FakeGraphService 按 `CONTAINS $value`（消歧候选查询的特征）路由返回值，
并记录每次 `(cypher, params, max_rows)` 调用，供断言「主查询是否被跳过」等行为。
"""

import inspect

import pytest

from bidding_qa_chatbot.agent.base import ToolRegistry
from bidding_qa_chatbot.agent.graph_tool import MAX_LIMIT, GraphQueryTool
from bidding_qa_chatbot.graph import GraphQueryError
from bidding_qa_chatbot.graph.service import GraphQueryResult


class FakeGraphService:
    def __init__(
        self, *, available=True, candidate_rows=None, rows=None, error=None, truncated=False
    ):
        self.available = available
        self.candidate_rows = list(candidate_rows or [])
        self.rows = list(rows or [])
        self.error = error
        self.truncated = truncated
        self.calls: list[tuple[str, dict, int | None]] = []

    async def run_cypher(self, cypher, params=None, *, max_rows=None):
        self.calls.append((cypher, dict(params or {}), max_rows))
        if self.error is not None:
            raise self.error
        is_candidate = "CONTAINS $value" in cypher
        rows = self.candidate_rows if is_candidate else self.rows
        return GraphQueryResult(rows=list(rows), truncated=self.truncated, elapsed_ms=1)


def _tool(service) -> GraphQueryTool:
    return GraphQueryTool(service)


# ---------------- 工具声明 ----------------


def test_spec_is_well_formed():
    spec = GraphQueryTool.spec
    assert spec.name == "query_bidding_graph"
    assert spec.parameters["required"] == []
    assert set(spec.parameters["properties"]["aspect"]["enum"]) == {
        "overview",
        "subject_matters",
        "transactions",
        "suppliers",
        "purchasers",
        "agencies",
        "locations",
    }
    assert "必须" in spec.description
    assert "search_bidding_knowledge" in spec.description
    assert "标的物级" in spec.description  # 数据边界声明，防止模型过度断言


def test_schema_matches_strict_signature():
    """JSON Schema 的属性集必须与 run() 的具名参数集一致，且不允许 **kwargs。"""
    properties = set(GraphQueryTool.spec.parameters["properties"])
    signature = inspect.signature(GraphQueryTool.run)
    parameters = [name for name in signature.parameters if name != "self"]
    kinds = {p.kind for p in signature.parameters.values()}
    assert inspect.Parameter.VAR_KEYWORD not in kinds, (
        "不允许 **_ 兜底，写错参数名要报 bad_arguments"
    )
    assert properties == set(parameters)


async def test_unknown_argument_becomes_bad_arguments_via_registry():
    registry = ToolRegistry()
    registry.register(_tool(FakeGraphService()))
    result = await registry.execute("query_bidding_graph", {"q": "空调"})
    assert result.ok is False
    assert result.error == "bad_arguments"


# ---------------- 不可用 / 参数冲突 / 校验 ----------------


async def test_unavailable_service():
    tool = _tool(None)
    result = await tool.run(subject_matter="空调")
    assert result.ok is False
    assert result.error == "graph_unavailable"
    assert "不要" in result.content


async def test_conflicting_arguments():
    tool = _tool(FakeGraphService())
    result = await tool.run(subject_matter="空调", cypher="MATCH (n) RETURN n")
    assert result.ok is False
    assert result.error == "conflicting_arguments"


async def test_invalid_cypher_rejected_without_touching_service():
    service = FakeGraphService()
    result = await _tool(service).run(cypher="MATCH (n) DETACH DELETE n")
    assert result.ok is False
    assert result.error == "invalid_cypher"
    assert service.calls == [], "被 guard 拒绝的查询绝不能发到服务端"


async def test_unknown_aspect():
    service = FakeGraphService()
    result = await _tool(service).run(subject_matter="空调", aspect="drop")
    assert result.ok is False
    assert result.error == "unknown_aspect"
    assert service.calls == []


@pytest.mark.parametrize("bad_limit", ["abc", {"a": 1}])
async def test_bad_limit(bad_limit):
    result = await _tool(FakeGraphService()).run(subject_matter="空调", limit=bad_limit)
    assert result.ok is False
    assert result.error == "bad_arguments"


# ---------------- 消歧 ----------------


async def test_disambiguation_stops_before_main_query():
    service = FakeGraphService(
        candidate_rows=[
            {"name": "合肥合燃华润燃气有限公司", "subject_matter_count": 12},
            {"name": "合肥快乐通冷链物流配送有限公司", "subject_matter_count": 9},
        ]
    )
    result = await _tool(service).run(supplier="合肥")

    assert result.ok is True
    assert "消歧" in result.content and "确认" in result.content
    assert "合肥合燃华润燃气有限公司" in result.content
    assert result.artifacts["summary"] == "匹配到 2 个候选，等待用户确认"
    assert "未" not in result.artifacts["summary"]
    # 只跑了消歧候选查询，主查询被跳过
    assert len(service.calls) == 1
    assert "CONTAINS $value" in service.calls[0][0]


async def test_exact_match_proceeds_despite_multiple_candidates():
    service = FakeGraphService(
        candidate_rows=[
            {"name": "空调", "trade_frequency": 179},
            {"name": "空调配件", "trade_frequency": 21},
        ],
        rows=[{"name": "空调", "trade_frequency": 179}],
    )
    result = await _tool(service).run(subject_matter="空调")

    assert result.ok is True
    assert len(service.calls) == 2  # 候选 + 主查询
    assert "CONTAINS $value" not in service.calls[1][0]


async def test_single_candidate_proceeds_and_notes_resolution():
    service = FakeGraphService(
        candidate_rows=[{"name": "中央空调", "trade_frequency": 30}],
        rows=[{"name": "中央空调", "trade_frequency": 30}],
    )
    result = await _tool(service).run(subject_matter="空调")

    assert result.ok is True
    assert "解析为「中央空调」" in result.content


async def test_no_match_is_ok_true_with_anti_fabrication_message():
    service = FakeGraphService(candidate_rows=[])
    result = await _tool(service).run(subject_matter="不存在的标的物")

    assert result.ok is True
    assert result.error is None
    assert "没有查到" in result.content
    assert "不要" in result.content
    assert result.artifacts["summary"] == "知识图谱中未查到匹配数据"


# ---------------- 模板路径与格式化 ----------------


async def test_template_query_uses_parameters_and_labels():
    service = FakeGraphService(
        candidate_rows=[{"name": "空调", "trade_frequency": 179}],
        rows=[
            {
                "name": "铜陵鸿腾商贸有限公司",
                "subject_matter_count": 3,
                "total_trade_frequency": 204,
            }
        ],
    )
    result = await _tool(service).run(subject_matter="空调", aspect="suppliers", limit=5)

    cypher, params, max_rows = service.calls[-1]
    assert "MATCH (s:SubjectMatter)-[:SUPPLIED_BY]->(e:Supplier)" in cypher
    assert params["subject_matter"] == "空调"
    assert params["limit"] == 5
    assert max_rows == 5
    assert "| 供应商 | 关联标的物数 | 关联标的物频次合计 |" in result.content
    assert "铜陵鸿腾商贸有限公司" in result.content
    assert result.artifacts["summary"] == "知识图谱查询到 1 条结果"
    assert "sources" not in result.artifacts, (
        "图谱结果绝不能进 artifacts['sources']（SourceRef 校验）"
    )
    assert result.artifacts["graph"]["aspect"] == "suppliers"


async def test_transactions_output_has_caveat_and_supplier_column():
    service = FakeGraphService(
        candidate_rows=[{"name": "中央空调", "trade_frequency": 30}],
        rows=[{"name": "中央空调", "amount": 887900, "matter_suppliers": ["甲公司", "乙公司"]}],
    )
    result = await _tool(service).run(subject_matter="中央空调", aspect="transactions")

    assert "该标的物关联供应商(参考)" in result.content
    assert "不代表该笔交易的中标人" in result.content
    assert "887,900" in result.content
    assert "甲公司、乙公司" in result.content


async def test_truncation_notice():
    service = FakeGraphService(
        candidate_rows=[{"name": "x", "trade_frequency": 1}],
        rows=[{"name": "x", "trade_frequency": 1}],
        truncated=True,
    )
    result = await _tool(service).run(subject_matter="x", limit=3)

    assert "仅显示前 3 行" in result.content
    assert f"上限 {MAX_LIMIT}" in result.content


async def test_empty_rows_from_template():
    service = FakeGraphService(candidate_rows=[{"name": "空调", "trade_frequency": 179}], rows=[])
    result = await _tool(service).run(subject_matter="空调")

    assert result.ok is True
    assert "没有返回结果" in result.content
    assert result.artifacts["summary"] == "知识图谱未查到数据"


@pytest.mark.parametrize(("given", "expected"), [(0, 1), (999, MAX_LIMIT), (None, 20)])
async def test_limit_clamping(given, expected):
    service = FakeGraphService(
        candidate_rows=[{"name": "x", "trade_frequency": 1}],
        rows=[{"name": "x", "trade_frequency": 1}],
    )
    await _tool(service).run(subject_matter="x", limit=given)
    assert service.calls[-1][2] == expected


# ---------------- cypher 兜底路径 ----------------


async def test_cypher_path_success():
    service = FakeGraphService(rows=[{"name": "空调"}])
    result = await _tool(service).run(
        cypher="MATCH (s:SubjectMatter) RETURN s.name AS name LIMIT 3"
    )

    cypher, params, max_rows = service.calls[-1]
    assert "MATCH (s:SubjectMatter)" in cypher
    assert params == {}
    assert max_rows == 20
    assert "只读 Cypher" in result.content
    assert result.artifacts["graph"]["mode"] == "cypher"


async def test_cypher_params_passthrough():
    service = FakeGraphService(rows=[{"name": "空调"}])
    await _tool(service).run(
        cypher="MATCH (s) WHERE s.name CONTAINS $kw RETURN s.name AS name",
        cypher_params={"kw": "空调"},
    )
    assert service.calls[-1][1] == {"kw": "空调"}


async def test_cypher_params_must_be_object():
    result = await _tool(FakeGraphService()).run(
        cypher="MATCH (n) RETURN n", cypher_params="kw=空调"
    )
    assert result.ok is False
    assert result.error == "bad_arguments"


async def test_cypher_params_too_many_keys():
    params = {f"p{i}": i for i in range(21)}
    result = await _tool(FakeGraphService()).run(cypher="MATCH (n) RETURN n", cypher_params=params)
    assert result.ok is False
    assert result.error == "bad_arguments"


# ---------------- 服务层错误映射 ----------------


async def test_graph_query_error_becomes_tool_error():
    service = FakeGraphService(
        error=GraphQueryError("graph_timeout", "知识图谱查询超时（超过 8 秒）")
    )
    result = await _tool(service).run(cypher="MATCH (n) RETURN n")

    assert result.ok is False
    assert result.error == "graph_timeout"
    assert "超时" in result.content
