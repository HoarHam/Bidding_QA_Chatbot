"""把 Neo4j 知识图谱查询注册成工具。

设计要点（详见各自注释）：

- `description` 是模型选工具/路由的**唯一依据**，它必须写清「什么时候用本工具、
  什么时候用 search_bidding_knowledge」。图谱里的供应商/采购人等是**标的物级**关联
  （不是逐笔交易级），这一点必须告知模型，否则它会过度断言「某笔交易是谁中标的」。
- 参数用**严格签名**（不带 `**_` 兜底）：9 个参数的出错面大，模型写错参数名时
  应触发注册表的 `bad_arguments` 分支得到可重试的报错，而不是被静默吞掉。
- 结构化参数 → 参数化模板（`graph/cypher.py`）；表达不了的复杂查询走只读 `cypher`
  兜底（`graph/guard.py` 先行校验，服务端读事务兜底）。
- 名称模糊匹配先消歧：精确匹配或唯一候选才继续，多候选时停止并请模型向用户确认。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from bidding_qa_chatbot.agent.base import ToolResult, ToolSpec
from bidding_qa_chatbot.config import Settings, get_settings
from bidding_qa_chatbot.graph import GraphQueryError, GraphService
from bidding_qa_chatbot.graph.cypher import ASPECTS, candidate_query, default_aspect, render
from bidding_qa_chatbot.graph.guard import validate_read_only

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 20
MAX_LIMIT = 50
CANDIDATE_LIMIT = 10  # 消歧时最多列出的候选数
MAX_CYPHER_PARAMS = 20  # 兜底查询的绑定参数数量上限

_ANCHOR_KEYS = ("subject_matter", "supplier", "purchaser", "agency", "location")
_ANCHOR_LABELS = {
    "subject_matter": "标的物",
    "supplier": "供应商",
    "purchaser": "采购人",
    "agency": "代理机构",
    "location": "所在地",
}

_ASPECT_LABELS = {
    "overview": "标的物概览",
    "subject_matters": "标的物清单（按交易频次降序）",
    "transactions": "交易明细（按中标金额降序）",
    "suppliers": "关联供应商",
    "purchasers": "关联采购人",
    "agencies": "关联代理机构",
    "locations": "关联所在地",
}

#: 列名映射（模板里的 snake_case 别名 → 给模型看的中文）。`name` 随 aspect 取义，
#: 在 `_labels_for` 里单独处理。
_COLUMN_LABELS = {
    "trade_frequency": "交易频次",
    "tx_count": "交易笔数",
    "supplier_count": "供应商数",
    "purchaser_count": "采购人数",
    "agency_count": "代理机构数",
    "location_count": "所在地数",
    "avg_amount": "平均中标金额",
    "max_amount": "最高中标金额",
    "amount": "中标金额",
    "matter_suppliers": "该标的物关联供应商(参考)",
    "subject_matter_count": "关联标的物数",
    "total_trade_frequency": "关联标的物频次合计",
}

_NAME_LABELS = {
    "overview": "标的物",
    "subject_matters": "标的物",
    "transactions": "标的物",
    "suppliers": "供应商",
    "purchasers": "采购人",
    "agencies": "代理机构",
    "locations": "所在地",
}


def _as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _as_int(value: Any) -> int | None:
    """把模型可能给成字符串的数字转成 int；非法返回 None。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fmt_cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, (list, tuple)):
        items = [str(item) for item in value[:3]]
        text = "、".join(items) + ("…" if len(value) > 3 else "")
    else:
        text = str(value)[:60]
    return text.replace("|", "丨")  # 防 Markdown 表格被竖线撑破


def _labels_for(aspect: str | None) -> dict[str, str]:
    labels = dict(_COLUMN_LABELS)
    labels["name"] = _NAME_LABELS.get(aspect or "", "名称")
    return labels


def _format_table(rows: list[dict[str, Any]], aspect: str | None) -> str:
    labels = _labels_for(aspect)
    columns = list(rows[0].keys())  # record.data() 保持 Cypher RETURN 的列序
    header = "| " + " | ".join(labels.get(col, col) for col in columns) + " |"
    separator = "|" + "---|" * len(columns)
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(_fmt_cell(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


class GraphQueryTool:
    """查询 Neo4j 知识图谱（标的物—交易关系）。"""

    spec = ToolSpec(
        name="query_bidding_graph",
        description=(
            "查询招投标知识图谱（Neo4j），回答关于标的物与交易事实的结构化数据问题。\n"
            "图谱内容：标的物（如「空调」「打印/复印纸」「网上超市采购」）及其交易频次；\n"
            "标的物关联的采购人、代理机构、中标供应商、所在地；每笔交易的中标金额。\n"
            "\n"
            "凡涉及以下内容，必须先调用本工具，不要凭记忆或估算回答：\n"
            "- 某标的物/品类的交易频次、交易笔数、热度排名；\n"
            "- 某供应商/采购人/代理机构/地区与哪些标的物有关联；\n"
            "- 某标的物的中标金额及关联的供应商、采购人、代理机构、所在地。\n"
            "\n"
            "不要用于：法律法规条款、期限时限、金额门槛、资格条件、评审办法等规范性知识"
            "——那类问题用 search_bidding_knowledge。\n"
            "\n"
            "用法：优先用结构化参数（subject_matter / supplier / purchaser / agency / "
            "location 搭配 aspect）；只有结构化参数表达不了时才用 cypher 写只读查询"
            "（单条语句、禁止写操作）。\n"
            "\n"
            "重要限制：供应商/采购人/代理机构/所在地是**标的物级**关联，不是逐笔交易级；"
            "同一标的物下的不同交易可能由不同供应商中标，图谱无法区分。\n"
            "\n"
            "结果为空时如实说明「知识图谱中没有查到相关数据」，不要编造。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "subject_matter": {
                    "type": "string",
                    "description": "标的物名称，支持部分匹配，如「空调」「打印/复印纸」「网上超市采购」。",
                },
                "supplier": {
                    "type": "string",
                    "description": "中标供应商名称，支持部分匹配。",
                },
                "purchaser": {
                    "type": "string",
                    "description": "采购人名称，支持部分匹配。",
                },
                "agency": {
                    "type": "string",
                    "description": "代理机构名称，支持部分匹配。",
                },
                "location": {
                    "type": "string",
                    "description": "所在地，支持部分匹配，如「合肥市」。",
                },
                "aspect": {
                    "type": "string",
                    "enum": list(ASPECTS),
                    "description": (
                        "要查什么。overview=标的物概览（频次+交易笔数+各关联实体数+金额统计）；"
                        "subject_matters=标的物清单及交易频次（问「某供应商/采购人/地区关联哪些标的物」"
                        "或「哪些标的物交易最多」时用它）；transactions=交易明细（按中标金额降序）；"
                        "suppliers/purchasers/agencies/locations=关联的供应商/采购人/代理机构/所在地。"
                        "省略时：给了 subject_matter 默认 overview，否则默认 subject_matters。"
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_LIMIT,
                    "description": f"返回行数上限，默认 {DEFAULT_LIMIT}。",
                },
                "cypher": {
                    "type": "string",
                    "description": (
                        "只读 Cypher 兜底，仅在结构化参数无法表达时使用。"
                        "禁止 CREATE/MERGE/SET/DELETE 等写操作与 CALL/LOAD 等过程调用；"
                        "只允许单条语句；结果会被截断。"
                        "请对返回的属性用 AS 起别名（不要返回整个节点，会很冗长）。"
                    ),
                },
                "cypher_params": {
                    "type": "object",
                    "description": "cypher 中用到的 $参数（键值对）。参数是绑定值，不是字符串拼接。",
                },
            },
            "required": [],
        },
    )

    def __init__(self, service: GraphService | None, settings: Settings | None = None) -> None:
        self.service = service
        self.settings = settings or get_settings()

    async def run(
        self,
        subject_matter: str | None = None,
        supplier: str | None = None,
        purchaser: str | None = None,
        agency: str | None = None,
        location: str | None = None,
        aspect: str | None = None,
        limit: int | None = None,
        cypher: str | None = None,
        cypher_params: dict[str, Any] | None = None,
    ) -> ToolResult:
        started = time.perf_counter()

        if self.service is None or not self.service.available:
            return ToolResult(
                name=self.spec.name,
                ok=False,
                error="graph_unavailable",
                content=(
                    "知识图谱查询当前不可用（Neo4j 未接入或连接失败）。"
                    "请如实告知用户知识图谱暂时无法查询，"
                    "不要凭自身记忆编造交易频次、中标金额或供应商等信息。"
                ),
            )

        raw_anchors = {
            "subject_matter": subject_matter,
            "supplier": supplier,
            "purchaser": purchaser,
            "agency": agency,
            "location": location,
        }
        anchors = {key: text for key, value in raw_anchors.items() if (text := _as_text(value))}

        cypher_text = _as_text(cypher)
        if cypher_text and anchors:
            return self._fail_with(
                "conflicting_arguments",
                "参数冲突：不能同时使用 cypher 与结构化参数（subject_matter/supplier/...）。"
                "请二选一：常规查询用结构化参数，复杂查询只用 cypher。",
            )

        limit_value = _as_int(limit)
        if limit is not None and limit_value is None:
            return self._fail_with(
                "bad_arguments", f"limit 必须是 1-{MAX_LIMIT} 的整数，收到：{limit!r}。"
            )
        effective_limit = DEFAULT_LIMIT if limit_value is None else limit_value
        cap = min(max(1, effective_limit), MAX_LIMIT)

        if cypher_text:
            return await self._run_cypher(cypher_text, cypher_params, cap, started)
        return await self._run_template(anchors, aspect, cap, started)

    # ---------------- 兜底路径 ----------------

    async def _run_cypher(
        self,
        cypher: str,
        cypher_params: dict[str, Any] | None,
        cap: int,
        started: float,
    ) -> ToolResult:
        ok, reason = validate_read_only(cypher)
        if not ok:
            return self._fail_with(
                "invalid_cypher",
                f"查询被拒绝：{reason}请改写为单条只读查询后重试（仅限 MATCH/RETURN 类语句）。",
                elapsed=started,
            )

        if cypher_params is None or cypher_params == {}:
            params: dict[str, Any] = {}
        elif isinstance(cypher_params, dict):
            if len(cypher_params) > MAX_CYPHER_PARAMS:
                return self._fail_with(
                    "bad_arguments",
                    f"cypher_params 最多 {MAX_CYPHER_PARAMS} 个键。",
                    elapsed=started,
                )
            params = dict(cypher_params)
        else:
            return self._fail_with(
                "bad_arguments", "cypher_params 必须是键值对对象（JSON object）。", elapsed=started
            )

        return await self._execute(
            cypher, params, mode="cypher", aspect=None, anchors={}, cap=cap, started=started
        )

    # ---------------- 模板路径 ----------------

    async def _run_template(
        self, anchors: dict[str, str], aspect: str | None, cap: int, started: float
    ) -> ToolResult:
        aspect_value = _as_text(aspect) or default_aspect(anchors.get("subject_matter"))
        if aspect_value not in ASPECTS:
            return self._fail_with(
                "unknown_aspect",
                f"未知 aspect：{aspect_value!r}。可选值：{'、'.join(ASPECTS)}。",
                elapsed=started,
            )

        # 逐个锚点消歧：精确匹配或唯一候选 → 继续；多候选 → 停下请模型向用户确认
        matched: dict[str, str] = {}
        for key, value in anchors.items():
            candidates_cypher, candidates_params = candidate_query(key, value)
            candidates = await self.service.run_cypher(
                candidates_cypher, candidates_params, max_rows=CANDIDATE_LIMIT
            )
            names = [row.get("name") for row in candidates.rows if row.get("name")]
            if not names:
                return ToolResult(
                    name=self.spec.name,
                    ok=True,
                    content=(
                        f"知识图谱中没有查到与{_ANCHOR_LABELS[key]}「{value}」匹配的数据"
                        "（已按名称部分匹配检索）。"
                        "请如实告知用户「知识图谱中没有查到相关数据」，不要凭记忆或估算编造；"
                        "若用户问的是法规/期限类问题，请改用 search_bidding_knowledge。"
                    ),
                    artifacts={
                        "summary": "知识图谱中未查到匹配数据",
                        "graph": {"mode": "template", "anchors": anchors, "rows": 0},
                    },
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                )
            exact = [name for name in names if name.strip().lower() == value.strip().lower()]
            if len(names) > 1 and not exact:
                return self._disambiguation(key, value, candidates.rows, started)
            matched[key] = exact[0] if exact else names[0]

        cypher, params = render(aspect_value, anchors, limit=cap)
        return await self._execute(
            cypher,
            params,
            mode="template",
            aspect=aspect_value,
            anchors=anchors,
            cap=cap,
            started=started,
            matched=matched,
        )

    def _disambiguation(
        self, key: str, value: str, rows: list[dict[str, Any]], started: float
    ) -> ToolResult:
        lines = [f"【知识图谱消歧】{_ANCHOR_LABELS[key]}「{value}」匹配到 {len(rows)} 个候选："]
        for i, row in enumerate(rows, start=1):
            extra = "、".join(
                f"{_COLUMN_LABELS.get(k, k)} {_fmt_cell(v)}"
                for k, v in row.items()
                if k != "name" and v is not None
            )
            lines.append(f"{i}. {row.get('name', '—')}" + (f"（{extra}）" if extra else ""))
        lines.append(
            f"请先向用户确认具体指哪一个，再用完整名称重新调用本工具（{key} 传完整名称）。"
            "不要自行猜测其中一个作答。"
        )
        return ToolResult(
            name=self.spec.name,
            ok=True,
            content="\n".join(lines),
            artifacts={
                "summary": f"匹配到 {len(rows)} 个候选，等待用户确认",
                "graph": {
                    "mode": "template",
                    "anchor": key,
                    "value": value,
                    "candidates": [row.get("name") for row in rows],
                },
            },
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    # ---------------- 执行与输出 ----------------

    async def _execute(
        self,
        cypher: str,
        params: dict[str, Any],
        *,
        mode: str,
        aspect: str | None,
        anchors: dict[str, str],
        cap: int,
        started: float,
        matched: dict[str, str] | None = None,
    ) -> ToolResult:
        try:
            result = await self.service.run_cypher(cypher, params, max_rows=cap)
        except GraphQueryError as exc:
            return self._fail_with(
                exc.code,
                f"知识图谱查询失败：{exc}。请调整查询方式后重试；"
                "若仍无法查询，请如实告知用户知识图谱暂时无法查询，不要编造数据。",
                elapsed=started,
            )

        if not result.rows:
            return ToolResult(
                name=self.spec.name,
                ok=True,
                content=(
                    "知识图谱查询没有返回结果。请如实告知用户「知识图谱中没有查到相关数据」，"
                    "不要凭记忆或估算编造。"
                ),
                artifacts={
                    "summary": "知识图谱未查到数据",
                    "graph": {"mode": mode, "aspect": aspect, "anchors": anchors, "rows": 0},
                },
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

        content = self._render_content(
            mode, aspect, anchors, matched or {}, cypher, result.rows, cap, result.truncated
        )
        # 注意：绝不能把行数据放进 artifacts["sources"]——agent.py 会把该键复制进
        # ToolResultNotice.sources，routes_chat 再对每项做 SourceRef.model_validate，
        # 非 SourceRef 结构会直接抛 ValidationError。
        return ToolResult(
            name=self.spec.name,
            ok=True,
            content=content,
            artifacts={
                "summary": f"知识图谱查询到 {len(result.rows)} 条结果",
                "graph": {
                    "mode": mode,
                    "aspect": aspect,
                    "anchors": anchors,
                    "rows": len(result.rows),
                    "truncated": result.truncated,
                },
            },
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    @staticmethod
    def _render_content(
        mode: str,
        aspect: str | None,
        anchors: dict[str, str],
        matched: dict[str, str],
        cypher: str,
        rows: list[dict[str, Any]],
        cap: int,
        truncated: bool,
    ) -> str:
        if mode == "cypher":
            snippet = " ".join(cypher.split())
            if len(snippet) > 120:
                snippet = snippet[:120] + "…"
            header = f"【知识图谱·只读 Cypher】{snippet}"
        else:
            parts = [
                f"{_ANCHOR_LABELS[key]}≈「{value}」"
                + (f"（解析为「{matched[key]}」）" if matched.get(key) not in (None, value) else "")
                for key, value in anchors.items()
            ]
            condition = "；".join(parts) if parts else "（无，按交易频次全局排序）"
            header = (
                f"【知识图谱·{_ASPECT_LABELS.get(aspect or '', aspect or '')}】条件：{condition}"
            )

        blocks = [header, "", _format_table(rows, aspect)]
        if aspect == "transactions":
            blocks.append(
                "注意：供应商列为该标的物的关联供应商（参考值），不代表该笔交易的中标人。"
            )
        if truncated:
            blocks.append(
                f"（结果超过 {cap} 行，仅显示前 {cap} 行；可调大 limit，上限 {MAX_LIMIT}。）"
            )
        blocks.append("（请只依据以上知识图谱结果作答，不要自行补充或编造。）")
        return "\n".join(blocks)

    def _fail_with(self, code: str, message: str, elapsed: float | None = None) -> ToolResult:
        return ToolResult(
            name=self.spec.name,
            ok=False,
            error=code,
            content=message,
            elapsed_ms=int((time.perf_counter() - elapsed) * 1000) if elapsed else 0,
        )
