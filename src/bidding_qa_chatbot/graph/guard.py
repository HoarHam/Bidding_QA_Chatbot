"""只读 Cypher 校验（`cypher` 兜底参数的守门人）。

真正的安全网是**服务端**：查询一律在 `execute_read` 的读事务里执行，
读事务中的写操作会被服务器直接拒绝（见 `service.py`）。这里的校验解决另外两件事：

1. 给模型一条**可理解、可重试**的失败信息，而不是一串服务端报错；
2. 挡住 `CALL dbms.*` / `CALL apoc.*` / `LOAD CSV` —— 这几类在**读事务里也合法**
   （过程调用与外部 IO 不受访问模式限制），只能在客户端挡。

实现要点（改动前请先读注释）：

- 注释/字符串剥离用**单遍词法扫描**，不能用独立正则各扫一遍：
  `MATCH (n) WHERE n.x = 'a//b' DELETE n` 里 `//` 在字符串**内部**，
  若先按行注释剥掉 `//b' DELETE n`，`DELETE` 会从校验文本里消失而被放行——
  但服务端词法是字符串感知的，`DELETE n` 是真实语句。单遍扫描先消费整个字符串，
  保证校验文本与服务器的理解一致。
- `$参数` 必须替换成占位符：否则 `WHERE x = $delete_flag` 会被 `\\bdelete\\b` 误杀。
- 黑名单**不含 `TRANSACTION`**：图谱自身有 `:Transaction` 标签（见 data/batch/to_neo4j.py），
  拦它等于废掉一半可用查询；`COMMIT` 已单独拦下（Neo4j 的显式事务由驱动 API 管理，
  Cypher 文本里不存在 BEGIN/COMMIT 语句）。
- `CALL { ... }` 是只读子查询，合法：先把 `CALL {` 里的 CALL 关键字去掉，
  剩下的**裸 `CALL`（过程调用）一律拒绝**。
"""

from __future__ import annotations

import re

#: 写操作 + 读事务里也合法的危险过程/IO。词边界匹配，忽略大小写。
_FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|FOREACH|CALL|LOAD|CSV|"
    r"APOC|DBMS|GDS|PERIODIC|COMMIT|GRANT|REVOKE|ALTER|RENAME|INSTALL|"
    r"TERMINATE|SHOW|USE|ADMIN|CONSTRAINT|INDEX|DATABASE)\b",
    re.IGNORECASE,
)

#: 允许出现的首词（OPTIONAL 指 OPTIONAL MATCH；`{` 指被拆掉 CALL 的只读子查询）
_ALLOWED_STARTS = ("MATCH", "OPTIONAL", "WITH", "UNWIND", "RETURN", "{")

_CALL_SUBQUERY = re.compile(r"\bCALL\s*\{", re.IGNORECASE)
_FIRST_WORD = re.compile(r"[A-Za-z{]+")


def _sanitize(text: str) -> str:
    """单遍扫描，抹掉注释、字符串字面量、参数名、反引号标识符，只留语句结构。

    返回的文本仅用于**关键字校验**，永不用于执行。
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "*":
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
            out.append(" ")
        elif ch == "/" and nxt == "/":
            end = text.find("\n", i + 2)
            i = n if end == -1 else end
            out.append(" ")
        elif ch in ("'", '"'):
            quote = ch
            i += 1
            while i < n:
                if text[i] == "\\":  # 转义序列整体跳过，避免 \" 误判字符串结束
                    i += 2
                    continue
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            out.append("''")
        elif ch == "`":
            end = text.find("`", i + 1)
            i = n if end == -1 else end + 1
            out.append("`x`")
        elif ch == "$":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            out.append("$p" if j > i + 1 else ch)
            i = j
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def validate_read_only(cypher: str) -> tuple[bool, str | None]:
    """校验 Cypher 是否只读。返回 `(是否通过, 拒绝原因)`。

    这是**可用性**层，不是安全边界——安全边界见模块 docstring。
    """
    if not isinstance(cypher, str) or not cypher.strip():
        return False, "Cypher 为空，请提供要执行的只读查询。"

    sanitized = _CALL_SUBQUERY.sub("{", _sanitize(cypher)).strip()
    # 允许一个结尾分号（模型常带上），其余分号视为多语句
    body = sanitized.rstrip(";").strip()
    if not body:
        return False, "Cypher 为空，请提供要执行的只读查询。"
    if ";" in body:
        return False, "只允许一条语句：检测到分号分隔的多条语句。"

    hit = _FORBIDDEN.search(body)
    if hit:
        return False, (
            f"检测到被禁止的关键字「{hit.group(0).upper()}」：只允许只读查询，"
            "禁止写操作、过程调用（CALL / APOC / DBMS）与 LOAD CSV。"
        )

    match = _FIRST_WORD.match(body)
    first_word = match.group(0).upper() if match else ""
    if first_word not in _ALLOWED_STARTS:
        return False, (
            "只允许以 MATCH / OPTIONAL MATCH / WITH / UNWIND / RETURN 开头的只读查询"
            f"（当前以「{first_word or body[:10]}」开头）。"
        )
    return True, None
