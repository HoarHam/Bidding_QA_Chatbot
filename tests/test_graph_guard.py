"""只读 Cypher 校验（graph/guard.py）。

用例刻意覆盖「看似写操作其实是字符串/参数/注释」与「看似只读其实藏了写操作」两条路，
后者尤其是 `//` 出现在字符串字面量里的绕过构造（校验文本与服务端词法必须一致）。
"""

import re

import pytest

from bidding_qa_chatbot.graph.guard import validate_read_only

HAS_CJK = re.compile(r"[一-鿿]")


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (s:SubjectMatter) RETURN s.name LIMIT 5",
        "OPTIONAL MATCH (s)-[:HAS_TRANSACTION]->(t:Transaction) RETURN t.amount",
        "WITH 1 AS x RETURN x",
        "UNWIND [1, 2, 3] AS x RETURN x",
        "RETURN 1",
        # :Transaction 是图谱自身的标签，必须可用（黑名单里刻意没有 TRANSACTION）
        "MATCH (t:Transaction) RETURN t.amount ORDER BY t.amount DESC LIMIT 3",
        "MATCH (s)-[:HAS_TRANSACTION]->(t) RETURN count(*) AS n",
        # CALL { ... } 是只读子查询，合法
        "CALL { MATCH (s:SubjectMatter) RETURN s } RETURN s",
        # 字符串字面量里的写关键字必须放行
        "RETURN 'DELETE' AS note",
        "MATCH (n) WHERE n.name CONTAINS 'SET' RETURN n",
        # $参数 必须替换后再校验，否则 $delete_flag 会被误杀
        "MATCH (n) WHERE n.x = $delete_flag RETURN n",
        # 反引号标识符同理
        "MATCH (n:`Create`) RETURN n",
        # 单个结尾分号容忍
        "MATCH (s) RETURN s.name;",
        # 注释里的关键字不算数
        "// 这里写 DELETE 也没事\nMATCH (s) RETURN s",
        "/* CREATE */ MATCH (s) RETURN s",
        # 字符串内部的 // 不是注释（单遍词法扫描的关键用例）
        "MATCH (n) WHERE n.x = 'a//b' RETURN n",
    ],
)
def test_accepts_read_only(cypher):
    ok, reason = validate_read_only(cypher)
    assert ok, f"应放行却被拒绝: {reason}"


@pytest.mark.parametrize(
    ("cypher", "keyword"),
    [
        ("CREATE (n:SubjectMatter)", "CREATE"),
        ("MATCH (n) MERGE (m:SubjectMatter)", "MERGE"),
        ("MATCH (n) SET n.x = 1", "SET"),
        ("MATCH (n) DELETE n", "DELETE"),
        ("MATCH (n) DETACH DELETE n", "DETACH"),
        ("MATCH (n) REMOVE n.x", "REMOVE"),
        ("DROP CONSTRAINT sm_name_unique", "DROP"),
        ("FOREACH (x IN [1] | SET n.x = 1)", "FOREACH"),
        ("LOAD CSV FROM 'file:///x' AS line RETURN line", "LOAD"),
        ("CALL apoc.coll.sum([1])", "CALL"),
        ("CALL dbms.security.listUsers()", "CALL"),
        ("RETURN apoc.coll.sum([1]) AS s", "APOC"),
        ("SHOW DATABASES", "SHOW"),
        ("GRANT ROLE reader TO neo4j", "GRANT"),
        ("REVOKE ROLE reader FROM neo4j", "REVOKE"),
        ("ALTER DATABASE x SET ACCESS READ ONLY", "ALTER"),
        ("USING PERIODIC COMMIT LOAD CSV FROM 'x' AS l RETURN l", "PERIODIC"),
        # 子查询拆掉 CALL 之后，里面的写操作仍必须拦下
        ("CALL { CREATE (n) RETURN n } RETURN n", "CREATE"),
        # `//` 藏在字符串里：校验文本必须与服务端理解一致，DELETE 是真实语句
        ("MATCH (n) WHERE n.x = 'a//b' DELETE n", "DELETE"),
    ],
)
def test_rejects_writes_and_procedures(cypher, keyword):
    ok, reason = validate_read_only(cypher)
    assert not ok
    assert reason and keyword in reason
    assert HAS_CJK.search(reason), "拒绝文案应是中文"


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n) RETURN n; DROP DATABASE neo4j",
        "RETURN 1; RETURN 2",
    ],
)
def test_rejects_multiple_statements(cypher):
    ok, reason = validate_read_only(cypher)
    assert not ok
    assert reason and "一条语句" in reason


def test_rejects_explain_prefix():
    ok, reason = validate_read_only("EXPLAIN MATCH (n) RETURN n")
    assert not ok
    assert reason and "开头" in reason


@pytest.mark.parametrize("cypher", ["", "   ", "// 仅注释", "/* 仅注释 */", ";"])
def test_rejects_empty(cypher):
    ok, reason = validate_read_only(cypher)
    assert not ok
    assert reason and HAS_CJK.search(reason)
