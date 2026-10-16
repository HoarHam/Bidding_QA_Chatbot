"""Neo4j 知识图谱查询：驱动构造、只读查询服务、只读 Cypher 校验与参数化模板。

门控开关是 `settings.enable_neo4j`（环境变量 `ENABLE_NEO4J`）；
工具层见 `bidding_qa_chatbot/agent/graph_tool.py`，图模型见 `data/batch/to_neo4j.py`。

分层：
- `client.py`  驱动构造（惰性建连，凭据走 auth 不进 URI）
- `guard.py`   只读校验（拦截读事务里也合法的 CALL/LOAD CSV；服务端读事务是兜底）
- `cypher.py`  Cypher 模板（结构化参数 → 参数化查询）
- `service.py` 查询服务（只读事务 + 超时 + 客户端截断 + 错误映射）
"""

from bidding_qa_chatbot.graph.client import build_driver
from bidding_qa_chatbot.graph.service import (
    GraphQueryError,
    GraphQueryResult,
    GraphService,
    probe_with_timeout,
)

__all__ = [
    "GraphQueryError",
    "GraphQueryResult",
    "GraphService",
    "build_driver",
    "probe_with_timeout",
]
