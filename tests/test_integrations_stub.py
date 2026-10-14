"""第二阶段接口桩测试。

这些用例锁住的是「第二阶段**没有**被实现」这个事实：
桩必须能报告配置状态与可达性，但任何真实业务调用都必须明确抛出 `NotImplementedError`，
而不是悄悄返回空结果让人误以为功能已经可用。

另外还验证：桩不依赖任何第二阶段驱动（`neo4j` / `psycopg` / `sqlalchemy` /
`tavily-python` 在本阶段根本没安装）。
"""

from __future__ import annotations

import importlib.util

import pytest

from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.integrations.stubs import (
    ExaStub,
    Neo4jStub,
    TavilyStub,
    build_phase2_status,
)


def _settings(**kw) -> Settings:
    base = {
        "qdrant_url": "http://x",
        "qdrant_api_key": "k",
        "deepseek_api_key": "sk-x",
        "zhipu_api_key": "i.s",
    }
    base.update(kw)
    return Settings(_env_file=None, **base)


def test_remaining_stub_drivers_are_not_installed():
    """仍未实现的集成，其驱动不该被安装——它们留在 optional-dependencies.phase2 里。"""
    for module in ("neo4j", "tavily"):
        assert importlib.util.find_spec(module) is None, f"{module} 不该被安装"


def test_postgres_is_a_real_dependency_now():
    """PostgreSQL 已从「桩」升级为真实实现（账号 + 对话历史）。

    这条断言防止有人把它当成未实现的功能又挪回 phase2 可选依赖里。
    """
    import importlib.util

    for module in ("sqlalchemy", "psycopg"):
        assert importlib.util.find_spec(module) is not None, f"{module} 应当是必需依赖"

    from bidding_qa_chatbot.db import models

    assert {"users", "conversations", "messages"} <= set(models.Base.metadata.tables)


def test_stubs_report_disabled_by_default():
    s = _settings()
    assert Neo4jStub(s).status() == "not_configured"
    assert TavilyStub(s).status() == "not_configured"
    assert ExaStub(s).status() == "not_configured"


def test_stubs_report_configured_when_env_enables_them():
    s = _settings(enable_neo4j=True, enable_web_search=True)
    assert Neo4jStub(s).status() == "configured"
    assert TavilyStub(s).status() == "configured"
    assert ExaStub(s).status() == "configured"


async def test_disabled_stubs_do_not_attempt_connection():
    """未启用时探针不该去连任何东西，直接返回 not_configured。"""
    health = await Neo4jStub(_settings()).probe()
    assert health.status == "not_configured"
    assert "第二阶段" in (health.message or "")


async def test_enabled_stub_probes_and_fails_cleanly():
    """启用后探针会真的连一次；连不上要返回 unavailable 而不是抛异常。"""
    s = _settings(enable_neo4j=True, neo4j_uri="bolt://127.0.0.1:1")
    health = await Neo4jStub(s).probe()
    assert health.status == "unavailable"
    assert health.message and "第二阶段" not in health.message


async def test_neo4j_query_is_explicitly_unimplemented():
    with pytest.raises(NotImplementedError, match="第二阶段"):
        await Neo4jStub(_settings()).query("MATCH (n) RETURN n")


def test_phase2_status_no_longer_lists_postgres_as_a_stub():
    """状态摘要里不该再把 postgres 当作未实现的集成。"""
    status = build_phase2_status(_settings())
    assert "postgres" not in status
    assert status["stage"] == "phase2"


async def test_tavily_search_is_explicitly_unimplemented():
    with pytest.raises(NotImplementedError, match="第二阶段"):
        await TavilyStub(_settings()).search("招投标")


async def test_exa_search_is_explicitly_unimplemented():
    with pytest.raises(NotImplementedError, match="第二阶段"):
        await ExaStub(_settings()).search("招投标")


async def test_tavily_probe_reports_missing_key():
    s = _settings(enable_web_search=True, tavily_api_key=None)
    health = await TavilyStub(s).probe()
    assert health.status == "not_configured"
    assert "TAVILY_API_KEY" in (health.message or "")


def test_phase2_status_summary():
    status = build_phase2_status(_settings())
    assert status["stage"] == "phase2"
    assert status["neo4j"] == "disabled"


def test_stub_never_echoes_secret():
    s = _settings(enable_neo4j=True, neo4j_password="超级机密口令")
    assert "超级机密口令" not in Neo4jStub(s).status()
