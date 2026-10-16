"""尚未实现的集成的接口桩测试（当前只剩联网搜索：Tavily / Exa）。

这些用例锁住的是「仍**没有**被实现」这个事实：桩必须能报告配置状态，
但任何真实业务调用都必须明确抛出 `NotImplementedError`，
而不是悄悄返回空结果让人误以为功能已经可用。

PostgreSQL 与 Neo4j 已升级为真实实现，各自的「是必需依赖」断言防止
有人把它们挪回 phase2 可选依赖里。
"""

from __future__ import annotations

import importlib.util

import pytest

from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.integrations.stubs import (
    ExaStub,
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
    assert importlib.util.find_spec("tavily") is None, "tavily 不该被安装"


def test_postgres_is_a_real_dependency_now():
    """PostgreSQL 已从「桩」升级为真实实现（账号 + 对话历史）。

    这条断言防止有人把它当成未实现的功能又挪回 phase2 可选依赖里。
    """
    for module in ("sqlalchemy", "psycopg"):
        assert importlib.util.find_spec(module) is not None, f"{module} 应当是必需依赖"

    from bidding_qa_chatbot.db import models

    assert {"users", "conversations", "messages"} <= set(models.Base.metadata.tables)


def test_neo4j_is_a_real_dependency_now():
    """Neo4j 已从「桩」升级为真实实现（知识图谱查询，见 graph/ 包）。"""
    assert importlib.util.find_spec("neo4j") is not None, "neo4j 应当是必需依赖"

    from bidding_qa_chatbot.graph import GraphService, build_driver, probe_with_timeout

    assert all(callable(obj) for obj in (GraphService, build_driver, probe_with_timeout))


def test_stubs_report_disabled_by_default():
    s = _settings()
    assert TavilyStub(s).status() == "not_configured"
    assert ExaStub(s).status() == "not_configured"


def test_stubs_report_configured_when_env_enables_them():
    s = _settings(enable_web_search=True)
    assert TavilyStub(s).status() == "configured"
    assert ExaStub(s).status() == "configured"


async def test_disabled_stubs_do_not_attempt_connection():
    """未启用时探针不该去连任何东西，直接返回 not_configured。"""
    health = await TavilyStub(_settings()).probe()
    assert health.status == "not_configured"
    assert "第二阶段" in (health.message or "")


async def test_enabled_web_search_stub_reports_ready():
    s = _settings(enable_web_search=True, tavily_api_key="tvly-test")
    health = await TavilyStub(s).probe()
    assert health.status == "ok"


def test_phase2_status_no_longer_lists_implemented_integrations():
    """状态摘要里不该再把 postgres / neo4j 当作未实现的集成。"""
    status = build_phase2_status(_settings())
    assert "postgres" not in status
    assert "neo4j" not in status
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
    status = build_phase2_status(_settings(enable_web_search=True))
    assert status["stage"] == "phase2"
    assert status["web_search"] == "enabled"
