"""联网搜索的配置级健康探测（不发付费请求）。"""

from bidding_qa_chatbot.api import routes_health as rh
from bidding_qa_chatbot.schemas import DepHealth


async def test_disabled_web_search_is_not_configured(settings_factory):
    settings = settings_factory(enable_web_search=False)

    assert (await rh._probe_tavily(settings)).status == "not_configured"
    assert (await rh._probe_exa(settings)).status == "not_configured"


async def test_missing_keys_are_not_configured(settings_factory):
    settings = settings_factory(enable_web_search=True, tavily_api_key=None, exa_api_key=None)

    tavily = await rh._probe_tavily(settings)
    exa = await rh._probe_exa(settings)
    assert tavily.status == "not_configured"
    assert "TAVILY_API_KEY" in (tavily.message or "")
    assert exa.status == "not_configured"
    assert "EXA_API_KEY" in (exa.message or "")


async def test_configured_keys_are_ok(settings_factory, monkeypatch):
    monkeypatch.setattr(rh.shutil, "which", lambda name: "C:/npx.cmd")
    settings = settings_factory(
        enable_web_search=True, tavily_api_key="tvly-x", exa_api_key="exa-x"
    )

    assert (await rh._probe_tavily(settings)).status == "ok"
    assert (await rh._probe_exa(settings)).status == "ok"


async def test_missing_npx_keeps_exa_ok_but_notes_fallback(settings_factory, monkeypatch):
    """主路径是 Exa API；npx 缺失只影响 MCP 回退，不应把 exa 标成不可用。"""
    monkeypatch.setattr(rh.shutil, "which", lambda name: None)
    settings = settings_factory(enable_web_search=True, exa_api_key="exa-x")

    dep = await rh._probe_exa(settings)

    assert dep.status == "ok"
    assert "npx" in (dep.message or "")


async def test_probe_all_includes_web_entries(monkeypatch, settings_factory):
    async def _ok(*args, **kwargs):
        return DepHealth(name="stub", status="ok")

    monkeypatch.setattr(rh, "_probe_qdrant", _ok)
    monkeypatch.setattr(rh, "_probe_openai_compatible", _ok)
    monkeypatch.setattr(rh, "_probe_zhipu", _ok)

    async def _db(settings, db=None):
        return DepHealth(name="postgres", status="ok")

    monkeypatch.setattr(rh, "_probe_database", _db)
    monkeypatch.setattr(rh.shutil, "which", lambda name: "npx")

    settings = settings_factory(
        enable_web_search=True, tavily_api_key="tvly-x", exa_api_key="exa-x"
    )
    deps = await rh.probe_all(settings, client=object())
    by_name = {dep.name: dep for dep in deps}

    assert by_name["tavily"].status == "ok"
    assert by_name["exa"].status == "ok"
    assert by_name["langfuse"].status == "not_configured"


async def test_langfuse_probe_is_config_level(settings_factory):
    disabled = await rh._probe_langfuse(settings_factory(langfuse_enabled=False))
    assert disabled.status == "not_configured"
    assert "LANGFUSE_ENABLED" in (disabled.message or "")

    missing = await rh._probe_langfuse(settings_factory(langfuse_enabled=True))
    assert missing.status == "not_configured"
    assert "LANGFUSE_PUBLIC_KEY" in (missing.message or "")

    configured = await rh._probe_langfuse(
        settings_factory(
            langfuse_enabled=True,
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
        )
    )
    assert configured.status == "ok"
    assert "不做联网探活" in (configured.message or "")
