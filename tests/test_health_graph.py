"""健康检查里的图谱探测：真实鉴权探测、独立超时、关闭临时驱动。

其余依赖的探测在本文件里被替换为桩（它们各自的网络行为不属于本文件的测试目标）。
"""

from bidding_qa_chatbot.api import routes_health as rh
from bidding_qa_chatbot.schemas import DepHealth


class FakeGraph:
    def __init__(self, ok: bool = True, error: str | None = None) -> None:
        self.ok = ok
        self.error = error
        self.closed = False

    async def healthcheck(self):
        return self.ok, self.error

    async def close(self) -> None:
        self.closed = True


class FakeResult:
    async def single(self):
        return {"ok": 1}


class FakeSession:
    def __init__(self, driver) -> None:
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def run(self, cypher, params=None):
        return FakeResult()


class FakeDriver:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False

    def session(self, **kwargs) -> FakeSession:
        return FakeSession(self)

    async def close(self) -> None:
        self.closed = True


async def _stub_other_probes(monkeypatch) -> None:
    async def _ok(*args, **kwargs):
        return DepHealth(name="stub", status="ok")

    monkeypatch.setattr(rh, "_probe_qdrant", _ok)
    monkeypatch.setattr(rh, "_probe_openai_compatible", _ok)
    monkeypatch.setattr(rh, "_probe_zhipu", _ok)


async def test_disabled_graph_is_not_configured_without_building_driver(
    monkeypatch, settings_factory
):
    def _boom(*args, **kwargs):
        raise AssertionError("ENABLE_NEO4J=false 时不应构造 Neo4j 驱动")

    monkeypatch.setattr("bidding_qa_chatbot.graph.build_driver", _boom)
    settings = settings_factory(enable_neo4j=False)

    dep = await rh._probe_graph(settings)

    assert dep.status == "not_configured"
    assert dep.name == "neo4j"


async def test_injected_graph_service_ok(settings_factory):
    settings = settings_factory(enable_neo4j=True)
    graph = FakeGraph(ok=True)

    dep = await rh._probe_graph(settings, graph)

    assert dep.status == "ok"
    assert graph.closed is False, "注入的服务不该被探测路径关闭"


async def test_injected_graph_service_unavailable_redacts_message(settings_factory):
    settings = settings_factory(enable_neo4j=True)
    # 用 redact() 覆盖的密钥形态（sk-*）：健康检查消息进了响应体，必须脱敏
    graph = FakeGraph(ok=False, error="AuthError: rejected key sk-abcdefghijklmnop12345678")

    dep = await rh._probe_graph(settings, graph)

    assert dep.status == "unavailable"
    assert dep.message and "sk-abcdefghijklmnop12345678" not in dep.message
    assert len(dep.message) <= 200


async def test_cli_path_builds_and_closes_temporary_driver(monkeypatch, settings_factory):
    """CLI `check` 走 graph=None 路径：临时建驱动，探完必须关掉。"""
    drivers: list[FakeDriver] = []

    def _fake_build(settings, **overrides):
        driver = FakeDriver(**overrides)
        drivers.append(driver)
        return driver

    monkeypatch.setattr("bidding_qa_chatbot.graph.build_driver", _fake_build)
    settings = settings_factory(enable_neo4j=True)

    dep = await rh._probe_graph(settings, None)

    assert dep.status == "ok"
    assert len(drivers) == 1
    assert drivers[0].closed is True, "临时驱动必须关闭，否则 CLI 每次 check 都泄漏连接池"


async def test_probe_all_includes_neo4j_entry(monkeypatch, settings_factory):
    await _stub_other_probes(monkeypatch)

    async def _db_ok(settings, db=None):
        return DepHealth(name="postgres", status="ok")

    monkeypatch.setattr(rh, "_probe_database", _db_ok)

    settings = settings_factory(enable_neo4j=True)
    deps = await rh.probe_all(settings, client=object(), graph=FakeGraph(ok=True))

    by_name = {d.name: d for d in deps}
    assert by_name["neo4j"].status == "ok"


async def test_health_deps_for_cli_still_positional(monkeypatch, settings_factory):
    """CLI 以位置参数调用（settings, client），别名与签名不能漂移。"""
    await _stub_other_probes(monkeypatch)

    async def _db_ok(settings, db=None):
        return DepHealth(name="postgres", status="ok")

    monkeypatch.setattr(rh, "_probe_database", _db_ok)

    settings = settings_factory(enable_neo4j=False)
    deps = await rh.health_deps_for_cli(settings, object())

    assert {d.name for d in deps} >= {"neo4j", "postgres"}
