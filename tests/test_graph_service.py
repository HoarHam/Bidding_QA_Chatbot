"""GraphService / build_driver：只读事务、超时、截断、错误映射、探活。

全部离线：用 FakeDriver（鸭子类型）记录 `driver.session(...)` 的实参、
`execute_read` 的回调与 fetch 大小；不引入任何网络。
"""

import asyncio

import pytest
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from bidding_qa_chatbot.graph.client import build_driver
from bidding_qa_chatbot.graph.service import (
    GraphQueryError,
    GraphService,
    probe_with_timeout,
)


class FakeRecord:
    def __init__(self, data: dict) -> None:
        self._data = data

    def data(self) -> dict:
        return dict(self._data)


class FakeResult:
    def __init__(self, records: list[FakeRecord], fetch_sizes: list[int]) -> None:
        self.records = records
        self.fetch_sizes = fetch_sizes

    async def fetch(self, n: int) -> list[FakeRecord]:
        self.fetch_sizes.append(n)
        return self.records[:n]

    async def single(self):
        return self.records[0] if self.records else None


class FakeTx:
    def __init__(self, records=(), error: Exception | None = None, delay: float = 0.0) -> None:
        self.records = [FakeRecord(r) for r in records]
        self.error = error
        self.delay = delay
        self.calls: list[tuple[str, dict]] = []
        self.fetch_sizes: list[int] = []

    async def run(self, cypher: str, params: dict):
        self.calls.append((cypher, params))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return FakeResult(self.records, self.fetch_sizes)


class FakeSession:
    def __init__(self, driver: "FakeDriver") -> None:
        self.driver = driver

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False

    async def execute_read(self, fn, *args, **kwargs):
        self.driver.executed.append(fn)
        return await fn(self.driver.tx)

    async def run(self, cypher: str, params: dict | None = None):
        self.driver.health_calls.append(cypher)
        if self.driver.health_error is not None:
            raise self.driver.health_error
        return FakeResult([FakeRecord({"ok": 1})], [])


class FakeDriver:
    def __init__(self, tx: FakeTx | None = None, health_error: Exception | None = None) -> None:
        self.tx = tx or FakeTx()
        self.health_error = health_error
        self.session_calls: list[dict] = []
        self.executed: list = []
        self.health_calls: list[str] = []
        self.closed = False

    def session(self, **kwargs) -> FakeSession:
        self.session_calls.append(kwargs)
        return FakeSession(self)

    async def close(self) -> None:
        self.closed = True


def _service(settings_factory, tx: FakeTx | None = None, **settings_kw):
    settings = settings_factory(**settings_kw)
    driver = FakeDriver(tx)
    return GraphService(driver, settings), driver


async def test_run_cypher_uses_read_session_and_timeout_attribute(settings_factory):
    tx = FakeTx(records=[{"name": "空调"}])
    service, driver = _service(settings_factory, tx, neo4j_query_timeout=7.5)

    result = await service.run_cypher("RETURN 1 AS n", {"a": 1})

    assert result.rows == [{"name": "空调"}]
    assert result.truncated is False
    assert driver.session_calls[0]["database"] == "BiddingRelations"
    assert driver.session_calls[0]["default_access_mode"] == "READ"
    assert tx.calls == [("RETURN 1 AS n", {"a": 1})]
    # 承重细节：驱动 6.x 从回调函数的 timeout 属性读取事务超时（不是 tx.run(timeout=)）
    assert len(driver.executed) == 1
    assert driver.executed[0].timeout == 7.5


async def test_truncation_fetches_cap_plus_one(settings_factory):
    tx = FakeTx(records=[{"i": i} for i in range(10)])
    service, _ = _service(settings_factory, tx)

    result = await service.run_cypher("RETURN x", max_rows=3)

    assert len(result.rows) == 3
    assert result.truncated is True
    assert tx.fetch_sizes == [4]  # cap + 1，多取一行用于判断截断


async def test_not_truncated_when_rows_under_cap(settings_factory):
    tx = FakeTx(records=[{"i": 0}, {"i": 1}])
    service, _ = _service(settings_factory, tx)

    result = await service.run_cypher("RETURN x", max_rows=3)

    assert len(result.rows) == 2
    assert result.truncated is False
    assert tx.fetch_sizes == [4]


async def test_cap_bounded_by_settings(settings_factory):
    tx = FakeTx(records=[])
    service, _ = _service(settings_factory, tx, neo4j_max_rows=5)

    await service.run_cypher("RETURN x", max_rows=999)

    assert tx.fetch_sizes == [6]  # 被 neo4j_max_rows 钳制为 5


async def test_service_unavailable_maps_to_graph_query_failed(settings_factory):
    tx = FakeTx(error=ServiceUnavailable("failed to connect"))
    service, _ = _service(settings_factory, tx)

    with pytest.raises(GraphQueryError) as excinfo:
        await service.run_cypher("RETURN 1")

    assert excinfo.value.code == "graph_query_failed"
    assert "Neo4j" in str(excinfo.value)


class _FakeTransactionTimeout(Neo4jError):
    """6.x 没有 TransactionTimedOut 异常类也没有公开的带 code 构造器，
    用子类覆写 code 属性，模拟服务端返回的超时代码。"""

    @property
    def code(self) -> str:
        return "Neo.TransientError.Transaction.TransactionTimedOut"


async def test_server_transaction_timeout_maps_to_graph_timeout(settings_factory):
    error = _FakeTransactionTimeout("Transaction timed out")
    service, _ = _service(settings_factory, FakeTx(error=error), neo4j_query_timeout=8.0)

    with pytest.raises(GraphQueryError) as excinfo:
        await service.run_cypher("RETURN 1")

    assert excinfo.value.code == "graph_timeout"


async def test_slow_query_hits_wait_for_backstop(settings_factory):
    # 服务端超时（0.01s）没生效时的兜底：wait_for(超时 + 2s)
    service, _ = _service(settings_factory, FakeTx(delay=2.2), neo4j_query_timeout=0.01)

    with pytest.raises(GraphQueryError) as excinfo:
        await service.run_cypher("RETURN 1")

    assert excinfo.value.code == "graph_timeout"


async def test_healthcheck_ok(settings_factory):
    service, driver = _service(settings_factory)

    ok, error = await service.healthcheck()

    assert ok is True and error is None
    assert driver.health_calls == ["RETURN 1 AS ok"]
    assert driver.session_calls[0]["database"] == "BiddingRelations"
    assert driver.session_calls[0]["default_access_mode"] == "READ"


async def test_healthcheck_failure_does_not_leak_password(settings_factory):
    driver = FakeDriver(health_error=ServiceUnavailable("failed to connect to localhost:7687"))
    service = GraphService(driver, settings_factory(neo4j_password="hunter2-secret"))

    ok, error = await service.healthcheck()

    assert ok is False
    assert error and "ServiceUnavailable" in error
    assert "hunter2-secret" not in error


async def test_driver_none_is_unavailable_but_never_raises(settings_factory):
    service = GraphService(None, settings_factory())

    assert service.available is False
    ok, error = await service.healthcheck()
    assert ok is False and error and "ENABLE_NEO4J" in error
    with pytest.raises(GraphQueryError) as excinfo:
        await service.run_cypher("RETURN 1")
    assert excinfo.value.code == "graph_unavailable"
    await service.close()  # 不应抛


async def test_close_closes_driver(settings_factory):
    service, driver = _service(settings_factory)
    await service.close()
    assert driver.closed is True


class _SlowService:
    async def healthcheck(self) -> tuple[bool, str | None]:
        await asyncio.sleep(1.0)
        return True, None


async def test_probe_with_timeout_returns_timeout_message():
    ok, error = await probe_with_timeout(_SlowService(), timeout=0.05)
    assert ok is False
    assert error and "探活超时" in error


async def test_probe_with_timeout_passes_through(settings_factory):
    service, _ = _service(settings_factory)
    ok, error = await probe_with_timeout(service, timeout=3.0)
    assert ok is True and error is None


def test_build_driver_uses_auth_and_never_inlines_password(monkeypatch, settings_factory):
    captured: dict = {}

    def fake_driver(uri, **kwargs):
        captured["uri"] = uri
        captured.update(kwargs)
        return "driver"

    monkeypatch.setattr("neo4j.AsyncGraphDatabase.driver", fake_driver)
    settings = settings_factory(neo4j_password="hunter2", neo4j_uri="bolt://localhost:7687")

    assert build_driver(settings) == "driver"
    assert captured["uri"] == "bolt://localhost:7687"
    assert "hunter2" not in captured["uri"]
    assert captured["auth"] == ("neo4j", "hunter2")
    assert (
        captured["connection_acquisition_timeout"] == settings.neo4j_connection_acquisition_timeout
    )
    assert captured["connection_timeout"] == settings.neo4j_connection_timeout


def test_build_driver_without_password_uses_empty_string(monkeypatch, settings_factory):
    captured: dict = {}

    def fake_driver(uri, **kwargs):
        captured.update(kwargs)
        return "driver"

    monkeypatch.setattr("neo4j.AsyncGraphDatabase.driver", fake_driver)
    build_driver(settings_factory())

    assert captured["auth"] == ("neo4j", "")
