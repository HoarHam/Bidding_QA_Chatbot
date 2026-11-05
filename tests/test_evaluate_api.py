"""/evaluate 页面与评测 JSON API 的路由测试（不启动真实评测）。"""

from __future__ import annotations

from typing import Any


class _FakeEvaluationService:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.run = {
            "id": "20260101-000000-deadbeef",
            "mode": "offline",
            "status": "completed",
            "created_at": "2026-01-01T00:00:00+00:00",
            "summary": {"tool_selection_accuracy": 1.0, "cases": 1},
            "progress": {"phase": "done", "completed": 1, "total": 1},
            "cases": [],
            "users": [],
            "metric_errors": [],
        }

    def config(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "reason": None,
            "judge": {"provider": "zhipu", "model": "judge"},
            "evaluated": {"provider": "deepseek", "model": "deepseek-flash"},
            "agent_available": True,
            "db_available": True,
            "langfuse_available": True,
            "encoder_available": True,
            "dataset": {"name": "evaluate_cases.json", "exists": True, "cases": 12},
            "metrics": [{"key": "faithfulness", "label": "忠诚度"}],
            "history_limit": 20,
            "online_limit": 10,
            "warnings": [],
        }

    def start_offline(self) -> dict[str, Any]:
        self.started.append("offline")
        return self.run

    def start_online(self) -> dict[str, Any]:
        self.started.append("online")
        return {**self.run, "mode": "online"}

    def list_runs(self, username: str | None = None) -> list[dict[str, Any]]:
        return [self.run]

    def list_users(self) -> list[dict[str, Any]]:
        return [{"username": "alice", "cases": 3}]

    def get_run(self, run_id: str, username: str | None = None) -> dict[str, Any] | None:
        return self.run if run_id == self.run["id"] else None


async def test_evaluate_page_and_public_config(api_client):
    api_client.app.state.evaluation = _FakeEvaluationService()

    page = await api_client.get("/evaluate")
    assert page.status_code == 200
    assert "Agent 综合评测" in page.text
    assert "运行离线评测" in page.text
    assert "bidding_qa_eval_token" not in page.text
    assert "按用户名筛选" in page.text

    config = await api_client.get("/api/evaluate/config")
    assert config.status_code == 200
    assert config.json()["judge"]["model"] == "judge"


async def test_evaluate_start_and_history_without_login(api_client):
    service = _FakeEvaluationService()
    api_client.app.state.evaluation = service

    started = await api_client.post("/api/evaluate/offline")
    assert started.status_code == 200
    assert started.json()["id"] == service.run["id"]
    assert service.started == ["offline"]

    online = await api_client.post("/api/evaluate/online")
    assert online.status_code == 200
    assert service.started == ["offline", "online"]

    runs = await api_client.get("/api/evaluate/runs")
    assert runs.status_code == 200
    assert runs.json()[0]["summary"]["tool_selection_accuracy"] == 1.0

    users = await api_client.get("/api/evaluate/users")
    assert users.status_code == 200
    assert users.json()[0]["username"] == "alice"

    detail = await api_client.get(f"/api/evaluate/runs/{service.run['id']}")
    assert detail.status_code == 200
    assert detail.json()["mode"] == "offline"

    missing = await api_client.get("/api/evaluate/runs/20260101-000000-ffffffff")
    assert missing.status_code == 404


async def test_evaluate_returns_503_without_service(api_client):
    # api_client 默认没有注入 evaluation（模拟未启动 lifespan 的测试 app）
    assert (await api_client.get("/api/evaluate/config")).status_code == 503
