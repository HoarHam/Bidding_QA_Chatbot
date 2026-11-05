"""Agent 综合评测：数据集、后台编排、在线/离线隔离（全程离线）。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bidding_qa_chatbot.agent.agent import (
    AgentDone,
    ToolCallNotice,
    ToolResultNotice,
)
from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.db import repository as repo
from bidding_qa_chatbot.evaluation.dataset import (
    EvaluationDatasetError,
    load_cases,
    normalize_expected_tool,
    tool_selection_matches,
)
from bidding_qa_chatbot.evaluation.ragas_runner import RagasBatch, mean
from bidding_qa_chatbot.evaluation.runner import (
    EvaluationDisabled,
    EvaluationService,
)
from bidding_qa_chatbot.llm.base import StreamEvent
from tests.conftest import REPO_ROOT, requires_corpus


def _write_cases(tmp_path: Path) -> Path:
    path = tmp_path / "processed" / "evaluate_cases.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "question": "q1",
                    "reference": "r1",
                    "expected_tool": "search_bidding_knowledge",
                },
                {
                    "question": "q2",
                    "reference": "r2",
                    "expected_tool": "search_knowledge_graph",
                },
                {
                    "question": "q3",
                    "reference": "",
                    "expected_tool": "ask_clarification",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _eval_settings(settings_factory, tmp_path: Path, **overrides) -> Settings:
    base = {
        "evaluate_enabled": True,
        "evaluate_judge_provider": "zhipu",
        "evaluate_judge_model": "judge-test",
        "data_dir": tmp_path,
    }
    base.update(overrides)
    return settings_factory(**base)


class _FakeRegistry:
    names = ["search_bidding_knowledge", "query_bidding_graph"]

    def __contains__(self, name: object) -> bool:
        return name in self.names


class _FakeAgent:
    registry = _FakeRegistry()

    async def astream(self, messages, mode="fast", **kwargs):
        question = messages[-1].content
        if question == "q1":
            yield ToolCallNotice("search_bidding_knowledge", {}, "x")
            yield ToolResultNotice("search_bidding_knowledge", True, 1, "ok", content="ctx-1")
            yield StreamEvent("content", "answer-1")
        elif question == "q2":
            yield ToolCallNotice("query_bidding_graph", {}, "x")
            yield ToolResultNotice("query_bidding_graph", True, 1, "ok", content="ctx-2")
            yield StreamEvent("content", "answer-2")
        else:
            yield StreamEvent("content", "请补充项目信息")
        yield AgentDone(
            finish_reason="stop",
            steps=1,
            tool_calls=1,
            provider="fake",
            model="fake-model",
        )


class _FakeEvaluator:
    calls: list[tuple[int, bool]] = []
    judge_calls: list[str] = []

    def __init__(self, settings: Settings, encoder: Any) -> None:
        pass

    async def judge_online_case(self, *, question: str, answer: str, actual_tools, available_tools):
        type(self).judge_calls.append(question)
        return SimpleNamespace(
            expected_tool="search_bidding_knowledge",
            tool_selection_correct=True,
            reference_answer="判分模型生成的参考答案",
            reasoning="工具与问题匹配",
        )

    async def evaluate(self, samples, *, include_context_recall: bool) -> RagasBatch:
        type(self).calls.append((len(samples), include_context_recall))
        scores = []
        for _ in samples:
            scores.append(
                {
                    "faithfulness": 0.9,
                    "answer_relevancy": 0.8,
                    "context_recall": 0.7 if include_context_recall else None,
                }
            )
        summary = {
            "faithfulness": 0.9 if samples else None,
            "answer_relevancy": 0.8 if samples else None,
            "context_recall": 0.7 if include_context_recall and samples else None,
        }
        return RagasBatch(scores=scores, summary=summary)

    def close(self) -> None:
        pass


def test_dataset_aliases_and_tool_matching(tmp_path):
    path = _write_cases(tmp_path)
    cases = load_cases(path)

    assert len(cases) == 3
    assert cases[0].expected_tool == "search_bidding_knowledge"
    assert cases[1].expected_tool == "query_bidding_graph"
    assert cases[2].expected_no_tool is True
    assert tool_selection_matches(cases[0], ["search_bidding_knowledge"])
    assert not tool_selection_matches(cases[0], ["query_bidding_graph"])
    assert tool_selection_matches(cases[1], ["query_bidding_graph"])
    assert tool_selection_matches(cases[2], [])
    assert not tool_selection_matches(cases[2], ["query_bidding_graph"])
    assert normalize_expected_tool("search_postgresql") == ("query_bidding_graph", False)


@requires_corpus
def test_dataset_real_file_is_utf8_and_has_cases():
    path = REPO_ROOT / "data" / "processed" / "evaluate_cases.json"
    cases = load_cases(path)
    assert len(cases) >= 10
    assert any("单一来源" in case.question for case in cases)


def test_dataset_rejects_bad_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(EvaluationDatasetError):
        load_cases(path)


def test_mean_ignores_missing_scores():
    assert mean([0.5, None, 0.7]) == 0.6
    assert mean([None, None]) is None


async def test_offline_run_generates_cases_and_persists(tmp_path, settings_factory, monkeypatch):
    _write_cases(tmp_path)
    monkeypatch.setattr("bidding_qa_chatbot.evaluation.runner.RagasEvaluator", _FakeEvaluator)
    _FakeEvaluator.calls.clear()
    settings = _eval_settings(settings_factory, tmp_path)
    service = EvaluationService(settings, agent=_FakeAgent(), encoder=object())

    started = service.start_offline()
    await service.wait_for_idle()
    run = service.get_run(started["id"])

    assert run is not None
    assert run["status"] == "completed"
    assert run["summary"]["tool_selection_accuracy"] == 1.0
    assert run["summary"]["faithfulness"] == 0.9
    assert run["summary"]["answer_relevancy"] == 0.8
    assert run["summary"]["context_recall"] == 0.7
    assert run["summary"]["cases"] == 3
    assert run["summary"]["avg_latency_ms"] is not None
    assert len(run["cases"]) == 3
    assert all("_contexts" not in case for case in run["cases"])
    assert _FakeEvaluator.calls == [(2, True)]
    assert (tmp_path / "evaluations" / f"{started['id']}.json").exists()


class _FakeObservability:
    enabled = True

    def fetch_trace_observations(self, trace_id: str, *, limit: int = 200):
        return [
            {
                "name": "tool.search_bidding_knowledge",
                "type": "tool",
                "input": {"query": "投标保证金"},
                "output": {"content": "成交结果公告后5个工作日内退还投标保证金。"},
            }
        ]


async def _seed_online_user(db, username: str, *, trace_id: str, latency: int):
    async with db.session() as session:
        user = await repo.create_user(
            session, username=username, email=f"{username}@example.com", password_hash="x"
        )
        conversation = await repo.create_conversation(session, user_id=user.id)
        await repo.add_message(
            session, conversation=conversation, role="user", content="投标保证金多久退？"
        )
        await repo.add_message(
            session,
            conversation=conversation,
            role="assistant",
            content="成交后5个工作日内退还。",
            sources=[{"question": "投标保证金退还期限？", "snippet": "成交后5个工作日内退还。"}],
            timings={"total_ms": latency, "thinking_ms": 0, "tool_ms": 100, "generation_ms": 1134},
            provider="fake",
            model="fake-model",
            trace_id=trace_id,
        )
        return str(user.id)


async def test_online_run_covers_all_users_and_judges_tools(
    tmp_path, settings_factory, db, monkeypatch
):
    monkeypatch.setattr("bidding_qa_chatbot.evaluation.runner.RagasEvaluator", _FakeEvaluator)
    _FakeEvaluator.calls.clear()
    _FakeEvaluator.judge_calls.clear()
    settings = _eval_settings(settings_factory, tmp_path)
    service = EvaluationService(
        settings, db=db, encoder=object(), observability=_FakeObservability()
    )
    await _seed_online_user(db, "alice", trace_id="a" * 32, latency=1234)
    await _seed_online_user(db, "bob", trace_id="b" * 32, latency=2345)

    started = service.start_online()
    await service.wait_for_idle()
    run = service.get_run(started["id"])

    assert run is not None and run["status"] == "completed"
    assert {case["username"] for case in run["cases"]} == {"alice", "bob"}
    assert run["summary"]["tool_selection_accuracy"] == 1.0
    assert run["summary"]["context_recall"] == 0.7
    assert run["summary"]["faithfulness"] == 0.9
    assert run["summary"]["answer_relevancy"] == 0.8
    assert run["summary"]["avg_latency_ms"] == 1789.5
    assert _FakeEvaluator.calls == [(2, True)]
    assert len(_FakeEvaluator.judge_calls) == 2
    assert [case["reference"] for case in run["cases"]] == [
        "判分模型生成的参考答案",
        "判分模型生成的参考答案",
    ]

    alice = service.get_run(started["id"], username="alice")
    assert alice is not None
    assert len(alice["cases"]) == 1
    assert alice["summary"]["cases"] == 1
    assert alice["filtered_by"] == "alice"
    assert service.get_run(started["id"], username="nobody") is None
    assert [u["username"] for u in service.list_users()] == ["alice", "bob"]
    assert service.list_runs(username="alice")


async def test_online_run_takes_latest_n_per_user(tmp_path, settings_factory, db, monkeypatch):
    """限制是「每个用户 N 条」，不是全表 N 条。"""
    monkeypatch.setattr("bidding_qa_chatbot.evaluation.runner.RagasEvaluator", _FakeEvaluator)
    settings = _eval_settings(settings_factory, tmp_path, evaluate_online_limit=1)
    service = EvaluationService(
        settings, db=db, encoder=object(), observability=_FakeObservability()
    )

    async with db.session() as session:
        for username, count in (("alice", 2), ("bob", 1)):
            user = await repo.create_user(
                session,
                username=username,
                email=f"{username}@example.com",
                password_hash="x",
            )
            conversation = await repo.create_conversation(session, user_id=user.id)
            for index in range(count):
                await repo.add_message(
                    session,
                    conversation=conversation,
                    role="user",
                    content=f"{username}-q{index}",
                )
                await repo.add_message(
                    session,
                    conversation=conversation,
                    role="assistant",
                    content=f"{username}-a{index}",
                    trace_id=f"{username}-trace-{index}",
                )

    started = service.start_online()
    await service.wait_for_idle()
    run = service.get_run(started["id"])

    assert run is not None and run["status"] == "completed"
    assert len(run["cases"]) == 2, "每个用户各取 1 条，共 2 条"
    assert {case["username"] for case in run["cases"]} == {"alice", "bob"}


async def test_online_run_requires_langfuse(tmp_path, settings_factory, db):
    settings = _eval_settings(settings_factory, tmp_path)
    service = EvaluationService(settings, db=db, encoder=object())
    with pytest.raises(Exception, match="Langfuse"):
        service.start_online()


async def test_disabled_evaluation_rejects_start(tmp_path, settings_factory):
    settings = _eval_settings(settings_factory, tmp_path, evaluate_enabled=False)
    service = EvaluationService(settings, agent=_FakeAgent(), encoder=object())
    assert service.enabled is False
    with pytest.raises(EvaluationDisabled):
        service.start_offline()
