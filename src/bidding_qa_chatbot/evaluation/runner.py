"""Agent 综合评测服务：离线数据集 + 在线真实对话。

## 两种模式

- **离线**：逐条跑 `data/processed/evaluate_cases.json`，由被测 Agent 自己选工具、
  检索、生成回答；之后用 Ragas 评 Faithfulness / AnswerRelevancy / ContextRecall，
  另外计算工具选择准确率与平均延迟。
- **在线**：取当前用户最近的真实问答（PostgreSQL 历史），用 Ragas 评
  Faithfulness / AnswerRelevancy / ContextRecall（使用**判分模型基于 Langfuse 工具输出生成的 `reference_answer`**），
  另外计算工具选择准确率（判分模型读取 Langfuse 中的实际工具调用（名称/参数/输出）与问题，
  判断工具选择是否合理，并给出期望工具；结果写入
  `case.tool_selection_correct`，汇总为准确率）
  与平均延迟（读取历史消息落库的 `messages.timings.total_ms`；旧消息没有 timings 时该条不计入均值）。

评测是后台任务：`/api/evaluate/*` 立即返回 run_id，页面轮询进度；结果写入
`data/evaluations/<run_id>.json`，服务重启后仍可回看。
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from bidding_qa_chatbot.agent.agent import (
    AgentDone,
    RagAgent,
    ToolCallNotice,
    ToolResultNotice,
)
from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.db import repository as repo
from bidding_qa_chatbot.evaluation.dataset import (
    WEB_TOOLS,
    EvaluationCase,
    EvaluationDatasetError,
    load_cases,
    tool_selection_matches,
)
from bidding_qa_chatbot.evaluation.ragas_runner import (
    METRIC_KEYS,
    RagasEvaluator,
    RagasUnavailable,
)
from bidding_qa_chatbot.evaluation.store import EvaluationStore, summary_of
from bidding_qa_chatbot.llm.base import StreamEvent
from bidding_qa_chatbot.llm.prompts import build_agent_messages
from bidding_qa_chatbot.log import redact

logger = logging.getLogger(__name__)

METRIC_LABELS: dict[str, str] = {
    "tool_selection_accuracy": "工具选择准确率",
    "faithfulness": "忠诚度",
    "answer_relevancy": "相关性",
    "context_recall": "召回率",
    "avg_latency_ms": "平均延迟",
}

#: 送入 Ragas 的单个工具上下文上限。判分模型单次输出有 token 上限，
#: 上下文/回答过长会让 Faithfulness 的 NLI JSON 被截断（表现为该指标缺失）。
MAX_CONTEXT_CHARS = 3000
MAX_CASE_ANSWER_CHARS = 8000


class EvaluationError(Exception):
    status_code = 400


class EvaluationDisabled(EvaluationError):
    status_code = 503


class EvaluationBusy(EvaluationError):
    status_code = 409


class EvaluationUnavailable(EvaluationError):
    status_code = 503


class EvaluationNotFound(EvaluationError):
    status_code = 404


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _clip(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    value = text.strip()
    return value if len(value) <= limit else value[:limit] + "…"


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 4)


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """从用例明细汇总指标；用于整轮结果与「按用户筛选」后的结果。"""
    succeeded = [c for c in cases if not c.get("error")]
    latencies = [float(c["latency_ms"]) for c in cases if c.get("latency_ms") is not None]
    tool_values = [
        1.0 if c["tool_selection_correct"] else 0.0
        for c in cases
        if c.get("tool_selection_correct") is not None
    ]
    scored_cases = [
        c for c in cases if any(v is not None for v in (c.get("scores") or {}).values())
    ]
    summary = {
        "tool_selection_accuracy": _mean(tool_values) if tool_values else None,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "cases": len(cases),
        "succeeded": len(succeeded),
        "failed": len(cases) - len(succeeded),
        "ragas_scored": len(scored_cases),
    }
    for key in METRIC_KEYS:
        summary[key] = _mean([(c.get("scores") or {}).get(key) for c in cases])
    return summary


DEFAULT_AGENT_TOOLS = [
    "search_bidding_knowledge",
    "query_bidding_graph",
    "search_web_keyword",
    "search_web_semantic",
]


class EvaluationService:
    def __init__(
        self,
        settings: Settings,
        *,
        agent: RagAgent | None = None,
        db: Any = None,
        encoder: Any = None,
        observability: Any = None,
    ) -> None:
        self.settings = settings
        self.agent = agent
        self.db = db
        self.encoder = encoder
        self.observability = observability
        self.store = EvaluationStore(settings)
        self._active_run_id: str | None = None
        self._runs: dict[str, dict[str, Any]] = {}
        self._tasks: set[asyncio.Task] = set()
        self.reason = self._disabled_reason()

    # ------------------------------------------------------------ 可用性
    def _disabled_reason(self) -> str | None:
        if not self.settings.evaluate_enabled:
            return "EVALUATE_ENABLED=false"
        if not self.settings.evaluate_judge_configured:
            return "判分模型密钥未配置"
        if importlib.util.find_spec("ragas") is None:
            return "Ragas 未安装"
        return None

    @property
    def enabled(self) -> bool:
        return self.reason is None

    @property
    def judge_same_source(self) -> bool:
        return self.settings.evaluate_judge_provider == self.settings.llm_provider

    def config(self) -> dict[str, Any]:
        dataset_exists = self.settings.evaluate_cases_path.exists()
        dataset_cases: int | None = None
        if dataset_exists:
            try:
                dataset_cases = len(load_cases(self.settings.evaluate_cases_path))
            except EvaluationDatasetError:
                dataset_cases = None
        warnings: list[str] = []
        if self.judge_same_source:
            warnings.append(
                "判分 provider 与被测 provider 相同；Faithfulness/相关性可能系统性虚高。"
            )
        return {
            "enabled": self.enabled,
            "reason": self.reason,
            "judge": {
                "provider": self.settings.evaluate_judge_provider,
                "model": self.settings.evaluate_judge_model,
                "same_source": self.judge_same_source,
            },
            "evaluated": {
                "provider": self.settings.llm_provider,
                "model": self.settings.model_for(self.settings.llm_provider, "fast"),
            },
            "agent_available": bool(self.agent is not None and self.settings.agent_enabled),
            "db_available": self.db is not None,
            "langfuse_available": bool(
                self.observability is not None and getattr(self.observability, "enabled", False)
            ),
            "encoder_available": self.encoder is not None,
            "dataset": {
                "name": self.settings.evaluate_cases_filename,
                "exists": dataset_exists,
                "cases": dataset_cases,
            },
            "metrics": [{"key": key, "label": label} for key, label in METRIC_LABELS.items()],
            "history_limit": self.settings.evaluate_history_limit,
            "online_limit": self.settings.evaluate_online_limit,
            "warnings": warnings,
        }

    # ------------------------------------------------------------ 启动运行
    def start_offline(self) -> dict[str, Any]:
        self._ensure_enabled()
        if self.agent is None or not self.settings.agent_enabled:
            raise EvaluationUnavailable("Agent 未启用，无法运行离线评测")
        try:
            cases = load_cases(self.settings.evaluate_cases_path)
        except EvaluationDatasetError as exc:
            raise EvaluationUnavailable(str(exc)) from exc
        self._ensure_idle()
        run = self._new_run("offline", total=len(cases))
        run["dataset"] = self.settings.evaluate_cases_filename
        self._start_task(run, self._run_offline(run, cases))
        return summary_of(run)

    def start_online(self) -> dict[str, Any]:
        self._ensure_enabled()
        if self.db is None:
            raise EvaluationUnavailable("数据库不可用，无法读取历史对话")
        if self.observability is None or not getattr(self.observability, "enabled", False):
            raise EvaluationUnavailable("在线评测需要 Langfuse 可观测性已启用")
        self._ensure_idle()
        run = self._new_run("online", total=0)
        run["dataset"] = "recent_conversations_all_users"
        self._start_task(run, self._run_online(run))
        return summary_of(run)

    def _ensure_enabled(self) -> None:
        if not self.enabled:
            raise EvaluationDisabled(self.reason or "评测未启用")

    def _ensure_idle(self) -> None:
        if self._active_run_id is not None:
            raise EvaluationBusy("已有评测在运行，请等待当前任务结束")

    def _new_run(self, mode: str, *, total: int) -> dict[str, Any]:
        now = datetime.now(UTC)
        run_id = now.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        run = {
            "id": run_id,
            "mode": mode,
            "status": "running",
            "created_at": now.isoformat(),
            "finished_at": None,
            "judge": {
                "provider": self.settings.evaluate_judge_provider,
                "model": self.settings.evaluate_judge_model,
            },
            "evaluated": {
                "provider": self.settings.llm_provider,
                "model": self.settings.model_for(self.settings.llm_provider, "fast"),
            },
            "progress": {
                "phase": "starting",
                "completed": 0,
                "total": total,
            },
            "summary": {
                "tool_selection_accuracy": None,
                "faithfulness": None,
                "answer_relevancy": None,
                "context_recall": None,
                "avg_latency_ms": None,
                "cases": 0,
                "succeeded": 0,
                "failed": 0,
            },
            "metric_errors": [],
            "cases": [],
            "error": None,
        }
        self._runs[run_id] = run
        return run

    def _start_task(self, run: dict[str, Any], coro: Any) -> None:
        self._active_run_id = run["id"]
        task = asyncio.create_task(coro, name=f"evaluate-{run['id']}")
        self._tasks.add(task)

        def _done(_: asyncio.Task) -> None:
            self._tasks.discard(task)
            self._active_run_id = None
            # 只保留最近 50 个内存结果；磁盘历史不受影响。
            if len(self._runs) > 50:
                for old in sorted(self._runs, key=lambda k: self._runs[k]["created_at"])[:-50]:
                    self._runs.pop(old, None)

        task.add_done_callback(_done)

    # ------------------------------------------------------------ 离线
    async def _run_offline(self, run: dict[str, Any], cases: list[EvaluationCase]) -> None:
        evaluator: RagasEvaluator | None = None
        try:
            run["progress"].update(phase="preparing", total=len(cases))
            evaluator = await asyncio.to_thread(RagasEvaluator, self.settings, self.encoder)
            for case in cases:
                run["progress"]["phase"] = "generating"
                result, contexts = await self._run_agent_case(case)
                # 上下文只用于本轮 Ragas，不把大段原文写进历史 JSON。
                result["contexts_count"] = len(contexts)
                result["_contexts"] = contexts
                run["cases"].append(result)
                run["progress"]["completed"] = len(run["cases"])

            eligible = [
                (i, case, result)
                for i, (case, result) in enumerate(zip(cases, run["cases"], strict=True))
                if case.reference
                and result.get("answer")
                and result.get("_contexts")
                and not result.get("error")
            ]
            samples = [
                self._sample(
                    question=result["question"],
                    answer=result["answer"],
                    contexts=result["_contexts"],
                    reference=case.reference,
                )
                for _, case, result in eligible
            ]
            run["progress"]["phase"] = "judging"
            batch = await evaluator.evaluate(samples, include_context_recall=True)
            for (case_index, _, result), scores in zip(eligible, batch.scores, strict=True):
                result["scores"] = scores
                run["cases"][case_index] = result
            run["metric_errors"] = batch.errors
            self._finalize(run, batch.summary, ragas_scored=len(eligible))
        except RagasUnavailable as exc:
            run["metric_errors"].append(redact(str(exc)))
            self._finalize(run, {key: None for key in METRIC_KEYS})
            # 判分器在生成用例之前就失败：没有可用结果，明确标记为失败。
            if not run["cases"]:
                run["status"] = "failed"
                run["error"] = "判分器不可用"
        except Exception as exc:  # noqa: BLE001 — 评测任务自身失败要落盘
            logger.exception("离线评测失败 %s", run["id"])
            run["status"] = "failed"
            run["error"] = redact(f"{type(exc).__name__}: {exc}")[:500]
        finally:
            if evaluator is not None:
                evaluator.close()
            self._finish(run)

    async def _run_agent_case(self, case: EvaluationCase) -> tuple[dict[str, Any], list[str]]:
        assert self.agent is not None
        graph_enabled = "query_bidding_graph" in self.agent.registry
        messages = build_agent_messages(
            case.question,
            [],
            graph_enabled=graph_enabled,
            web_search_enabled=False,
        )
        allowed_tools = set(self.agent.registry.names) - WEB_TOOLS
        actual_tools: list[str] = []
        contexts: list[str] = []
        answer_parts: list[str] = []
        provider: str | None = None
        model: str | None = None
        finish_reason: str | None = None
        error: str | None = None
        started = time.perf_counter()

        try:
            async for item in self.agent.astream(messages, "fast", allowed_tools=allowed_tools):
                if isinstance(item, ToolCallNotice):
                    actual_tools.append(item.name)
                elif isinstance(item, ToolResultNotice):
                    contexts.extend(self._contexts_from_notice(item))
                elif isinstance(item, StreamEvent):
                    if item.kind == "content":
                        answer_parts.append(item.text)
                elif isinstance(item, AgentDone):
                    provider = item.provider
                    model = item.model
                    finish_reason = item.finish_reason
        except Exception as exc:  # noqa: BLE001 — 单条失败不终止整轮
            error = redact(f"{type(exc).__name__}: {exc}")[:500]

        latency_ms = int((time.perf_counter() - started) * 1000)
        answer = _clip("".join(answer_parts), MAX_CASE_ANSWER_CHARS)
        deduped_contexts = list(dict.fromkeys(c for c in contexts if c.strip()))
        tool_correct = bool(error is None and tool_selection_matches(case, actual_tools))
        result = {
            "index": case.index,
            "question": case.question,
            "reference": case.reference,
            "expected_tool": case.original_expected_tool,
            "expected_tool_normalized": case.expected_tool,
            "expected_no_tool": case.expected_no_tool,
            "actual_tools": actual_tools,
            "tool_selection_correct": tool_correct,
            "answer": answer,
            "latency_ms": latency_ms,
            "provider": provider,
            "model": model,
            "finish_reason": finish_reason,
            "scores": {key: None for key in METRIC_KEYS},
            "error": error,
        }
        return result, deduped_contexts

    @staticmethod
    def _contexts_from_notice(notice: ToolResultNotice) -> list[str]:
        if not notice.ok:
            return []
        values: list[str] = []
        if notice.content.strip():
            values.append(_clip(notice.content))
        # 兼容没有 content 的旧工具/测试替身：退回结构化来源摘要。
        if not values:
            for item in notice.sources:
                if isinstance(item, dict):
                    question = str(item.get("question") or "").strip()
                    snippet = str(item.get("snippet") or item.get("answer") or "").strip()
                    text = "\n".join(part for part in (question, snippet) if part)
                    if text:
                        values.append(_clip(text))
            for item in notice.web_sources:
                if isinstance(item, dict):
                    title = str(item.get("title") or "").strip()
                    snippet = str(item.get("snippet") or "").strip()
                    text = "\n".join(part for part in (title, snippet) if part)
                    if text:
                        values.append(_clip(text))
        return list(dict.fromkeys(values))

    def _finalize(
        self,
        run: dict[str, Any],
        ragas_summary: dict[str, float | None],
        *,
        ragas_scored: int | None = None,
    ) -> None:
        cases = run["cases"]
        summary = run["summary"]
        summary.update(summarize_cases(cases))
        # 整轮汇总以 Ragas 的 batch 结果为准（与逐用例分数一致）。
        summary["faithfulness"] = ragas_summary.get("faithfulness")
        summary["answer_relevancy"] = ragas_summary.get("answer_relevancy")
        summary["context_recall"] = ragas_summary.get("context_recall")
        if ragas_scored is not None:
            summary["ragas_scored"] = ragas_scored
        run["status"] = "completed"
        run["finished_at"] = _now_iso()
        run["progress"].update(
            phase="done",
            completed=len(cases),
            total=len(cases),
        )

    # ------------------------------------------------------------ 在线
    async def _run_online(self, run: dict[str, Any]) -> None:
        evaluator: RagasEvaluator | None = None
        try:
            run["progress"].update(phase="loading_history")
            pairs = await self._recent_qa_pairs()
            if not pairs:
                raise EvaluationUnavailable("没有找到带 Langfuse trace 的最近问答")
            run["progress"]["total"] = len(pairs)
            evaluator = await asyncio.to_thread(RagasEvaluator, self.settings, self.encoder)
            available_tools = self._available_tools()

            for user_msg, assistant_msg, conversation, user in pairs:
                timings = assistant_msg.timings if isinstance(assistant_msg.timings, dict) else {}
                latency = timings.get("total_ms")
                result = {
                    "index": len(run["cases"]),
                    "user_id": str(user.id),
                    "username": user.username,
                    "conversation_id": str(conversation.id),
                    "conversation_title": conversation.title,
                    "message_id": str(assistant_msg.id),
                    "trace_id": assistant_msg.trace_id,
                    "question": user_msg.content,
                    "reference": "",
                    "expected_tool": None,
                    "expected_tool_normalized": None,
                    "expected_no_tool": False,
                    "actual_tools": [],
                    "tool_selection_correct": None,
                    "answer": _clip(assistant_msg.content or "", MAX_CASE_ANSWER_CHARS),
                    "latency_ms": int(latency) if latency is not None else None,
                    "provider": assistant_msg.provider,
                    "model": assistant_msg.model,
                    "finish_reason": None,
                    "judge_reasoning": None,
                    "scores": {key: None for key in METRIC_KEYS},
                    "error": None,
                    "contexts_count": 0,
                    "_contexts": [],
                }
                try:
                    observations = await self._fetch_trace(assistant_msg.trace_id)
                    if not observations:
                        raise EvaluationUnavailable("Langfuse 中未找到该 trace 的 observation")
                    tools, contexts = self._evidence_from_observations(observations)
                    if not contexts:
                        contexts = self._contexts_from_stored_sources(assistant_msg.sources)
                    judgement = await evaluator.judge_online_case(
                        question=user_msg.content,
                        answer=assistant_msg.content or "",
                        actual_tools=tools,
                        available_tools=available_tools,
                    )
                    expected = judgement.expected_tool.strip()
                    reference = judgement.reference_answer.strip()
                    result.update(
                        expected_tool=expected or None,
                        expected_tool_normalized=expected or None,
                        expected_no_tool=expected.lower()
                        in {"ask_clarification", "none", "无", "不需要工具"},
                        actual_tools=[
                            str(tool.get("name") or "") for tool in tools if tool.get("name")
                        ],
                        tool_selection_correct=bool(judgement.tool_selection_correct),
                        reference=reference,
                        judge_reasoning=judgement.reasoning,
                        contexts_count=len(contexts),
                        _contexts=contexts,
                    )
                    if not reference:
                        result["error"] = "判分模型未生成参考答案"
                except Exception as exc:  # noqa: BLE001 — 单条失败继续
                    result["error"] = redact(f"{type(exc).__name__}: {exc}")[:500]
                run["cases"].append(result)
                run["progress"]["completed"] = len(run["cases"])

            eligible = [
                (i, result)
                for i, result in enumerate(run["cases"])
                if result.get("reference")
                and result.get("answer")
                and result.get("_contexts")
                and not result.get("error")
            ]
            run["progress"]["phase"] = "judging"
            samples = [
                self._sample(
                    question=result["question"],
                    answer=result["answer"],
                    contexts=result["_contexts"],
                    reference=result["reference"],
                )
                for _, result in eligible
            ]
            batch = await evaluator.evaluate(samples, include_context_recall=True)
            for (case_index, result), scores in zip(eligible, batch.scores, strict=True):
                result["scores"] = scores
                run["cases"][case_index] = result
            run["metric_errors"] = batch.errors
            self._finalize(run, batch.summary, ragas_scored=len(eligible))
        except RagasUnavailable as exc:
            run["metric_errors"].append(redact(str(exc)))
            self._finalize(run, {key: None for key in METRIC_KEYS})
            if not run["cases"]:
                run["status"] = "failed"
                run["error"] = "判分器不可用"
        except EvaluationUnavailable as exc:
            run["status"] = "failed"
            run["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("在线评测失败 %s", run["id"])
            run["status"] = "failed"
            run["error"] = redact(f"{type(exc).__name__}: {exc}")[:500]
        finally:
            if evaluator is not None:
                evaluator.close()
            self._finish(run)

    async def _recent_qa_pairs(self) -> list[tuple[Any, Any, Any, Any]]:
        assert self.db is not None
        limit = max(1, self.settings.evaluate_online_limit)
        async with self.db.session() as session:
            return await repo.recent_qa_pairs_per_user(session, limit_per_user=limit)

    def _available_tools(self) -> list[str]:
        if self.agent is not None:
            return sorted(self.agent.registry.names)
        return list(DEFAULT_AGENT_TOOLS)

    async def _fetch_trace(self, trace_id: str | None) -> list[dict[str, Any]]:
        if not trace_id or self.observability is None:
            return []
        for attempt in range(3):
            observations = await asyncio.to_thread(
                self.observability.fetch_trace_observations, trace_id, limit=200
            )
            if observations:
                return observations
            if attempt < 2:
                await asyncio.sleep(1.0)
        return []

    @classmethod
    def _evidence_from_observations(
        cls, observations: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        tools: list[dict[str, Any]] = []
        contexts: list[str] = []
        for observation in observations:
            name = str(observation.get("name") or "")
            obs_type = str(observation.get("type") or "").lower()
            if obs_type not in {"tool", "retriever"} and not name.startswith("tool."):
                continue
            tool_name = name.removeprefix("tool.") or name or "unknown"
            output = cls._observation_text(observation.get("output"))
            tools.append(
                {
                    "name": tool_name,
                    "arguments": cls._maybe_json(observation.get("input")),
                    "output": output,
                }
            )
            if output:
                contexts.append(_clip(output))
        return tools, list(dict.fromkeys(contexts))

    @staticmethod
    def _maybe_json(value: Any) -> Any:
        """Langfuse 的 input/output 以原始字符串返回，能解析时还原成结构。"""
        if not isinstance(value, str):
            return value
        text = value.strip()
        if text[:1] not in "[{":
            return value
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value

    @classmethod
    def _observation_text(cls, value: Any) -> str:
        if isinstance(value, str):
            decoded = cls._maybe_json(value)
            if decoded is not value:
                return cls._observation_text(decoded)
            return value
        if isinstance(value, dict):
            for key in ("content", "output", "text", "answer", "result"):
                if value.get(key):
                    text = cls._observation_text(value[key])
                    if text:
                        return text
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, (list, tuple)):
            return "\n".join(text for item in value if (text := cls._observation_text(item)))
        return "" if value is None else str(value)

    @staticmethod
    def _contexts_from_stored_sources(raw_sources: Any) -> list[str]:
        values: list[str] = []
        for item in raw_sources or []:
            if not isinstance(item, dict):
                continue
            if item.get("kind") == "web":
                text = "\n".join(
                    part
                    for part in (
                        str(item.get("title") or "").strip(),
                        str(item.get("snippet") or "").strip(),
                    )
                    if part
                )
            else:
                text = "\n".join(
                    part
                    for part in (
                        str(item.get("question") or "").strip(),
                        str(item.get("snippet") or "").strip(),
                    )
                    if part
                )
            if text:
                values.append(_clip(text))
        return list(dict.fromkeys(values))

    @staticmethod
    def _sample(
        *,
        question: str,
        answer: str,
        contexts: list[str],
        reference: str | None,
    ) -> Any:
        from ragas import SingleTurnSample

        return SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference,
        )

    # ------------------------------------------------------------ 查询
    def list_runs(self, username: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, self.settings.evaluate_history_limit)
        summaries: dict[str, dict[str, Any]] = {}
        for run in self._runs.values():
            if username and (
                run.get("mode") != "online"
                or not any(str(case.get("username")) == username for case in run.get("cases", []))
            ):
                continue
            summaries[run["id"]] = summary_of(run)
        for item in self.store.list_summaries(limit=limit, username=username):
            summaries.setdefault(str(item.get("id")), item)
        ordered = sorted(
            summaries.values(),
            key=lambda r: str(r.get("created_at") or ""),
            reverse=True,
        )
        return ordered[:limit]

    def list_users(self) -> list[dict[str, Any]]:
        # 只从磁盘结果统计：内存中的运行完成后也已落盘，避免同一 run 被计两次。
        counts = self.store.all_usernames()
        return [{"username": name, "cases": count} for name, count in sorted(counts.items())]

    def get_run(self, run_id: str, username: str | None = None) -> dict[str, Any] | None:
        run = self._runs.get(run_id) or self.store.load(run_id)
        if run is None:
            return None
        cases = run.get("cases", [])
        if username:
            if run.get("mode") != "online":
                return None
            cases = [case for case in cases if str(case.get("username")) == username]
            if not cases:
                return None
        public = {k: v for k, v in run.items() if k not in {"user_id", "cases"}}
        public["cases"] = [
            {k: v for k, v in case.items() if not k.startswith("_")} for case in cases
        ]
        public["users"] = sorted(
            {str(case.get("username")) for case in cases if case.get("username")}
        )
        if username:
            public["filtered_by"] = username
            # 指标卡展示的是当前筛选用户的子集，而不是全量 run 的均值。
            public["summary"] = summarize_cases(cases)
        return public

    async def wait_for_idle(self) -> None:
        """等待当前后台评测结束（测试与优雅退出用）。"""
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def _finish(self, run: dict[str, Any]) -> None:
        if run.get("status") == "running":
            run["status"] = "failed"
            run["error"] = run.get("error") or "评测异常结束"
        run["finished_at"] = run.get("finished_at") or _now_iso()
        run["progress"]["phase"] = "done"
        # `_contexts` 只用于本轮判分，不写进历史文件（体积大且无展示价值）。
        for case in run.get("cases", []):
            case.pop("_contexts", None)
        with contextlib.suppress(Exception):
            self.store.save(run)
