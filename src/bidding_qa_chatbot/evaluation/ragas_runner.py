"""Ragas 判分适配层。

- 判分 LLM 走智谱/DeepSeek 的 OpenAI 兼容端点，使用 `llm_factory` 的
  Instructor 适配器（Ragas 0.4 的结构化输出路径）；
- AnswerRelevancy 需要嵌入模型，这里复用本地 bge-small-zh（与检索同一模型），
  不额外依赖外部 embedding API；
- Ragas 的导入全部延迟到真正启动评测时，未启用评测时不会增加服务启动开销。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.log import redact

#: 指标 key 与 Ragas 输出列名一一对应。
METRIC_KEYS = ("faithfulness", "answer_relevancy", "context_recall")


class OnlineJudgement(BaseModel):
    """在线评测判分模型的结构化输出。"""

    expected_tool: str = Field(
        description="根据问题应使用的工具名；不需要工具或应澄清时填 ask_clarification"
    )
    tool_selection_correct: bool = Field(
        description="Agent 实际工具选择是否合理（不要求与期望工具字符串完全相同）"
    )
    reference_answer: str = Field(
        description="基于 Langfuse 记录中的检索证据写出的简洁参考答案，用于 ContextRecall"
    )
    reasoning: str = Field(default="", description="一句话说明判断依据")


class RagasUnavailable(RuntimeError):
    """Ragas / 判分模型不可用。"""


@dataclass(slots=True)
class RagasBatch:
    scores: list[dict[str, float | None]] = field(default_factory=list)
    summary: dict[str, float | None] = field(
        default_factory=lambda: {key: None for key in METRIC_KEYS}
    )
    errors: list[str] = field(default_factory=list)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 4)


def _build_local_embeddings(encoder: Any) -> Any:
    """把项目的 DenseEncoder 包成 Ragas 旧版 Embeddings 接口。

    Ragas 的 AnswerRelevancy 把原始问题与生成的问题都当作文本比较；
    两者都是问句，所以统一使用 query 侧编码（带 bge 的指令前缀），
    与「文档侧」前缀混用会引入无意义的分数偏移。
    """
    from ragas.embeddings import BaseRagasEmbeddings

    class LocalQuestionEmbeddings(BaseRagasEmbeddings):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def embed_query(self, text: str) -> list[float]:
            return self.inner.encode_query(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.inner.encode_query(text) for text in texts]

        async def aembed_query(self, text: str) -> list[float]:
            return await asyncio.to_thread(self.embed_query, text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_documents, texts)

    return LocalQuestionEmbeddings(encoder)


class RagasEvaluator:
    """一次评测运行复用的判分器（LLM client、嵌入、缓存）。"""

    def __init__(self, settings: Settings, encoder: Any | None) -> None:
        self.settings = settings
        self.encoder = encoder
        self.errors: list[str] = []
        self._client: Any | None = None
        self._llm: Any | None = None
        self._embeddings: Any | None = None
        self._cache: Any | None = None
        self._max_tokens = max(1024, settings.evaluate_judge_max_tokens)
        self._build()

    def _build(self) -> None:
        # Ragas 匿名遥测默认开启；项目 .env 已关，这里兜底防止测试/其他入口漏配。
        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
        try:
            from openai import OpenAI
            from ragas.cache import DiskCacheBackend
        except Exception as exc:  # noqa: BLE001 — 依赖缺失/版本不兼容
            raise RagasUnavailable(redact(f"Ragas 依赖不可用: {exc}")) from exc

        settings = self.settings
        if settings.evaluate_judge_provider == "zhipu":
            if settings.zhipu_api_key is None:
                raise RagasUnavailable("判分 provider zhipu 缺少 ZHIPU_API_KEY")
            api_key = settings.zhipu_api_key.get_secret_value()
            base_url = settings.zhipu_base_url
        else:
            if settings.deepseek_api_key is None:
                raise RagasUnavailable("判分 provider deepseek 缺少 DEEPSEEK_API_KEY")
            api_key = settings.deepseek_api_key.get_secret_value()
            base_url = settings.deepseek_base_url

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=settings.evaluate_judge_timeout,
            max_retries=0,  # 重试由 Ragas RunConfig/指标层控制
        )
        if settings.evaluate_ragas_cache_dir:
            cache_dir = (
                settings.evaluate_ragas_cache_path
                / f"{settings.evaluate_judge_provider}-{settings.evaluate_judge_model}"
            )
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                self._cache = DiskCacheBackend(str(cache_dir))
            except Exception as exc:  # noqa: BLE001 — 缓存只是加速项
                self.errors.append(redact(f"Ragas 磁盘缓存不可用: {exc}"))

        try:
            self._llm = self._make_llm(self._max_tokens)
        except Exception as exc:  # noqa: BLE001
            raise RagasUnavailable(redact(f"判分 LLM 初始化失败: {exc}")) from exc

        if self.encoder is None:
            self.errors.append("本地嵌入模型未就绪，AnswerRelevancy 将跳过")
        else:
            try:
                self._embeddings = _build_local_embeddings(self.encoder)
            except Exception as exc:  # noqa: BLE001
                self.errors.append(redact(f"本地嵌入适配失败，AnswerRelevancy 将跳过: {exc}"))

    def _make_llm(self, max_tokens: int) -> Any:
        from ragas.llms import llm_factory

        return llm_factory(
            self.settings.evaluate_judge_model,
            client=self._client,
            max_tokens=max_tokens,
            cache=self._cache,
        )

    def _metrics(self, llm: Any, requested: set[str]) -> list[Any]:
        from ragas.metrics import ContextRecall, Faithfulness, ResponseRelevancy

        metrics: list[Any] = [Faithfulness(llm=llm)]
        if "answer_relevancy" in requested and self._embeddings is not None:
            # strictness=1：GLM 的 OpenAI 兼容端点不支持一次返回多个候选，
            # 默认 3 会触发 "1 generations instead of requested 3" 并浪费 token。
            metrics.append(
                ResponseRelevancy(
                    llm=llm,
                    embeddings=self._embeddings,
                    strictness=1,
                )
            )
        if "context_recall" in requested:
            metrics.append(ContextRecall(llm=llm))
        return metrics

    async def judge_online_case(
        self,
        *,
        question: str,
        answer: str,
        actual_tools: list[dict[str, Any]],
        available_tools: list[str],
    ) -> OnlineJudgement:
        """基于 Langfuse 的实际工具调用判断工具选择，并生成参考回答。

        `actual_tools` 每项包含 name / arguments / output（均来自 Langfuse
        observation）；判分模型看不到 Trace 之外的隐式信息。
        """
        if self._llm is None:
            raise RagasUnavailable("判分 LLM 未初始化")

        evidence = []
        for item in actual_tools:
            evidence.append(
                {
                    "name": item.get("name"),
                    "arguments": item.get("arguments"),
                    "output": str(item.get("output") or "")[:3000],
                }
            )
        prompt = f"""你是招投标问答 Agent 的评测裁判。请严格依据下面提供的信息判断工具选择，并写出参考答案。

【用户问题】
{question[:3000]}

【Agent 最终回答】
{answer[:5000]}

【Langfuse 记录中的实际工具调用】
{json.dumps(evidence, ensure_ascii=False, indent=2)}

【当前可用工具】
{json.dumps(available_tools, ensure_ascii=False)}

判断规则：
1. 规范性知识（法规条款、期限、金额门槛、资格条件、评审办法）应使用
   search_bidding_knowledge；
2. 具体交易事实/统计（交易频次、供应商、采购人、金额、地区）应使用
   query_bidding_graph；
3. 时效性外部信息才需要 search_web_keyword / search_web_semantic；
4. 问题信息不足、无法确定对象时，应理解为需要向用户澄清（ask_clarification）；
5. 工具选择正确指“实际调用的工具与问题匹配”，允许为补充信息多调用一个工具；
   如果问题需要澄清却强行调用工具，或工具与问题类型不符，则判错。

另外，请基于工具输出中的证据写一段简洁、事实充分的参考答案（reference_answer），
供召回率评测使用；不要编造工具输出中没有的事实。只返回结构化结果。"""
        # llm_factory 用同步 OpenAI client 构建；Ragas 的 agenerate 明确拒绝
        # 同步 client，因此放到线程里调用 generate，避免阻塞事件循环。
        result = await asyncio.to_thread(self._llm.generate, prompt, OnlineJudgement)
        if isinstance(result, OnlineJudgement):
            return result
        return OnlineJudgement.model_validate(result)

    async def evaluate(
        self,
        samples: list[Any],
        *,
        include_context_recall: bool,
    ) -> RagasBatch:
        batch = RagasBatch(errors=list(self.errors))
        if not samples:
            batch.errors.append("没有可判分的样本（缺少参考/检索上下文/回答）")
            return batch
        if self._llm is None:
            batch.errors.append("判分 LLM 未初始化")
            return batch

        try:
            from ragas import EvaluationDataset, aevaluate
            from ragas.run_config import RunConfig
        except Exception as exc:  # noqa: BLE001
            batch.errors.append(redact(f"导入 Ragas 失败: {exc}"))
            return batch

        requested = {"faithfulness"}
        if self._embeddings is not None:
            requested.add("answer_relevancy")
        if include_context_recall:
            requested.add("context_recall")

        run_config = RunConfig(
            timeout=max(30, int(self.settings.evaluate_judge_timeout)),
            max_retries=2,
            max_workers=2,
        )

        async def run_batch(llm: Any) -> list[dict[str, float | None]]:
            result = await aevaluate(
                dataset=EvaluationDataset(samples=samples),
                metrics=self._metrics(llm, requested),
                llm=llm,
                embeddings=self._embeddings,
                show_progress=False,
                raise_exceptions=False,
                run_config=run_config,
            )
            raw_scores = getattr(result, "scores", None) or []
            rows: list[dict[str, float | None]] = []
            for sample_scores in raw_scores[: len(samples)]:
                rows.append({key: _as_float((sample_scores or {}).get(key)) for key in METRIC_KEYS})
            while len(rows) < len(samples):
                rows.append({key: None for key in METRIC_KEYS})
            return rows

        try:
            batch.scores = await run_batch(self._llm)
        except Exception as exc:  # noqa: BLE001 — Ragas 自身失败不拖垮整轮评测
            batch.errors.append(redact(f"Ragas 评测失败: {exc}"))
            return batch

        failed = {key for key in requested if all(row.get(key) is None for row in batch.scores)}
        # 判分模型（尤其 GLM）的 reasoning 也占 max_tokens；长回答会让
        # Faithfulness 的 NLI JSON 被截断。失败时用更大的输出上限重试一次，
        # 而不是直接把这个指标留空。
        if failed and self._max_tokens < 16384:
            retry_tokens = min(32768, max(16384, self._max_tokens * 2))
            try:
                retry_llm = self._make_llm(retry_tokens)
                retry_rows = await run_batch(retry_llm)
                for row, retry_row in zip(batch.scores, retry_rows, strict=True):
                    for key in failed:
                        if row.get(key) is None and retry_row.get(key) is not None:
                            row[key] = retry_row[key]
                if any(row.get(key) is not None for row in batch.scores for key in failed):
                    batch.errors.append(
                        f"部分指标首次输出被 token 上限截断，已用 {retry_tokens} max_tokens 重试"
                    )
            except Exception as exc:  # noqa: BLE001
                batch.errors.append(redact(f"更大 token 上限重试失败: {exc}"))

        for key in requested:
            batch.summary[key] = mean([row.get(key) for row in batch.scores])

        # Ragas 在单个指标失败时会给 NaN；把失败信息明确展示出来，避免误读为 0 分。
        for key in requested:
            if all(row.get(key) is None for row in batch.scores):
                batch.errors.append(f"{key} 全部用例判分失败（详见服务日志）")

        return batch

    def close(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.close()
