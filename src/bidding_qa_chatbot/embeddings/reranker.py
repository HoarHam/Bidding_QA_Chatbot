"""精排：BAAI/bge-reranker-base（CrossEncoder，需求 6）。

同样禁止顶层导入 `sentence_transformers`——原因见包 docstring。

## 分数量纲提醒

sentence-transformers 对单标签 CrossEncoder 会套一层默认 activation，
因此 `predict()` 返回的是 (0,1) 区间的值而非原始 logit。网上大量文章引用的
logit 阈值（如 >0.5 才算相关）在这里**不适用**。所以 `RERANK_MIN_SCORE`
默认 0.0（即不做截断），先看真实分布再决定阈值。

另外：RRF 分数是排名融合产物，与这里的分数**量纲不同**，两者不可比较。
UI 上只展示精排分，且不称作「置信度」。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from bidding_qa_chatbot.config import Settings, get_settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

#: bge-reranker-base 的位置编码上限，超过会被静默截断。
MAX_LENGTH = 512


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> CrossEncoder:
    import os

    import torch
    from sentence_transformers import CrossEncoder

    from bidding_qa_chatbot.embeddings import load_with_cache_fallback

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    logger.info("加载精排模型 %s (device=%s)", model_name, device)
    return load_with_cache_fallback(
        model_name,
        lambda local_only: CrossEncoder(
            model_name, max_length=MAX_LENGTH, device=device, local_files_only=local_only
        ),
    )


@dataclass(slots=True)
class RankedItem:
    """一条带精排分的候选。"""

    index: int
    score: float


class Reranker:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def enabled(self) -> bool:
        return self.settings.rerank_enabled

    def rerank(
        self, query: str, candidates: list[str], top_k: int | None = None
    ) -> list[RankedItem]:
        """对候选文本打分并按分数降序返回。

        精排输入用的是与入库**完全相同**的文本。若两边文本不一致，
        可能出现「检索得分高、精排得分低」的结果，而原因完全不可见、不可测。
        """
        if not candidates:
            return []

        limit = top_k if top_k is not None else self.settings.rerank_top_k
        model = _load_model(self.settings.rerank_model_name)
        pairs = [(query, text) for text in candidates]
        raw = model.predict(pairs, batch_size=8, show_progress_bar=False)

        ranked = sorted(
            (RankedItem(index=i, score=float(s)) for i, s in enumerate(raw)),
            key=lambda item: item.score,
            reverse=True,
        )

        threshold = self.settings.rerank_min_score
        if threshold > 0.0:
            ranked = [item for item in ranked if item.score >= threshold]

        if ranked:
            logger.debug(
                "精排分数区间 %.4f ~ %.4f（共 %d 条）",
                ranked[-1].score,
                ranked[0].score,
                len(ranked),
            )
        return ranked[:limit]

    def warmup(self) -> None:
        _load_model(self.settings.rerank_model_name)


def clear_model_cache() -> None:
    _load_model.cache_clear()
