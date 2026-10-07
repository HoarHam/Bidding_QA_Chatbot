"""Reciprocal Rank Fusion 的纯函数实现。

线上融合走 Qdrant 的 `FusionQuery(Fusion.RRF)`（服务端算）。这里的实现用于两个场景：

1. **测试**——用可以手算的小例子验证融合顺序符合预期，而不用起 Qdrant；
2. **降级**——当稀疏一路不可用（词表缺失/陈旧）时，仍能把两路候选按等价规则融合，
   而不是直接丢掉一路。

Qdrant 的 RRF 用 k=60（与原始论文一致），这里保持相同默认值，
保证降级路径与正常路径的排序口径一致。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

DEFAULT_K = 60


@dataclass(slots=True)
class FusedHit:
    doc_id: str
    score: float
    ranks: dict[str, int]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[tuple[str, Sequence[str]]],
    k: int = DEFAULT_K,
) -> list[FusedHit]:
    """融合多路有序结果。

    参数 `ranked_lists` 是 `[(路由名, [doc_id, ...]), ...]`，每路已按相关性降序。
    返回按融合分降序的 `FusedHit` 列表，score = Σ 1/(k + rank)，rank 从 1 开始。
    """
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for route, doc_ids in ranked_lists:
        for position, doc_id in enumerate(doc_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position)
            ranks.setdefault(doc_id, {})[route] = position

    return [
        FusedHit(doc_id=doc_id, score=score, ranks=ranks[doc_id])
        for doc_id, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ]


def take_ids(hits: Iterable[FusedHit], limit: int) -> list[str]:
    return [hit.doc_id for hit in list(hits)[:limit]]
