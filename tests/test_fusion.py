"""RRF 融合测试（纯函数，可手算验证）。"""

from __future__ import annotations

import pytest

from bidding_qa_chatbot.retrieval.fusion import (
    DEFAULT_K,
    reciprocal_rank_fusion,
    take_ids,
)


def test_single_route_preserves_order():
    hits = reciprocal_rank_fusion([("dense", ["a", "b", "c"])])
    assert [h.doc_id for h in hits] == ["a", "b", "c"]
    assert hits[0].score == pytest.approx(1 / (DEFAULT_K + 1))


def test_scores_match_formula():
    hits = reciprocal_rank_fusion([("dense", ["a", "b"])], k=60)
    assert hits[0].score == pytest.approx(1 / 61)
    assert hits[1].score == pytest.approx(1 / 62)


def test_document_in_both_routes_outranks_single_route_document():
    """两路都命中的文档应排在只有一路命中的文档之前。"""
    hits = reciprocal_rank_fusion(
        [
            ("dense", ["a", "b"]),
            ("sparse", ["b", "c"]),
        ]
    )
    assert hits[0].doc_id == "b"
    assert hits[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_union_contains_all_documents():
    hits = reciprocal_rank_fusion(
        [
            ("dense", ["a", "b"]),
            ("sparse", ["c", "d"]),
        ]
    )
    assert {h.doc_id for h in hits} == {"a", "b", "c", "d"}


def test_records_per_route_ranks():
    hits = reciprocal_rank_fusion(
        [
            ("dense", ["a", "b"]),
            ("sparse", ["b", "a"]),
        ]
    )
    by_id = {h.doc_id: h for h in hits}
    assert by_id["a"].ranks == {"dense": 1, "sparse": 2}
    assert by_id["b"].ranks == {"dense": 2, "sparse": 1}


def test_empty_input_returns_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([("dense", [])]) == []


def test_take_ids_limits_result():
    hits = reciprocal_rank_fusion([("dense", ["a", "b", "c"])])
    assert take_ids(hits, 2) == ["a", "b"]


def test_k_parameter_shifts_scores_but_not_order():
    routs = [("dense", ["a", "b", "c"])]
    small = reciprocal_rank_fusion(routs, k=1)
    large = reciprocal_rank_fusion(routs, k=1000)
    assert [h.doc_id for h in small] == [h.doc_id for h in large]
    assert small[0].score > large[0].score
