"""BM25 稀疏编码测试。

重点覆盖两个「错了也不会报错、只会让排序悄悄变差」的陷阱：
- IDF 出现 0 或负数（`ln(N/df)` 形式在高频词上就会这样）；
- 词表映射不稳定（重建后所有已入库稀疏向量失效）。
"""

from __future__ import annotations

import math

from bidding_qa_chatbot.retrieval import bm25


def test_tokenize_drops_stopwords_and_punctuation():
    tokens = bm25.tokenize("招标文件的  要求是，投标人应当提供资质证明。")
    assert "的" not in tokens
    assert "是" not in tokens
    assert "，" not in tokens
    assert any("招标" in t for t in tokens)


def test_tokenize_empty_returns_empty():
    assert bm25.tokenize("") == []
    assert bm25.tokenize("   。，！  ") == []


def test_search_mode_recall_subwords():
    """搜索引擎模式应切出子词，提高「竞争性磋商」这类术语的召回。"""
    tokens = bm25.tokenize("竞争性磋商文件")
    assert any("磋商" in t for t in tokens)


def test_vocab_is_sorted_and_unique():
    docs = ["招标投标", "投标资质", "资质要求"]
    art = bm25.build_artifact(docs, "sha")
    indices = list(art.vocab.values())
    assert indices == sorted(indices)
    assert len(indices) == len(set(indices))
    assert list(art.vocab) == sorted(art.vocab)


def test_vocab_is_stable_regardless_of_doc_order():
    """同样的文档集合，顺序不同也必须得到同一套映射。"""
    a = bm25.build_artifact(["招标投标", "投标资质"], "sha")
    b = bm25.build_artifact(["投标资质", "招标投标"], "sha")
    assert a.vocab == b.vocab
    assert a.vocab_sha1 == b.vocab_sha1


def test_idf_is_always_positive():
    """用 ln(1 + ...) 形式的原因：高频词不能得到 0 或负权重。

    若用 ln(N/df)，出现在全部 3 篇里的词会得到 0，出现在 2/3 篇里会得到负数，
    负权重会把稀疏向量点积的排序整体反转。
    """
    docs = ["招标", "招标", "招标 投标"]
    art = bm25.build_artifact(docs, "sha")
    assert art.idf["招标"] > 0
    assert all(v > 0 for v in art.idf.values())


def test_idf_rank_order_low_frequency_term_scores_higher():
    docs = ["招标 资质", "招标 资质", "招标 投诉"]
    art = bm25.build_artifact(docs, "sha")
    # 「投诉」只在 1 篇出现，「资质」在 2 篇出现
    assert art.idf["投诉"] > art.idf["资质"]


def test_document_weights_match_bm25_formula():
    """手工核算一篇文档的词权重，确认公式实现正确。"""
    docs = ["甲 乙", "甲 丙", "甲 丁"]
    art = bm25.build_artifact(docs, "sha")

    vec = bm25.encode_document("甲 乙", art)
    weights = dict(zip(vec.indices, vec.values, strict=True))

    n, dl, avgdl, k1, b = 3, 2, 2.0, art.k1, art.b
    expected_idf = math.log(1 + (n - 1 + 0.5) / (1 + 0.5))
    denom = 1 + k1 * (1 - b + b * dl / avgdl)
    expected_weight = expected_idf * (1 * (k1 + 1)) / denom

    assert math.isclose(weights[art.vocab["乙"]], expected_weight, rel_tol=1e-9)


def test_query_weights_are_idf():
    docs = ["甲 乙", "甲 丙", "甲 丁"]
    art = bm25.build_artifact(docs, "sha")
    vec = bm25.encode_query("乙", art)
    assert list(vec.indices) == [art.vocab["乙"]]
    assert vec.values[0] == art.idf["乙"]


def test_query_indices_sorted_and_unique():
    docs = ["甲 乙 丙 丁"]
    art = bm25.build_artifact(docs, "sha")
    vec = bm25.encode_query("丁 甲 丁 乙", art)
    assert list(vec.indices) == sorted(vec.indices)
    assert len(set(vec.indices)) == len(vec.indices)


def test_query_drops_out_of_vocabulary_terms():
    art = bm25.build_artifact(["招标投标"], "sha")
    vec = bm25.encode_query("完全不存在的词汇组合", art)
    assert vec.indices == []
    assert vec.values == []


def test_vocab_sha1_changes_when_corpus_changes():
    """词表指纹必须随语料变化——它是检测「稀疏向量已失效」的唯一依据。"""
    a = bm25.build_artifact(["甲 乙"], "sha")
    b = bm25.build_artifact(["甲 丙"], "sha")
    assert a.vocab_sha1 != b.vocab_sha1


def test_artifact_roundtrip(tmp_path):
    art = bm25.build_artifact(["招标 投标", "资质 要求"], "corpus-sha")
    path = tmp_path / "bm25.json"
    bm25.save_artifact(art, path)
    loaded = bm25.load_artifact(path)

    assert loaded is not None
    assert loaded.vocab == art.vocab
    assert loaded.vocab_sha1 == art.vocab_sha1
    assert loaded.corpus_sha1 == "corpus-sha"


def test_load_missing_returns_none(tmp_path):
    assert bm25.load_artifact(tmp_path / "nope.json") is None


def test_load_rejects_schema_mismatch(tmp_path):
    path = tmp_path / "bm25.json"
    path.write_text('{"schema_version": 999}', encoding="utf-8")
    assert bm25.load_artifact(path) is None
