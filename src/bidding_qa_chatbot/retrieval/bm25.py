"""jieba 分词 + BM25 稀疏编码（需求 7）。

只依赖 qdrant-client 的 `SparseVector` 数据结构，不引入 `rank_bm25`：
那个库的最新版停在 2022 年，且它的 API 返回「文档打分」而非
Qdrant 稀疏向量需要的 `(term_index, weight)` 对，帮不上忙。

## 为什么自己算 IDF 而不用 Qdrant 的 `Modifier.IDF`

Qdrant 支持服务端算 IDF，代码更少，但会带来两个代价：离线单测（`QdrantClient(":memory:")`）
与线上行为不再一致；且增量入库时集合统计量在过程中是变化的。语料只有 80 条、
静态不变，客户端算反而更可控、可测。

## IDF 公式为什么带 `+1` 的壳

用 `ln(1 + (N - df + 0.5)/(df + 0.5))`（BM25+ 形式）而非 `ln(N/df)`：
后者在词出现在**全部**文档时等于 0，出现在**过半**文档时为**负数**，
而稀疏向量里的负权重会直接把排序反过来。招投标语料里「招标」这类词
几乎每篇都有，这不是假想问题。

## 词表稳定性（最容易踩的坑）

一旦词表被重建（换了语料或换了 jieba 词典），所有 term→index 映射整体平移，
已入库的稀疏向量全部失效——但它**不会报错**，因为 dense 一侧仍然正常工作，
检索结果看起来「有点像样」。所以：

1. 词表用 `sorted(unique_terms)` 生成，保证同样语料永远得到同样映射；
2. 把 `vocab_sha1` 写进每个 point 的 payload 与集合元数据；
3. 查询前比对，不一致就告警并降级为纯 dense（`SPARSE_STRICT=true` 时直接报错）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jieba
from qdrant_client import models as qm

from bidding_qa_chatbot.log import redact

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: BM25 自由参数
K1 = 1.5
B = 0.75

_PUNCT_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)

#: 中文停用词。只收真正无区分度的功能词，宁少勿多——
#: 招投标领域的词（如「招标」「投标」「采购」）虽然高频，但绝不能删。
STOPWORDS = frozenset(
    [
        "的",
        "了",
        "和",
        "与",
        "及",
        "或",
        "在",
        "是",
        "为",
        "对",
        "从",
        "到",
        "就",
        "都",
        "也",
        "还",
        "而",
        "但",
        "并",
        "等",
        "中",
        "上",
        "下",
        "之",
        "其",
        "该",
        "这",
        "那",
        "有",
        "无",
        "不",
        "没",
        "可以",
        "应该",
        "应当",
        "需要",
        "必须",
        "进行",
        "按照",
        "根据",
        "关于",
        "对于",
        "由于",
        "因此",
        "所以",
        "但是",
        "以及",
        "或者",
        "并且",
        "如果",
        "那么",
        "一个",
        "一种",
        "我们",
        "你们",
        "他们",
        "什么",
        "怎么",
        "如何",
        "哪些",
        "哪个",
        "是否",
    ]
)


@dataclass(slots=True)
class BM25Artifact:
    """持久化的 BM25 统计量与词表。"""

    schema_version: int
    corpus_sha1: str
    vocab_sha1: str
    n_docs: int
    avgdl: float
    k1: float
    b: float
    vocab: dict[str, int]
    idf: dict[str, float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


@dataclass(slots=True)
class _Stats:
    doc_freq: dict[str, int] = field(default_factory=dict)
    total_len: int = 0
    n_docs: int = 0


def tokenize(text: str) -> list[str]:
    """jieba 搜索引擎模式分词。

    用 `lcut_for_search` 而非 `lcut`：它会把长词再切出子词，
    对「竞争性磋商」「询价文件」这类招投标术语能显著提高召回。
    """
    if not text:
        return []
    tokens: list[str] = []
    for raw in jieba.lcut_for_search(text.lower()):
        tok = raw.strip()
        if not tok or tok in STOPWORDS:
            continue
        if _PUNCT_RE.match(tok):
            continue
        tokens.append(tok)
    return tokens


def _build_artifact(docs_tokens: list[list[str]], corpus_sha1: str) -> BM25Artifact:
    stats = _Stats()
    for tokens in docs_tokens:
        stats.n_docs += 1
        stats.total_len += len(tokens)
        for term in set(tokens):
            stats.doc_freq[term] = stats.doc_freq.get(term, 0) + 1

    # sorted 保证映射可复现，与文档遍历顺序无关。
    vocab = {term: i for i, term in enumerate(sorted(stats.doc_freq))}
    n = stats.n_docs
    idf = {term: math.log(1.0 + (n - df + 0.5) / (df + 0.5)) for term, df in stats.doc_freq.items()}
    vocab_sha1 = hashlib.sha1(
        json.dumps(vocab, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return BM25Artifact(
        schema_version=SCHEMA_VERSION,
        corpus_sha1=corpus_sha1,
        vocab_sha1=vocab_sha1,
        n_docs=n,
        avgdl=(stats.total_len / n) if n else 0.0,
        k1=K1,
        b=B,
        vocab=vocab,
        idf=idf,
    )


def build_artifact(texts: list[str], corpus_sha1: str) -> BM25Artifact:
    """从语料构建词表与 IDF（入库路径）。"""
    return _build_artifact([tokenize(t) for t in texts], corpus_sha1)


def encode_document(text: str, artifact: BM25Artifact) -> qm.SparseVector:
    """把文档编码成稀疏向量，权重为完整 BM25 词权重。"""
    tokens = tokenize(text)
    if not tokens:
        return qm.SparseVector(indices=[], values=[])

    tf: dict[str, int] = {}
    for tok in tokens:
        tf[tok] = tf.get(tok, 0) + 1

    dl = len(tokens)
    avgdl = artifact.avgdl or 1.0
    k1, b = artifact.k1, artifact.b
    denom_norm = k1 * (1 - b + b * dl / avgdl)
    vocabulary = artifact.vocab
    idf = artifact.idf

    pairs: list[tuple[int, float]] = []
    for term, freq in tf.items():
        idx = vocabulary.get(term)
        if idx is None:
            continue
        weight = idf[term] * (freq * (k1 + 1)) / (freq + denom_norm)
        if weight != 0.0:
            pairs.append((idx, weight))

    # Qdrant 要求 indices 唯一且（本地模式）有序。
    pairs.sort()
    return qm.SparseVector(indices=[i for i, _ in pairs], values=[w for _, w in pairs])


def encode_query(query: str, artifact: BM25Artifact) -> qm.SparseVector:
    """把查询编码成稀疏向量，权重为 IDF。

    这样与文档向量做点积，恰好等于 BM25 在词交集上的求和。
    """
    tokens = tokenize(query)
    if not tokens:
        return qm.SparseVector(indices=[], values=[])

    vocabulary = artifact.vocab
    idf = artifact.idf
    pairs: list[tuple[int, float]] = []
    seen: set[str] = set()
    oov = 0
    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        idx = vocabulary.get(tok)
        if idx is None:
            oov += 1
            continue
        pairs.append((idx, idf[tok]))

    if seen and oov == len(seen):
        # 整个查询都在词表之外：稀疏一路会返回空，RRF 自动退化为纯 dense。
        # 这本身也是「语料不覆盖该问题」的信号，值得记一笔。
        logger.info("查询词全部未命中 BM25 词表，稀疏检索将返回空: %s", redact(query[:60]))

    pairs.sort()
    return qm.SparseVector(indices=[i for i, _ in pairs], values=[w for _, w in pairs])


def save_artifact(artifact: BM25Artifact, path: Path) -> None:
    """原子写入：先写临时文件再 replace，避免入库中断留下半截词表。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(artifact.to_json(), encoding="utf-8")
    os.replace(tmp, path)


def load_artifact(path: Path) -> BM25Artifact | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("BM25 词表读取失败，将退化为纯 dense 检索: %s", exc)
        return None
    if raw.get("schema_version") != SCHEMA_VERSION:
        logger.warning("BM25 词表 schema 版本不匹配（%s），忽略", raw.get("schema_version"))
        return None
    return BM25Artifact(**raw)
