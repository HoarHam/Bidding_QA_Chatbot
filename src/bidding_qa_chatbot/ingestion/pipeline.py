"""入库编排：Excel → 图片 → 视觉转录 → 分块 → 向量化 → Qdrant。

幂等性由三件事共同保证：
1. 确定性 point id（uuid5）→ 重复 upsert 收敛而非堆积；
2. 视觉结果按图片 sha1 缓存 → 重跑不再调用视觉模型（省钱且更快）；
3. 入库完成后清理语料中已不存在的点。

视觉解析失败**不会**中断入库：记为 `vision_status="failed"` 并继续。
第三方视觉 API 不应该成为整条链路的单点故障。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from qdrant_client import models as qm

from bidding_qa_chatbot.config import Settings, get_settings
from bidding_qa_chatbot.embeddings.dense import DenseEncoder
from bidding_qa_chatbot.ingestion import chunker, reader, xlsx_images
from bidding_qa_chatbot.ingestion.ids import dataset_name_from_path
from bidding_qa_chatbot.retrieval import bm25 as bm25_mod
from bidding_qa_chatbot.retrieval.qdrant_store import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    delete_stale,
    ensure_collection,
    set_collection_meta,
    upsert_chunks,
)

logger = logging.getLogger(__name__)

DEFAULT_XLSX = Path("data/processed/问题答案样例数据0113.xlsx")


class SupportsDescribeImage(Protocol):
    async def describe_image(self, data: bytes, media_type: str = "image/png") -> str: ...


@dataclass(slots=True)
class IngestReport:
    dataset: str
    rows: int
    chunks: int
    images: int
    vision_ok: int
    vision_failed: int
    deleted_stale: int
    bm25_terms: int
    collection: str


def _load_vision_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("视觉缓存损坏，将重建: %s", path)
        return {}


def _save_vision_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


async def _transcribe_images(
    images: list[xlsx_images.EmbeddedImage],
    vision: SupportsDescribeImage | None,
    cache: dict[str, str],
    *,
    refresh: bool,
) -> tuple[dict[str, str], int, int]:
    """返回 (sha1 → 转录文本, 成功数, 失败数)。"""
    descriptions: dict[str, str] = {}
    ok = failed = 0

    for img in images:
        cached = None if refresh else cache.get(img.sha1)
        if cached:
            descriptions[img.sha1] = cached
            ok += 1
            continue

        if vision is None:
            failed += 1
            continue

        try:
            text = await vision.describe_image(img.data)
        except Exception as exc:  # noqa: BLE001 — 视觉失败不能阻断入库
            logger.warning("图片 %s 解析失败，跳过: %s", img.name, exc)
            failed += 1
            continue

        if text.strip():
            descriptions[img.sha1] = text.strip()
            cache[img.sha1] = text.strip()
            ok += 1
        else:
            failed += 1

    return descriptions, ok, failed


async def run_ingest(
    qdrant_client: Any,
    xlsx_path: Path | None = None,
    settings: Settings | None = None,
    *,
    encoder: DenseEncoder | None = None,
    vision: SupportsDescribeImage | None = None,
    recreate: bool = False,
    skip_vision: bool = False,
) -> IngestReport:
    settings = settings or get_settings()
    xlsx_path = xlsx_path or DEFAULT_XLSX
    if not xlsx_path.exists():
        raise FileNotFoundError(f"语料不存在: {xlsx_path}")

    dataset = dataset_name_from_path(xlsx_path)
    rows = reader.read_qa_rows(xlsx_path)
    logger.info("读取 %d 条问答 (%s)", len(rows), xlsx_path.name)

    images = xlsx_images.extract_images(xlsx_path)
    logger.info("提取到 %d 张内嵌图片", len(images))

    by_index: dict[int, list[xlsx_images.EmbeddedImage]] = {}
    for img in images:
        by_index.setdefault(img.to_data_index(), []).append(img)

    # 图片落盘，供前端在「参考来源」里展示原图。
    if images:
        target = settings.images_dir / dataset
        xlsx_images.save_images(images, target)

    cache = _load_vision_cache(settings.vision_cache_path)
    if skip_vision:
        descriptions, vision_ok, vision_failed = {}, 0, len(images)
    else:
        descriptions, vision_ok, vision_failed = await _transcribe_images(
            images, vision, cache, refresh=False
        )
        _save_vision_cache(settings.vision_cache_path, cache)

    chunks = [
        chunker.build_chunk(
            row,
            dataset,
            by_index.get(row.data_index, []),
            descriptions,
            vision_status="failed"
            if (by_index.get(row.data_index) and not descriptions)
            else "none",
        )
        for row in rows
    ]

    # ---- 向量化 ----
    encoder = encoder or DenseEncoder(settings)
    dense_vectors = encoder.encode_documents([c.text for c in chunks])

    corpus_sha1 = _corpus_sha1(c.text for c in chunks)
    artifact = bm25_mod.build_artifact([c.text for c in chunks], corpus_sha1)
    bm25_mod.save_artifact(artifact, settings.extracted_dir / "bm25.json")
    logger.info("BM25 词表 %d 个词条", len(artifact.vocab))

    points = [
        qm.PointStruct(
            id=c.point_id,
            vector={
                DENSE_VECTOR: dense_vectors[i],
                SPARSE_VECTOR: bm25_mod.encode_document(c.text, artifact),
            },
            payload={
                "chunk_id": c.chunk_id,
                "dataset": c.dataset,
                "excel_row": c.excel_row,
                "question": c.question,
                "answer": c.answer,
                "text": c.text,
                "source_label": c.source_label,
                "images": chunker.build_image_payload(c),
                "vision_status": c.vision_status,
                "content_sha1": c.content_sha1,
                "token_len": c.token_len_estimate,
                "bm25_vocab_sha1": artifact.vocab_sha1,
            },
        )
        for i, c in enumerate(chunks)
    ]

    await ensure_collection(qdrant_client, settings, recreate=recreate)
    await upsert_chunks(qdrant_client, points, settings)
    deleted = await delete_stale(qdrant_client, {c.chunk_id for c in chunks}, settings)
    await set_collection_meta(
        qdrant_client,
        {
            "bm25_vocab_sha1": artifact.vocab_sha1,
            "corpus_sha1": corpus_sha1,
            "embed_model": settings.embed_model_name,
            "embed_dim": settings.qdrant_vector_size,
        },
        settings,
    )

    _write_corpus_cache(settings.corpus_cache_path, chunks)

    return IngestReport(
        dataset=dataset,
        rows=len(rows),
        chunks=len(chunks),
        images=len(images),
        vision_ok=vision_ok,
        vision_failed=vision_failed,
        deleted_stale=deleted,
        bm25_terms=len(artifact.vocab),
        collection=settings.qdrant_collection,
    )


def _corpus_sha1(texts: Any) -> str:
    import hashlib

    digest = hashlib.sha1()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _write_corpus_cache(path: Path, chunks: list[chunker.Chunk]) -> None:
    """把分块落成 jsonl，便于离线调试与检索质量排查。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "excel_row": c.excel_row,
                        "question": c.question,
                        "answer": c.answer,
                        "text": c.text,
                        "source_label": c.source_label,
                        "images": chunker.build_image_payload(c),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
