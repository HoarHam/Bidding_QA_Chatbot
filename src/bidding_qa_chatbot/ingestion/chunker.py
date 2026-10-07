"""把问答行组装成入库分块。

## 分块单位：一行问答 = 一个分块（共 80 个）

答案本身就是人工整理的、自洽的完整条目，再切分只会破坏语义。
尺寸上也支持这个判断：最长答案 500 字，而 bge 系列的位置编码上限是 512 token，
中文约 1 token/字——最长的一行**已经贴着截断边界**了。切分不但没有收益，
还会把一条完整规则拆散。

唯一的例外是带图的那一行：它的正文只有一句「…招采情况如下：」，
真正的答案在 3 张图里。视觉模型转录出来的文本会**追加**到正文之后，
使这一行变得可检索——这正是需求 6 存在的意义。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from bidding_qa_chatbot.ingestion.ids import (
    content_sha1,
    make_chunk_id,
    make_image_url,
    make_point_id,
    make_source_label,
)
from bidding_qa_chatbot.ingestion.reader import QARow
from bidding_qa_chatbot.ingestion.xlsx_images import EmbeddedImage

logger = logging.getLogger(__name__)

#: 图片转录文本在正文中的分隔标记
VISION_MARKER = "【图片内容】"

#: bge 系列位置编码上限；超出会被静默截断，因此主动告警。
TOKEN_WARN_THRESHOLD = 512


@dataclass(slots=True)
class ChunkImage:
    index: int
    filename: str
    sha1: str
    storage_path: str
    description: str | None = None
    vision_status: str = "pending"

    @property
    def url(self) -> str:
        return ""  # 由 Chunk 填充（需要 chunk_id）


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    point_id: str
    dataset: str
    data_index: int
    excel_row: int
    question: str
    answer: str
    text: str
    source_label: str
    content_sha1: str
    images: list[ChunkImage] = field(default_factory=list)
    vision_status: str = "none"

    @property
    def token_len_estimate(self) -> int:
        """粗略估算 token 数。中文按 1 字≈1 token 估，足够触发告警。"""
        return len(self.text)


def build_chunk(
    row: QARow,
    dataset: str,
    images: list[EmbeddedImage],
    descriptions: dict[str, str] | None = None,
    vision_status: str = "none",
) -> Chunk:
    """组装一个分块。`descriptions` 以图片 sha1 为键。"""
    chunk_id = make_chunk_id(dataset, row.data_index)
    descriptions = descriptions or {}

    text = f"{row.question}\n{row.answer}"
    chunk_images: list[ChunkImage] = []

    if images:
        transcribed: list[str] = []
        for idx, img in enumerate(images, start=1):
            desc = descriptions.get(img.sha1)
            if desc:
                transcribed.append(desc)
            chunk_images.append(
                ChunkImage(
                    index=idx,
                    filename=img.name,
                    sha1=img.sha1,
                    storage_path=f"{chunk_id}/{img.sha1[:8]}-{img.name}",
                    description=desc,
                    vision_status="ok" if desc else vision_status,
                )
            )
        if transcribed:
            # 追加而非替换：保留原文，同时让图片内容进入检索范围。
            text = f"{text}\n{VISION_MARKER}\n" + "\n\n".join(transcribed)

    chunk = Chunk(
        chunk_id=chunk_id,
        point_id=make_point_id(chunk_id),
        dataset=dataset,
        data_index=row.data_index,
        excel_row=row.excel_row,
        question=row.question,
        answer=row.answer,
        text=text,
        source_label=make_source_label(row.excel_row),
        content_sha1=content_sha1(text),
        images=chunk_images,
        vision_status="ok" if (images and descriptions) else vision_status,
    )

    if chunk.token_len_estimate > TOKEN_WARN_THRESHOLD:
        logger.warning(
            "分块 %s 约 %d token，超过 bge 的 %d 上限，嵌入时会被截断",
            chunk.chunk_id,
            chunk.token_len_estimate,
            TOKEN_WARN_THRESHOLD,
        )
    return chunk


def build_image_payload(chunk: Chunk) -> list[dict[str, object]]:
    """产出写入 Qdrant payload 的图片描述列表。"""
    return [
        {
            "index": img.index,
            "filename": img.filename,
            "sha1": img.sha1,
            "storage_path": img.storage_path,
            "description": img.description,
            "vision_status": img.vision_status,
            "url": make_image_url(chunk.chunk_id, img.index),
        }
        for img in chunk.images
    ]
