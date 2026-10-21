"""把附件解析为文本。

- 图片：直接交给 GLM-OCR；
- PDF：PyMuPDF 按 200 DPI 渲染每一页（最多 N 页）后逐页 OCR；
- DOCX：python-docx 提取正文/表格，再用 zipfile 读出 `word/media/*` 图片逐张 OCR。

所有格式的输出都会加显式标签并做长度截断，最终作为**不可信数据**进入提示词。
"""

from __future__ import annotations

import asyncio
import io
import logging
import zipfile
from typing import Protocol

from bidding_qa_chatbot.attachments.store import (
    AttachmentError,
    AttachmentRecord,
    sniff_image_type,
)
from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.llm.ocr import OCRUnavailable

logger = logging.getLogger(__name__)

#: 单个 DOCX 内嵌图片的解压上限（防 zip bomb）
_DOCX_IMAGE_MAX_BYTES = 8 * 1024 * 1024
_DOCX_MEDIA_TOTAL_MAX_BYTES = 30 * 1024 * 1024


class OCRLike(Protocol):
    async def describe_image(self, data: bytes, media_type: str = "image/png") -> str: ...


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n…（附件内容过长，已截断）"


async def extract_attachment_text(
    record: AttachmentRecord,
    data: bytes,
    *,
    ocr: OCRLike,
    settings: Settings,
) -> str:
    """解析附件并返回带标签的文本。失败抛 `AttachmentError`。"""
    try:
        if record.kind == "image":
            text = await _extract_image(record, data, ocr)
        elif record.kind == "pdf":
            text = await _extract_pdf(record, data, ocr, settings)
        else:
            text = await _extract_docx(record, data, ocr, settings)
    except AttachmentError:
        raise
    except OCRUnavailable as exc:
        raise AttachmentError("ocr_unavailable", str(exc), status_code=503) from exc
    except Exception as exc:  # noqa: BLE001 — 解析失败统一转成可展示错误
        logger.exception("附件解析失败: id=%s kind=%s", record.id, record.kind)
        raise AttachmentError(
            "attachment_parse_failed",
            f"无法解析附件「{record.filename}」（{type(exc).__name__}）",
            status_code=422,
        ) from exc

    text = _truncate(text, settings.max_attachment_chars)
    if not text:
        raise AttachmentError(
            "attachment_parse_failed",
            f"未从附件「{record.filename}」中识别到可用文字",
            status_code=422,
        )
    return text


async def _extract_image(record: AttachmentRecord, data: bytes, ocr: OCRLike) -> str:
    text = (await ocr.describe_image(data, record.content_type)).strip()
    if not text:
        return ""
    return f"【图片附件：{record.filename}】\n{text}"


def _render_pdf_pages(data: bytes, max_pages: int) -> tuple[list[bytes], int, bool]:
    """同步渲染 PDF（CPU 密集，由调用方放到线程里）。返回 (页图片, 总页数, 是否截断)。"""
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as document:
        if document.needs_pass:
            raise AttachmentError(
                "attachment_parse_failed", "PDF 已加密，无法解析", status_code=422
            )
        total = document.page_count
        page_count = min(total, max_pages)
        pages: list[bytes] = []
        for index in range(page_count):
            page = document.load_page(index)
            pixmap = page.get_pixmap(dpi=200, colorspace=pymupdf.csRGB, alpha=False)
            pages.append(pixmap.tobytes("png"))
        return pages, total, total > page_count


async def _extract_pdf(
    record: AttachmentRecord, data: bytes, ocr: OCRLike, settings: Settings
) -> str:
    pages, total, truncated = await asyncio.to_thread(
        _render_pdf_pages, data, settings.attachment_pdf_max_pages
    )
    blocks: list[str] = []
    for index, page_png in enumerate(pages, start=1):
        text = (await ocr.describe_image(page_png, "image/png")).strip()
        if text:
            blocks.append(f"【PDF 第 {index} 页：{record.filename}】\n{text}")
    if not blocks:
        return ""
    if truncated:
        blocks.append(f"（PDF 共 {total} 页，仅解析前 {settings.attachment_pdf_max_pages} 页）")
    return "\n\n".join(blocks)


def _extract_docx_parts(data: bytes, max_images: int) -> tuple[str, list[tuple[str, bytes]]]:
    """同步提取 DOCX 正文与内嵌图片（由调用方放到线程里）。"""
    from docx import Document

    lines: list[str] = []
    document = Document(io.BytesIO(data))
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            lines.append(text)
    for table in document.tables:
        for row in table.rows:
            cells = [" ".join(cell.text.split()) for cell in row.cells]
            if any(cells):
                lines.append(" | ".join(cells))

    images: list[tuple[str, bytes]] = []
    total_bytes = 0
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            if not info.filename.startswith("word/media/"):
                continue
            if len(images) >= max_images:
                break
            if info.file_size > _DOCX_IMAGE_MAX_BYTES:
                logger.warning("跳过过大的 DOCX 内嵌图片: %s", info.filename)
                continue
            total_bytes += info.file_size
            if total_bytes > _DOCX_MEDIA_TOTAL_MAX_BYTES:
                logger.warning("DOCX 内嵌图片总量超限，停止提取")
                break
            payload = archive.read(info)
            if sniff_image_type(payload) is None:
                # EMF/WMF 等矢量格式 GLM-OCR 读不了，跳过而不是误读
                continue
            images.append((info.filename.rsplit("/", 1)[-1], payload))
    return "\n".join(lines), images


async def _extract_docx(
    record: AttachmentRecord, data: bytes, ocr: OCRLike, settings: Settings
) -> str:
    body, images = await asyncio.to_thread(
        _extract_docx_parts, data, settings.attachment_docx_max_images
    )
    blocks: list[str] = []
    if body.strip():
        blocks.append(f"【文档正文：{record.filename}】\n{body.strip()}")
    for index, (name, payload) in enumerate(images, start=1):
        text = (await ocr.describe_image(payload)).strip()
        if text:
            blocks.append(f"【文档内图片 {index}：{name}】\n{text}")
    return "\n\n".join(blocks)
