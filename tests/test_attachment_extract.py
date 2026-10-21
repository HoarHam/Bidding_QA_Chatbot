"""附件格式解析：图片直接 OCR；PDF 逐页渲染；DOCX 正文 + 内嵌图片。

OCR 用 FakeOCR；PDF/DOCX 在测试内现场生成，不依赖外部文件。
"""

import io
import uuid

import pymupdf
import pytest
from docx import Document

from bidding_qa_chatbot.attachments.extract import extract_attachment_text
from bidding_qa_chatbot.attachments.store import AttachmentError, AttachmentRecord
from bidding_qa_chatbot.llm.ocr import OCRUnavailable
from tests.xlsx_builder import make_png

PNG = make_png()


class FakeOCR:
    def __init__(self, texts=None, error: Exception | None = None) -> None:
        self.texts = texts if texts is not None else ["识别文本"]
        self.error = error
        self.calls: list[tuple[int, str]] = []

    async def describe_image(self, data: bytes, media_type: str = "image/png") -> str:
        self.calls.append((len(data), media_type))
        if self.error is not None:
            raise self.error
        return self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]


def _record(tmp_path, kind: str, filename: str, content_type: str) -> AttachmentRecord:
    path = tmp_path / "stored.bin"
    path.write_bytes(b"x")
    return AttachmentRecord(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        filename=filename,
        content_type=content_type,
        size=1,
        kind=kind,
        suffix=".bin",
        path=path,
    )


def _pdf_bytes(pages: int = 2, *, encrypted: bool = False) -> bytes:
    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_text((72, 72), f"Page {index + 1}")
    if encrypted:
        data = document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user"
        )
    else:
        data = document.tobytes()
    document.close()
    return data


def _docx_bytes(text: str = "正文", *, images: int = 0) -> bytes:
    document = Document()
    if text:
        document.add_paragraph(text)
    for _ in range(images):
        document.add_picture(io.BytesIO(PNG))
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


async def test_image_ocr(tmp_path, settings_factory):
    record = _record(tmp_path, "image", "photo.png", "image/png")
    ocr = FakeOCR(["识别出的文字"])

    text = await extract_attachment_text(record, PNG, ocr=ocr, settings=settings_factory())

    assert "【图片附件：photo.png】" in text
    assert "识别出的文字" in text
    assert ocr.calls == [(len(PNG), "image/png")]


async def test_image_empty_text_is_error(tmp_path, settings_factory):
    record = _record(tmp_path, "image", "photo.png", "image/png")
    ocr = FakeOCR([""])

    with pytest.raises(AttachmentError) as excinfo:
        await extract_attachment_text(record, PNG, ocr=ocr, settings=settings_factory())

    assert excinfo.value.status_code == 422


async def test_pdf_pages_are_ocrd(tmp_path, settings_factory):
    record = _record(tmp_path, "pdf", "招标文件.pdf", "application/pdf")
    ocr = FakeOCR(["第一页内容", "第二页内容"])

    text = await extract_attachment_text(
        record, _pdf_bytes(2), ocr=ocr, settings=settings_factory()
    )

    assert "【PDF 第 1 页：招标文件.pdf】" in text
    assert "第一页内容" in text
    assert "【PDF 第 2 页：招标文件.pdf】" in text
    assert "第二页内容" in text
    assert len(ocr.calls) == 2
    assert all(media_type == "image/png" for _, media_type in ocr.calls)


async def test_pdf_page_limit(tmp_path, settings_factory):
    record = _record(tmp_path, "pdf", "long.pdf", "application/pdf")
    ocr = FakeOCR(["第1页"])

    text = await extract_attachment_text(
        record,
        _pdf_bytes(3),
        ocr=ocr,
        settings=settings_factory(attachment_pdf_max_pages=1),
    )

    assert len(ocr.calls) == 1
    assert "仅解析前 1 页" in text


async def test_encrypted_pdf_rejected(tmp_path, settings_factory):
    record = _record(tmp_path, "pdf", "secret.pdf", "application/pdf")

    with pytest.raises(AttachmentError) as excinfo:
        await extract_attachment_text(
            record, _pdf_bytes(1, encrypted=True), ocr=FakeOCR(), settings=settings_factory()
        )

    assert excinfo.value.status_code == 422
    assert "加密" in str(excinfo.value)


async def test_docx_text_and_images(tmp_path, settings_factory):
    record = _record(
        tmp_path,
        "docx",
        "方案.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    ocr = FakeOCR(["图片里的文字"])

    text = await extract_attachment_text(
        record, _docx_bytes("正文第一段", images=1), ocr=ocr, settings=settings_factory()
    )

    assert "【文档正文：方案.docx】" in text
    assert "正文第一段" in text
    assert "【文档内图片 1：" in text
    assert "图片里的文字" in text
    assert len(ocr.calls) == 1


async def test_docx_image_limit(tmp_path, settings_factory):
    record = _record(tmp_path, "docx", "a.docx", "application/octet-stream")
    ocr = FakeOCR(["图"])

    await extract_attachment_text(
        record,
        _docx_bytes("正文", images=2),
        ocr=ocr,
        settings=settings_factory(attachment_docx_max_images=1),
    )

    assert len(ocr.calls) == 1


async def test_empty_docx_is_error(tmp_path, settings_factory):
    record = _record(tmp_path, "docx", "empty.docx", "application/octet-stream")

    with pytest.raises(AttachmentError) as excinfo:
        await extract_attachment_text(
            record, _docx_bytes("", images=0), ocr=FakeOCR(), settings=settings_factory()
        )

    assert excinfo.value.status_code == 422


async def test_ocr_unavailable_maps_to_503(tmp_path, settings_factory):
    record = _record(tmp_path, "image", "a.png", "image/png")
    ocr = FakeOCR(error=OCRUnavailable("模型未启动"))

    with pytest.raises(AttachmentError) as excinfo:
        await extract_attachment_text(record, PNG, ocr=ocr, settings=settings_factory())

    assert excinfo.value.status_code == 503
    assert "模型未启动" in str(excinfo.value)


async def test_text_is_truncated(tmp_path, settings_factory):
    record = _record(tmp_path, "image", "a.png", "image/png")
    ocr = FakeOCR(["x" * 200])

    text = await extract_attachment_text(
        record, PNG, ocr=ocr, settings=settings_factory(max_attachment_chars=50)
    )

    assert "已截断" in text
    assert len(text) < 200
