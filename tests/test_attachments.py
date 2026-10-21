"""附件存储与上下文组装：魔法校验、所有权、清理、追问注入。"""

import io
import os
import uuid

import pytest
from docx import Document

from bidding_qa_chatbot.attachments.service import (
    PreparedAttachment,
    build_attachment_context,
    prepare_attachments,
)
from bidding_qa_chatbot.attachments.store import (
    AttachmentError,
    AttachmentStore,
    classify_upload,
    sniff_image_type,
)
from tests.xlsx_builder import make_png

PNG = make_png()
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 20
WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 10
PDF = b"%PDF-1.4\n%fake\n"


def _docx_bytes(text: str = "正文") -> bytes:
    document = Document()
    document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


DOCX = _docx_bytes()


class FakeOCR:
    def __init__(self, text: str = "识别文本") -> None:
        self.text = text
        self.calls: list[tuple[int, str]] = []

    async def describe_image(self, data: bytes, media_type: str = "image/png") -> str:
        self.calls.append((len(data), media_type))
        return self.text


def _store(tmp_path, *, max_bytes: int = 1024 * 1024) -> AttachmentStore:
    return AttachmentStore(tmp_path / "uploads", max_bytes=max_bytes)


# ---------------------------------------------------------------- 分类


@pytest.mark.parametrize(
    ("filename", "data", "kind", "content_type"),
    [
        ("a.png", PNG, "image", "image/png"),
        ("a.JPG", JPEG, "image", "image/jpeg"),
        ("a.jpeg", JPEG, "image", "image/jpeg"),
        ("a.webp", WEBP, "image", "image/webp"),
        ("a.pdf", PDF, "pdf", "application/pdf"),
        (
            "a.docx",
            DOCX,
            "docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
    ],
    ids=["png", "jpg", "jpeg", "webp", "pdf", "docx"],
)
def test_classify_accepts_supported_types(filename, data, kind, content_type):
    assert classify_upload(filename, data)[:2] == (kind, content_type)


def test_sniff_image_type():
    assert sniff_image_type(PNG) == "image/png"
    assert sniff_image_type(JPEG) == "image/jpeg"
    assert sniff_image_type(WEBP) == "image/webp"
    assert sniff_image_type(PDF) is None


def test_extension_mismatch_rejected():
    with pytest.raises(AttachmentError) as excinfo:
        classify_upload("a.pdf", PNG)
    assert excinfo.value.status_code == 415


def test_disguised_executable_rejected():
    with pytest.raises(AttachmentError) as excinfo:
        classify_upload("a.png", b"MZ\x90\x00executable")
    assert excinfo.value.status_code == 415
    assert excinfo.value.code == "unsupported_attachment"


def test_broken_docx_rejected():
    with pytest.raises(AttachmentError):
        classify_upload("a.docx", b"PK\x03\x04not-a-real-zip")


# ---------------------------------------------------------------- 存储


async def test_save_load_roundtrip_sanitizes_filename(tmp_path):
    store = _store(tmp_path)
    user_id = uuid.uuid4()

    record = await store.save(user_id=user_id, filename="../../evil.png", data=PNG)

    assert record.filename == "evil.png"
    assert record.path.name == f"{record.id}.png"
    assert "evil" not in str(record.path)
    loaded = await store.load(user_id=user_id, attachment_id=record.id)
    assert loaded.filename == "evil.png"
    assert await store.read(loaded) == PNG
    sidecar = record.path.with_suffix(".json")
    assert sidecar.exists()


async def test_owner_isolation(tmp_path):
    store = _store(tmp_path)
    owner = uuid.uuid4()
    record = await store.save(user_id=owner, filename="a.png", data=PNG)

    with pytest.raises(AttachmentError) as excinfo:
        await store.load(user_id=uuid.uuid4(), attachment_id=record.id)

    assert excinfo.value.status_code == 404


async def test_size_limit(tmp_path):
    store = _store(tmp_path, max_bytes=10)
    with pytest.raises(AttachmentError) as excinfo:
        await store.save(user_id=uuid.uuid4(), filename="a.png", data=PNG)
    assert excinfo.value.status_code == 413


async def test_empty_file_rejected(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(AttachmentError) as excinfo:
        await store.save(user_id=uuid.uuid4(), filename="a.png", data=b"")
    assert excinfo.value.status_code == 400


async def test_read_missing_file(tmp_path):
    store = _store(tmp_path)
    user_id = uuid.uuid4()
    record = await store.save(user_id=user_id, filename="a.png", data=PNG)
    record.path.unlink()

    with pytest.raises(AttachmentError) as excinfo:
        await store.read(record)
    assert excinfo.value.status_code == 404


async def test_purge_expired(tmp_path):
    store = _store(tmp_path)
    user_id = uuid.uuid4()
    record = await store.save(user_id=user_id, filename="a.png", data=PNG)
    sidecar = record.path.with_suffix(".json")
    old = 1_000_000.0
    os.utime(sidecar, (old, old))

    removed = await store.purge_expired(user_id=user_id, retention_days=7)

    assert removed == 1
    assert not record.path.exists()
    assert not sidecar.exists()


# ---------------------------------------------------------------- 上下文


async def test_prepare_attachments(tmp_path, settings_factory):
    store = _store(tmp_path)
    user_id = uuid.uuid4()
    record = await store.save(user_id=user_id, filename="a.png", data=PNG)
    ocr = FakeOCR("图片里的文字")

    prepared = await prepare_attachments(
        settings=settings_factory(),
        store=store,
        ocr=ocr,
        user_id=user_id,
        attachment_ids=[str(record.id)],
    )

    assert len(prepared) == 1
    assert "图片里的文字" in prepared[0].ocr_text
    assert "【图片附件：a.png】" in prepared[0].ocr_text
    assert ocr.calls and ocr.calls[0][1] == "image/png"


async def test_prepare_rejects_multiple_and_invalid_ids(tmp_path, settings_factory):
    store = _store(tmp_path)
    with pytest.raises(AttachmentError) as excinfo:
        await prepare_attachments(
            settings=settings_factory(),
            store=store,
            ocr=FakeOCR(),
            user_id=uuid.uuid4(),
            attachment_ids=["a", "b"],
        )
    assert excinfo.value.status_code == 422

    with pytest.raises(AttachmentError) as excinfo:
        await prepare_attachments(
            settings=settings_factory(),
            store=store,
            ocr=FakeOCR(),
            user_id=uuid.uuid4(),
            attachment_ids=["not-a-uuid"],
        )
    assert excinfo.value.status_code == 400


def test_build_context_injects_current_and_prior():
    prepared = [
        PreparedAttachment(
            id="cur",
            filename="current.png",
            content_type="image/png",
            size=1,
            kind="image",
            ocr_text="本轮内容",
        )
    ]
    prior_rows = [
        [
            {
                "id": "old",
                "filename": "old.pdf",
                "content_type": "application/pdf",
                "size": 2,
                "kind": "pdf",
                "ocr_text": "上一轮内容",
            },
            {"id": "cur", "filename": "current.png", "ocr_text": "本轮内容"},
        ]
    ]

    ctx = build_attachment_context(
        prepared, prior_rows, max_current_chars=1000, max_prior_chars=1000
    )

    assert ctx.has_text is True
    assert "本轮内容" in (ctx.current_text or "")
    assert "上一轮内容" in (ctx.prior_text or "")
    assert ctx.prior_text.count("本轮内容") == 0  # 当前附件不重复注入
    assert "本轮内容" in ctx.persisted[0]["ocr_text"]


def test_build_context_truncates():
    prepared = [
        PreparedAttachment(
            id="a",
            filename="a.png",
            content_type="image/png",
            size=1,
            kind="image",
            ocr_text="x" * 500,
        )
    ]
    ctx = build_attachment_context(prepared, None, max_current_chars=100, max_prior_chars=100)
    assert "已截断" in (ctx.current_text or "")
    assert ctx.prior_text is None
