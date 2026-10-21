"""附件存储与校验。

布局：`data/uploads/<user_id>/<uuid><canonical_suffix>` + 同名 `.json` sidecar。
- 用户文件名只用于展示，**绝不进入文件系统路径**；
- 所有权在读取时校验，不匹配一律当作不存在（404），避免探测他人附件 id；
- data/ 已在 .gitignore 中，且上传目录不挂任何静态路由。
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

AttachmentKind = Literal["image", "pdf", "docx"]

IMAGE_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}
DOCX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

_IMAGE_SUFFIXES = {
    ".png": ("image/png", ".png"),
    ".jpg": ("image/jpeg", ".jpg"),
    ".jpeg": ("image/jpeg", ".jpg"),
    ".webp": ("image/webp", ".webp"),
}


class AttachmentError(Exception):
    """附件相关错误。`status_code` 直接映射到 HTTP 响应。"""

    def __init__(self, code: str, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class AttachmentRecord:
    id: uuid.UUID
    user_id: uuid.UUID
    filename: str
    content_type: str
    size: int
    kind: AttachmentKind
    suffix: str
    path: Path


def sniff_image_type(data: bytes) -> str | None:
    """按魔数识别图片类型；仅支持 GLM-OCR 能直接吃的三种。"""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _is_docx(data: bytes) -> bool:
    if not data.startswith(b"PK\x03\x04"):
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile:
        return False
    return "[Content_Types].xml" in names and "word/document.xml" in names


def _clean_filename(filename: str) -> str:
    """只保留纯文件名用于展示；拒绝空名与控制字符。"""
    name = (filename or "").replace("\\", "/").split("/")[-1].strip()
    name = "".join(ch for ch in name if ch >= " " and ch != "\x7f")
    if not name or name in (".", ".."):
        raise AttachmentError("invalid_filename", "附件文件名无效", status_code=400)
    return name[:200]


def classify_upload(filename: str, data: bytes) -> tuple[AttachmentKind, str, str]:
    """魔数 + 扩展名双重校验，返回 `(kind, content_type, canonical_suffix)`。"""
    if not data:
        raise AttachmentError("empty_attachment", "附件为空", status_code=400)

    suffix = Path(filename).suffix.lower()
    image_type = sniff_image_type(data)
    if image_type is not None:
        expected = _IMAGE_SUFFIXES.get(suffix)
        if expected is None or expected[0] != image_type:
            raise AttachmentError(
                "unsupported_attachment", "附件扩展名与文件内容不一致", status_code=415
            )
        return "image", expected[0], expected[1]

    if data.startswith(b"%PDF-"):
        if suffix != ".pdf":
            raise AttachmentError(
                "unsupported_attachment", "附件扩展名与文件内容不一致", status_code=415
            )
        return "pdf", "application/pdf", ".pdf"

    if _is_docx(data):
        if suffix != ".docx":
            raise AttachmentError(
                "unsupported_attachment", "附件扩展名与文件内容不一致", status_code=415
            )
        return "docx", DOCX_CONTENT_TYPE, ".docx"

    raise AttachmentError(
        "unsupported_attachment",
        "不支持的附件类型（仅 PNG/JPEG/WebP/PDF/DOCX）",
        status_code=415,
    )


class AttachmentStore:
    def __init__(self, root: Path, *, max_bytes: int) -> None:
        self.root = Path(root)
        self.max_bytes = max_bytes

    def _user_dir(self, user_id: uuid.UUID) -> Path:
        return self.root / str(user_id)

    async def save(self, *, user_id: uuid.UUID, filename: str, data: bytes) -> AttachmentRecord:
        if not data:
            raise AttachmentError("empty_attachment", "附件为空", status_code=400)
        if len(data) > self.max_bytes:
            limit_mb = self.max_bytes // (1024 * 1024)
            raise AttachmentError(
                "attachment_too_large", f"附件超过 {limit_mb}MB 上限", status_code=413
            )

        display_name = _clean_filename(filename)
        kind, content_type, suffix = classify_upload(display_name, data)
        attachment_id = uuid.uuid4()
        user_dir = self._user_dir(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        path = user_dir / f"{attachment_id}{suffix}"
        sidecar = user_dir / f"{attachment_id}.json"
        temp = user_dir / f"{attachment_id}.tmp"

        meta = {
            "id": str(attachment_id),
            "user_id": str(user_id),
            "filename": display_name,
            "content_type": content_type,
            "size": len(data),
            "kind": kind,
            "suffix": suffix,
            "created_at": time.time(),
        }
        try:
            temp.write_bytes(data)
            os.replace(temp, path)
            sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            for candidate in (temp, path, sidecar):
                with contextlib.suppress(OSError):
                    candidate.unlink(missing_ok=True)
            raise

        logger.info("附件已保存: id=%s kind=%s size=%d", attachment_id, kind, len(data))
        return AttachmentRecord(
            id=attachment_id,
            user_id=user_id,
            filename=display_name,
            content_type=content_type,
            size=len(data),
            kind=kind,
            suffix=suffix,
            path=path,
        )

    async def load(self, *, user_id: uuid.UUID, attachment_id: uuid.UUID) -> AttachmentRecord:
        user_dir = self._user_dir(user_id)
        sidecar = user_dir / f"{attachment_id}.json"
        not_found = AttachmentError("attachment_not_found", "附件不存在或已过期", status_code=404)
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise not_found from None
        if not isinstance(meta, dict):
            raise not_found
        if str(meta.get("user_id")) != str(user_id) or str(meta.get("id")) != str(attachment_id):
            # 所有权不匹配与「不存在」返回同一错误，避免探测
            raise not_found
        suffix = str(meta.get("suffix") or "")
        path = user_dir / f"{attachment_id}{suffix}"
        if not path.is_file():
            raise not_found
        return AttachmentRecord(
            id=attachment_id,
            user_id=user_id,
            filename=str(meta.get("filename") or "附件"),
            content_type=str(meta.get("content_type") or ""),
            size=int(meta.get("size") or 0),
            kind=meta.get("kind") if meta.get("kind") in ("image", "pdf", "docx") else "image",
            suffix=suffix,
            path=path,
        )

    async def read(self, record: AttachmentRecord) -> bytes:
        try:
            return record.path.read_bytes()
        except OSError as exc:
            raise AttachmentError(
                "attachment_not_found", "附件不存在或已过期", status_code=404
            ) from exc

    async def purge_expired(self, *, user_id: uuid.UUID, retention_days: int) -> int:
        """删除该用户目录下超过保留期的数据与 sidecar（历史 OCR 文本已在 DB，不受影响）。"""
        user_dir = self._user_dir(user_id)
        if not user_dir.is_dir():
            return 0
        cutoff = time.time() - retention_days * 86400
        removed = 0
        for sidecar in user_dir.glob("*.json"):
            try:
                if sidecar.stat().st_mtime >= cutoff:
                    continue
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                data_path = user_dir / f"{meta.get('id')}{meta.get('suffix')}"
                with contextlib.suppress(OSError):
                    data_path.unlink(missing_ok=True)
                sidecar.unlink()
                removed += 1
            except (OSError, ValueError):
                continue
        return removed
