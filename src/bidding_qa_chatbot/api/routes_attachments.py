"""附件上传接口。

**只做校验与存储，不解析**：OCR 在发送消息时执行（见 routes_chat），
这样解析失败可以返回真正的 HTTP 错误且不会创建会话/落用户消息。
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, File, Request, UploadFile, status
from fastapi.responses import JSONResponse

from bidding_qa_chatbot.attachments import AttachmentError, AttachmentStore
from bidding_qa_chatbot.auth.deps import CurrentUser
from bidding_qa_chatbot.schemas import ApiError, ApiErrorResponse, AttachmentOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/attachments", tags=["attachments"])


def _error(exc: AttachmentError) -> JSONResponse:
    payload = ApiErrorResponse(error=ApiError(code=exc.code, message=str(exc)))
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump(mode="json"))


def _get_store(request: Request) -> AttachmentStore:
    """lifespan 会注入 store；测试/未跑 lifespan 时按 settings 懒构造。"""
    store = getattr(request.app.state, "attachment_store", None)
    if store is None:
        settings = request.app.state.settings
        store = AttachmentStore(settings.uploads_dir, max_bytes=settings.max_upload_bytes)
        request.app.state.attachment_store = store
    return store


@router.post("", response_model=AttachmentOut, status_code=status.HTTP_201_CREATED)
async def upload_attachment(
    request: Request,
    user: CurrentUser,
    file: Annotated[UploadFile, File(description="待解析的附件文件")],
) -> AttachmentOut | JSONResponse:
    settings = request.app.state.settings
    store = _get_store(request)

    try:
        # Next 代理会剥掉 content-length，必须在读取侧限制大小
        data = await file.read(settings.max_upload_bytes + 1)
    finally:
        await file.close()

    if len(data) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes // (1024 * 1024)
        return _error(
            AttachmentError("attachment_too_large", f"附件超过 {limit_mb}MB 上限", status_code=413)
        )

    try:
        await store.purge_expired(user_id=user.id, retention_days=settings.uploads_retention_days)
        record = await store.save(user_id=user.id, filename=file.filename or "附件", data=data)
    except AttachmentError as exc:
        return _error(exc)
    except Exception:  # noqa: BLE001 — 存储失败不该 500 裸奔
        logger.exception("附件保存失败: user=%s", user.id)
        return _error(AttachmentError("attachment_upload_failed", "附件保存失败", status_code=503))

    return AttachmentOut(
        id=str(record.id),
        filename=record.filename,
        content_type=record.content_type,
        size=record.size,
    )
