"""分块详情与图片服务。

图片路由直接由用户输入拼路径，必须做目录穿越防护：
`chunk_id` 用正则白名单校验，解析后的绝对路径还要再确认仍在允许目录之下。
只靠前者不够（URL 编码、`..` 变体很多），只靠后者也不够（会放过符号链接）。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.schemas import SourceDetail, SourceImage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sources", tags=["sources"])

#: chunk_id 形态：dataset-0000，dataset 只允许字母数字
_CHUNK_ID_RE = re.compile(r"^[A-Za-z0-9]{1,32}-\d{1,6}$")

_PAYLOAD_FIELDS = (
    "chunk_id",
    "dataset",
    "source_label",
    "question",
    "answer",
    "excel_row",
    "images",
)


async def _fetch_payload(request: Request, chunk_id: str) -> dict:
    from qdrant_client import models as qm

    from bidding_qa_chatbot.ingestion.ids import make_point_id

    settings: Settings = request.app.state.settings
    records = await request.app.state.qdrant.retrieve(
        collection_name=settings.qdrant_collection,
        ids=[make_point_id(chunk_id)],
        with_payload=True,
        with_vectors=False,
    )
    if not records:
        raise HTTPException(status_code=404, detail=f"未找到分块 {chunk_id}")
    _ = qm  # 保持 import 语义清晰
    return dict(records[0].payload or {})


@router.get("/{chunk_id}", response_model=SourceDetail)
async def get_source(chunk_id: str, request: Request) -> SourceDetail:
    if not _CHUNK_ID_RE.match(chunk_id):
        raise HTTPException(status_code=400, detail="chunk_id 格式非法")

    payload = await _fetch_payload(request, chunk_id)
    return SourceDetail(
        chunk_id=str(payload.get("chunk_id", chunk_id)),
        dataset=str(payload.get("dataset", "")),
        source_label=str(payload.get("source_label", "")),
        question=str(payload.get("question", "")),
        answer=str(payload.get("answer", "")),
        excel_row=int(payload.get("excel_row", 0)),
        images=[
            SourceImage(
                index=int(img.get("index", 0)),
                url=str(img.get("url", "")),
                description=img.get("description"),
            )
            for img in (payload.get("images") or [])
        ],
    )


@router.get("/{chunk_id}/images/{index}")
async def get_source_image(chunk_id: str, index: int, request: Request) -> FileResponse:
    if not _CHUNK_ID_RE.match(chunk_id):
        raise HTTPException(status_code=400, detail="chunk_id 格式非法")
    if index < 1 or index > 50:
        raise HTTPException(status_code=400, detail="图片序号越界")

    payload = await _fetch_payload(request, chunk_id)
    settings: Settings = request.app.state.settings

    images = payload.get("images") or []
    match = next((img for img in images if int(img.get("index", -1)) == index), None)
    if match is None:
        raise HTTPException(status_code=404, detail="该分块没有这张图片")

    root = settings.images_dir.resolve()
    stored = Path(str(match.get("storage_path", "")))
    target = (root / str(payload.get("dataset", "")) / stored).resolve()

    # 目录穿越防护：解析后必须仍在 images_dir 之内。
    if not target.is_relative_to(root):
        logger.warning("拦截越权图片访问: %s", target)
        raise HTTPException(status_code=400, detail="路径非法")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="图片文件不存在，请重新运行 ingest")

    return FileResponse(target, media_type=_media_type(target.name))


def _media_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
