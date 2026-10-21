"""消息级操作：用户对助手回答的反馈（👍/👎）。

反馈用于后续分析检索与生成策略；只允许操作**本人会话里的助手消息**。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from bidding_qa_chatbot.auth.deps import CurrentUser, DbSession
from bidding_qa_chatbot.db import repository as repo
from bidding_qa_chatbot.schemas import FeedbackOut, FeedbackRequest

router = APIRouter(prefix="/api/messages", tags=["messages"])


@router.post("/{message_id}/feedback", response_model=FeedbackOut)
async def set_feedback(
    message_id: str,
    body: FeedbackRequest,
    user: CurrentUser,
    db: DbSession,
) -> FeedbackOut:
    try:
        message_uuid = uuid.UUID(message_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="消息 id 格式非法") from exc

    # 非本人 / 不存在 / 非助手消息统一 404，避免探测他人消息
    ok = await repo.set_message_feedback(
        db, user_id=user.id, message_id=message_uuid, rating=body.rating
    )
    if not ok:
        raise HTTPException(status_code=404, detail="消息不存在")

    return FeedbackOut(id=message_id, feedback=body.rating)
