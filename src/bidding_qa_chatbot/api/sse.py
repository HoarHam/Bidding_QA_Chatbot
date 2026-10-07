"""SSE 帧编码。

几个必须做对、否则会以「看起来能用」的方式出错的点：

1. **`ensure_ascii=False`**——否则中文会变成 `\\uXXXX` 转义汤，
   虽然协议上合法，但调试时完全不可读。
2. **具名事件**（`event: token`）而非裸 `data:`——前端按事件名分发，
   不用做字符串嗅探。
3. **JSON 字符串天然不含裸换行**，所以前端按 `\\n\\n` 切帧是安全的。
4. **心跳**：深度思考模式下模型可能 30 秒不吐字，
   没有心跳的话 Next 的开发代理或任何中间代理会掐断连接。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

HEARTBEAT_FRAME = ": ping\n\n"

#: SSE 响应头。`X-Accel-Buffering: no` 关掉 nginx 的缓冲，否则流式会变成一次性返回。
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def encode_event(event: str, payload: BaseModel | dict[str, Any]) -> str:
    """编码一个具名 SSE 事件。"""
    if isinstance(payload, BaseModel):
        data = payload.model_dump(by_alias=True, mode="json")
    else:
        data = payload
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {body}\n\n"
