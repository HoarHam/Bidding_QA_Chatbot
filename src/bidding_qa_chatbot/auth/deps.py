"""FastAPI 依赖：数据库会话与当前登录用户。"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from bidding_qa_chatbot.auth.security import TokenError, decode_access_token
from bidding_qa_chatbot.db.models import User
from bidding_qa_chatbot.db.repository import get_user_by_id

logger = logging.getLogger(__name__)

#: auto_error=False：缺少 Authorization 头时不由 FastAPI 直接抛 403，
#: 而是走我们自己的 401 —— 语义更对（未认证 ≠ 无权限），前端也更好处理
#: （收到 401 就跳登录页）。
_bearer = HTTPBearer(auto_error=False)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="请先登录",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    """每个请求一个会话，退出时提交（异常则回滚）。

    Database 挂在 `app.state` 上而不是用模块级单例，是为了让测试能整个替换掉——
    测试里用的是内存 SQLite，不碰真实 Postgres。
    """
    db = request.app.state.db
    async with db.session() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(get_db_session)]


async def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    session: DbSession,
) -> User:
    if credentials is None or not credentials.credentials:
        raise _UNAUTHORIZED

    # 必须用 app 上的 settings，不能用 decode_access_token 的默认值（全局 settings）：
    # 两者在生产环境恰好是同一个对象，一旦被替换（测试注入、多环境）就会
    # 用错误的密钥校验，表现为「刚拿到的 token 立刻 401」。
    settings = getattr(request.app.state, "settings", None)
    try:
        payload = decode_access_token(credentials.credentials, settings)
    except TokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=exc.args[0] if exc.args else "登录凭证无效",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    try:
        user_id = uuid.UUID(str(payload.get("sub", "")))
    except ValueError as exc:
        raise _UNAUTHORIZED from exc

    user = await get_user_by_id(session, user_id)
    if user is None:
        # token 合法但用户已被删除
        raise _UNAUTHORIZED
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="账号已被停用")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
