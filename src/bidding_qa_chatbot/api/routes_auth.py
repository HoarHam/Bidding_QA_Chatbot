"""账号管理：注册、登录、获取当前用户。"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status

from bidding_qa_chatbot.auth.deps import CurrentUser, DbSession
from bidding_qa_chatbot.auth.security import (
    BCRYPT_MAX_BYTES,
    TokenError,
    create_access_token,
    hash_password,
    verify_password,
)
from bidding_qa_chatbot.db import repository as repo
from bidding_qa_chatbot.db.models import User
from bidding_qa_chatbot.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserOut,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])


def to_user_out(user: User) -> UserOut:
    return UserOut(
        id=str(user.id),
        username=user.username,
        email=user.email,
        display_name=user.display_name,
        created_at=user.created_at,
    )


def _issue(user: User, request: Request) -> TokenResponse:
    settings = request.app.state.settings
    try:
        token, expires_in = create_access_token(
            user_id=user.id, username=user.username, settings=settings
        )
    except TokenError as exc:
        # 没配 JWT_SECRET 属于服务端配置问题，不该报 401 让用户以为是自己输了错密码
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=exc.args[0] if exc.args else "认证服务未就绪",
        ) from exc
    return TokenResponse(access_token=token, expires_in=expires_in, user=to_user_out(user))


def _ensure_token_issuable(request: Request) -> None:
    """签发凭证所需的配置是否就绪。

    必须在**写库之前**检查。否则会留下半成功状态：用户行已经建好（事务在
    请求结束时提交），而响应是错误——用户以为注册失败去重试，却被告知
    「用户名已被占用」，此时他既登不进去、也注册不了。
    """
    if getattr(request.app.state.settings, "jwt_secret", None) is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "服务端未配置 JWT_SECRET，无法完成登录。请在 .env 中加入："
                'JWT_SECRET=$(python -c "import secrets;print(secrets.token_urlsafe(48))")'
            ),
        )


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, request: Request, session: DbSession) -> TokenResponse:
    settings = request.app.state.settings
    _ensure_token_issuable(request)

    if len(body.password.encode("utf-8")) > BCRYPT_MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"密码过长（上限 {BCRYPT_MAX_BYTES} 字节，约 24 个汉字）",
        )
    if len(body.password) < settings.password_min_length:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"密码至少 {settings.password_min_length} 位",
        )

    conflict = await repo.username_or_email_taken(session, body.username, body.email)
    if conflict == "username":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="该用户名已被占用")
    if conflict == "email":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="该邮箱已被注册")

    user = await repo.create_user(
        session,
        username=body.username,
        email=body.email.lower(),
        password_hash=hash_password(body.password),
        display_name=body.display_name,
    )
    logger.info("新用户注册: %s", user.username)
    return _issue(user, request)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, request: Request, session: DbSession) -> TokenResponse:
    identifier = body.username.strip()
    # 允许用户名或邮箱登录
    user = await repo.get_user_by_username(session, identifier)
    if user is None:
        user = await repo.get_user_by_email(session, identifier.lower())

    # 用户不存在与密码错误返回**完全相同**的响应，避免成为用户名枚举接口
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="账号已被停用")

    return _issue(user, request)


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser) -> UserOut:
    return to_user_out(user)
