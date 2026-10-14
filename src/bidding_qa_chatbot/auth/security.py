"""密码哈希与 JWT。

## 直接用 bcrypt，不用 passlib

passlib 已多年不更新，且与 bcrypt 4.x 存在兼容告警；这里只是「哈希一个密码、
校验一个密码」两件事，bcrypt 本身足够且少一层依赖。

## 72 字节上限

bcrypt 只取密码的前 72 字节，超出部分**被静默丢弃**——即两个前 72 字节相同的
长密码会被判定为同一个密码。中文一个字符 3 字节，不到 24 个汉字就会触顶。
这里显式拒绝超长密码，而不是让它悄悄降级。
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt

from bidding_qa_chatbot.config import Settings, get_settings
from bidding_qa_chatbot.log import redact

logger = logging.getLogger(__name__)

#: bcrypt 的硬上限（字节）
BCRYPT_MAX_BYTES = 72

#: 密码长度上限（字符）。设成 128 是为了留出余量，真正生效的是上面的字节上限。
MAX_PASSWORD_CHARS = 128


class TokenError(Exception):
    """令牌无效或过期。"""


def _prepare(password: str) -> bytes:
    raw = password.encode("utf-8")
    if len(raw) > BCRYPT_MAX_BYTES:
        raise ValueError(
            f"密码过长：{len(raw)} 字节，bcrypt 上限为 {BCRYPT_MAX_BYTES} 字节"
            f"（约 24 个汉字或 72 个 ASCII 字符）。"
        )
    return raw


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    """校验密码。任何异常都按「不匹配」处理，不向上抛。

    哈希串损坏、算法不识别等情况都只是「这次登录失败」，
    让它们以异常形式冒出去会把 401 变成 500，还会在日志里泄漏哈希细节。
    """
    try:
        return bcrypt.checkpw(_prepare(password), password_hash.encode("ascii"))
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------ JWT
def create_access_token(
    *,
    user_id: uuid.UUID,
    username: str,
    settings: Settings | None = None,
    expires_minutes: int | None = None,
) -> tuple[str, int]:
    """签发 token，返回 (token, 有效秒数)。"""
    settings = settings or get_settings()
    secret = _require_secret(settings)

    minutes = expires_minutes or settings.jwt_expire_minutes
    now = datetime.now(UTC)
    expire = now + timedelta(minutes=minutes)

    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    token = jwt.encode(payload, secret, algorithm=settings.jwt_algorithm)
    return token, minutes * 60


def decode_access_token(token: str, settings: Settings | None = None) -> dict[str, Any]:
    """校验并解析 token。失败一律抛 `TokenError`。"""
    settings = settings or get_settings()
    secret = _require_secret(settings)
    try:
        return jwt.decode(token, secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("登录已过期，请重新登录") from exc
    except jwt.InvalidTokenError as exc:
        # 不把底层原因（签名不符/格式错乱）透给客户端，避免成为探测手段
        logger.info("无效 token: %s", redact(str(exc)))
        raise TokenError("登录凭证无效") from exc


def _require_secret(settings: Settings) -> str:
    if settings.jwt_secret is None:
        raise TokenError(
            "未配置 JWT_SECRET，无法签发或校验登录凭证。"
            '请生成一个随机值写入 .env：python -c "import secrets;print(secrets.token_urlsafe(48))"'
        )
    return settings.jwt_secret.get_secret_value()
