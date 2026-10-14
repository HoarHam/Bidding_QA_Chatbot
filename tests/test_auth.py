"""认证测试：密码哈希、JWT、注册登录接口。

全部离线：数据落在内存 SQLite 上，不碰任何外部服务。
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from bidding_qa_chatbot.auth.security import (
    BCRYPT_MAX_BYTES,
    TokenError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from tests.conftest import auth_header, register_user


# ---------------------------------------------------------------- 密码哈希
def test_hash_and_verify_roundtrip():
    h = hash_password("正确的密码-abc123")
    assert h != "正确的密码-abc123"  # 绝不能明文存储
    assert verify_password("正确的密码-abc123", h)
    assert not verify_password("错误的密码", h)


def test_hash_is_salted():
    """同一密码两次哈希结果必须不同，否则可以用彩虹表批量破解。"""
    assert hash_password("same-password") != hash_password("same-password")


def test_verify_tolerates_corrupt_hash():
    """哈希串损坏只应导致「登录失败」，不该把 401 变成 500。"""
    assert verify_password("any", "这不是一个合法的 bcrypt 哈希") is False
    assert verify_password("any", "") is False


def test_overlong_password_is_rejected_not_silently_truncated():
    """bcrypt 只取前 72 字节，超出部分被静默丢弃。

    中文一个字 3 字节，不到 24 个汉字就会触顶——若不拒绝，两个前 72 字节
    相同的长密码会被判为同一个密码。
    """
    too_long = "密" * 30  # 90 字节
    assert len(too_long.encode()) > BCRYPT_MAX_BYTES
    with pytest.raises(ValueError, match="密码过长"):
        hash_password(too_long)


# ---------------------------------------------------------------- JWT
def test_token_roundtrip(auth_settings):
    uid = uuid.uuid4()
    token, expires_in = create_access_token(user_id=uid, username="alice", settings=auth_settings)
    assert expires_in > 0

    payload = decode_access_token(token, auth_settings)
    assert payload["sub"] == str(uid)
    assert payload["username"] == "alice"


def test_expired_token_is_rejected(auth_settings):
    token, _ = create_access_token(
        user_id=uuid.uuid4(), username="alice", settings=auth_settings, expires_minutes=-1
    )
    with pytest.raises(TokenError, match="过期"):
        decode_access_token(token, auth_settings)


def test_tampered_token_is_rejected(auth_settings):
    """改一位签名就必须失败——这是 JWT 的全部安全性所在。"""
    token, _ = create_access_token(user_id=uuid.uuid4(), username="a", settings=auth_settings)
    head, body, sig = token.split(".")
    tampered = f"{head}.{body}.{sig[:-2]}xx"
    with pytest.raises(TokenError):
        decode_access_token(tampered, auth_settings)


def test_token_signed_with_other_secret_is_rejected(settings_factory):
    """别人用自己密钥签的 token 不能被接受。"""
    other = settings_factory(jwt_secret="attacker-secret-" + "y" * 40)
    token, _ = create_access_token(user_id=uuid.uuid4(), username="a", settings=other)

    mine = settings_factory(jwt_secret="my-secret-" + "z" * 40)
    with pytest.raises(TokenError):
        decode_access_token(token, mine)


def test_missing_jwt_secret_gives_actionable_error(settings_factory):
    """没配 JWT_SECRET 时要报「怎么修」，而不是一个含糊的失败。"""
    settings = settings_factory(jwt_secret=None)
    with pytest.raises(TokenError, match="JWT_SECRET"):
        create_access_token(user_id=uuid.uuid4(), username="a", settings=settings)


def test_expiry_is_actually_in_the_token(auth_settings):
    """确认 exp 被写进了载荷，而不是只在我们自己的代码里判断。"""
    token, _ = create_access_token(
        user_id=uuid.uuid4(), username="a", settings=auth_settings, expires_minutes=30
    )
    raw = jwt.decode(token, options={"verify_signature": False})
    expected = datetime.now(UTC) + timedelta(minutes=30)
    assert abs(raw["exp"] - int(expected.timestamp())) < 10


# ---------------------------------------------------------------- 注册
async def test_register_returns_token_and_user(api_client):
    token, user = await register_user(api_client, username="alice")
    assert token
    assert user["username"] == "alice"
    assert user["email"] == "alice@example.com"
    # 响应里绝不能出现密码或哈希
    assert "password" not in user and "password_hash" not in user


async def test_register_rejects_duplicate_username(api_client):
    await register_user(api_client, username="bob")
    resp = await api_client.post(
        "/api/auth/register",
        json={"username": "bob", "email": "other@example.com", "password": "pw-12345678"},
    )
    assert resp.status_code == 409
    assert "用户名" in resp.json()["detail"]


async def test_register_rejects_duplicate_email(api_client):
    await register_user(api_client, username="carol", email="dup@example.com")
    resp = await api_client.post(
        "/api/auth/register",
        json={"username": "dave", "email": "dup@example.com", "password": "pw-12345678"},
    )
    assert resp.status_code == 409
    assert "邮箱" in resp.json()["detail"]


async def test_register_rejects_short_password(api_client):
    resp = await api_client.post(
        "/api/auth/register",
        json={"username": "shorty", "email": "s@example.com", "password": "abc"},
    )
    assert resp.status_code == 422


async def test_register_rejects_invalid_email(api_client):
    resp = await api_client.post(
        "/api/auth/register",
        json={"username": "badmail", "email": "不是邮箱", "password": "pw-12345678"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------- 登录
async def test_login_with_username(api_client):
    await register_user(api_client, username="erin", password="pw-12345678")
    resp = await api_client.post(
        "/api/auth/login", json={"username": "erin", "password": "pw-12345678"}
    )
    assert resp.status_code == 200
    assert resp.json()["access_token"]


async def test_login_with_email(api_client):
    await register_user(
        api_client, username="frank", email="frank@example.com", password="pw-12345678"
    )
    resp = await api_client.post(
        "/api/auth/login", json={"username": "frank@example.com", "password": "pw-12345678"}
    )
    assert resp.status_code == 200


async def test_login_with_wrong_password_fails(api_client):
    await register_user(api_client, username="grace", password="pw-12345678")
    resp = await api_client.post(
        "/api/auth/login", json={"username": "grace", "password": "错误的密码"}
    )
    assert resp.status_code == 401


async def test_login_does_not_reveal_whether_user_exists(api_client):
    """用户不存在与密码错误必须返回**完全相同**的响应。

    否则登录接口会变成一个用户名枚举接口：攻击者能据此筛出有效账号，
    再去撞库或钓鱼。
    """
    await register_user(api_client, username="henry", password="pw-12345678")

    wrong_pw = await api_client.post(
        "/api/auth/login", json={"username": "henry", "password": "错误的密码"}
    )
    no_such_user = await api_client.post(
        "/api/auth/login", json={"username": "根本不存在的人", "password": "错误的密码"}
    )

    assert wrong_pw.status_code == no_such_user.status_code == 401
    assert wrong_pw.json() == no_such_user.json()


# ---------------------------------------------------------------- 鉴权
async def test_me_returns_current_user(api_client):
    token, user = await register_user(api_client, username="ivy")
    resp = await api_client.get("/api/auth/me", headers=auth_header(token))
    assert resp.status_code == 200
    assert resp.json()["username"] == "ivy"
    assert resp.json()["id"] == user["id"]


async def test_me_requires_token(api_client):
    resp = await api_client.get("/api/auth/me")
    assert resp.status_code == 401
    assert "登录" in resp.json()["detail"]


async def test_me_rejects_garbage_token(api_client):
    # 用 ASCII：HTTP 头不能直接放非 ASCII 字符（httpx 会拒绝编码）
    resp = await api_client.get("/api/auth/me", headers=auth_header("not-a-real-token"))
    assert resp.status_code == 401


async def test_me_rejects_token_with_valid_shape_but_bad_signature(api_client, auth_settings):
    """三段式结构正确但签名不对，也必须拒绝。"""
    forged = jwt.encode(
        {"sub": str(uuid.uuid4()), "username": "attacker"},
        "wrong-secret-" + "w" * 40,  # 够长，避免 PyJWT 的短密钥告警干扰输出
        algorithm="HS256",
    )
    resp = await api_client.get("/api/auth/me", headers=auth_header(forged))
    assert resp.status_code == 401


async def test_me_rejects_token_of_deleted_user(api_client, db):
    """token 签名有效但用户已被删除——必须按未认证处理。"""
    import sqlalchemy as sa

    from bidding_qa_chatbot.db.models import User

    token, user = await register_user(api_client, username="temp")
    async with db.session() as session:
        await session.execute(sa.delete(User).where(User.username == "temp"))

    resp = await api_client.get("/api/auth/me", headers=auth_header(token))
    assert resp.status_code == 401
    assert user["id"]  # 确认此前确实存在
