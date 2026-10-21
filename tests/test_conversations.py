"""对话历史：仓储层 + 接口层。

重点覆盖**越权隔离**：每个查询都必须带上 user_id 条件，只按主键取会话/消息
会形成越权漏洞——任何人拿到一个 UUID 就能读到别人的对话。
"""

from __future__ import annotations

import pytest

from bidding_qa_chatbot.db import repository as repo
from tests.conftest import auth_header, register_user


# ---------------------------------------------------------------- 仓储层
async def _make_user(db, username="u1"):
    async with db.session() as session:
        return await repo.create_user(
            session, username=username, email=f"{username}@e.com", password_hash="x"
        )


async def test_create_conversation_and_append_messages(db):
    user = await _make_user(db)
    async with db.session() as session:
        conv = await repo.create_conversation(session, user_id=user.id)
        await repo.add_message(
            session,
            conversation=conv,
            role="user",
            content="投标保证金什么时候退？",
            title_from=True,
        )
        await repo.add_message(
            session, conversation=conv, role="assistant", content="签订合同后5日内。"
        )
        conv_id = conv.id

    async with db.session() as session:
        conv = await repo.get_conversation(session, user_id=user.id, conversation_id=conv_id)
        assert conv.message_count == 2  # 冗余计数要跟着维护
        messages = await repo.list_messages(session, user_id=user.id, conversation_id=conv_id)
        assert [m.role for m in messages] == ["user", "assistant"]


async def test_messages_stay_in_insertion_order(db):
    """同一事务里连续插入的消息必须按插入顺序取回。

    回归守卫：曾经按 `(created_at, id)` 排序，而这些消息的 created_at 完全
    相同（Windows 上 datetime.now() 只有约 15ms 分辨率），兜底比较的又是随机
    UUID——于是顺序是随机的，表现为用户回看历史时消息错位。
    实测该问题约每 8 次运行出现 1 次，所以这条用例要插足够多条来放大暴露概率。
    """
    user = await _make_user(db)
    roles = ["user", "assistant"] * 10  # 20 条，同一事务内一次性写入

    async with db.session() as session:
        conv = await repo.create_conversation(session, user_id=user.id)
        for i, role in enumerate(roles):
            await repo.add_message(session, conversation=conv, role=role, content=f"第{i}条")
        conv_id = conv.id

    async with db.session() as session:
        messages = await repo.list_messages(session, user_id=user.id, conversation_id=conv_id)

    assert [m.role for m in messages] == roles
    assert [m.content for m in messages] == [f"第{i}条" for i in range(len(roles))]
    # 序号必须是 1..N 连续无洞
    assert [m.seq for m in messages] == list(range(1, len(roles) + 1))


async def test_message_seq_is_unique_per_conversation(db):
    """序号撞车必须报错，而不是悄悄产生两条顺序相同的消息。"""
    import sqlalchemy.exc

    from bidding_qa_chatbot.db.models import Message

    user = await _make_user(db)
    async with db.session() as session:
        conv = await repo.create_conversation(session, user_id=user.id)
        await repo.add_message(session, conversation=conv, role="user", content="第一条")
        conv_id = conv.id

    with pytest.raises(sqlalchemy.exc.IntegrityError):
        async with db.session() as session:
            session.add(Message(conversation_id=conv_id, seq=1, role="user", content="重复序号"))


async def test_title_derived_from_first_user_message(db):
    user = await _make_user(db)
    async with db.session() as session:
        conv = await repo.create_conversation(session, user_id=user.id)
        await repo.add_message(
            session,
            conversation=conv,
            role="user",
            content="供应商针对单一来源异议被驳回，可以再次异议吗？",
            title_from=True,
        )
        assert conv.title.startswith("供应商针对单一来源异议被驳回")


async def test_long_title_is_truncated(db):
    user = await _make_user(db)
    async with db.session() as session:
        conv = await repo.create_conversation(session, user_id=user.id)
        await repo.add_message(
            session, conversation=conv, role="user", content="问" * 200, title_from=True
        )
        assert len(conv.title) <= repo.TITLE_MAX_CHARS + 1  # +1 是省略号
        assert conv.title.endswith("…")


async def test_conversations_listed_newest_first(db):
    user = await _make_user(db)
    async with db.session() as session:
        first = await repo.create_conversation(session, user_id=user.id, title="旧")
        second = await repo.create_conversation(session, user_id=user.id, title="新")
        await repo.add_message(session, conversation=first, role="user", content="把旧的顶上去")

    async with db.session() as session:
        items = await repo.list_conversations(session, user_id=user.id)
        assert [c.id for c in items][0] == first.id  # 刚有消息的排最前
        assert second.id in [c.id for c in items]


async def test_delete_conversation_cascades_messages(db):
    user = await _make_user(db)
    async with db.session() as session:
        conv = await repo.create_conversation(session, user_id=user.id)
        await repo.add_message(session, conversation=conv, role="user", content="内容")
        conv_id = conv.id

    async with db.session() as session:
        assert await repo.delete_conversation(session, user_id=user.id, conversation_id=conv_id)

    async with db.session() as session:
        # 级联删除生效，消息不该残留
        assert await repo.list_messages(session, user_id=user.id, conversation_id=conv_id) == []


async def test_search_finds_matching_messages(db):
    user = await _make_user(db)
    async with db.session() as session:
        conv = await repo.create_conversation(session, user_id=user.id, title="保证金问题")
        await repo.add_message(
            session, conversation=conv, role="user", content="投标保证金多久退还"
        )
        await repo.add_message(session, conversation=conv, role="assistant", content="无关内容")

    async with db.session() as session:
        hits = await repo.search_messages(session, user_id=user.id, query="保证金")
        assert len(hits) == 1
        msg, found_conv = hits[0]
        assert "保证金" in msg.content
        assert found_conv.id == conv.id  # 带上所属会话，前端才能跳过去


async def test_search_is_scoped_to_own_conversations(db):
    """搜索绝不能返回别人的对话。"""
    mine = await _make_user(db, "mine")
    theirs = await _make_user(db, "theirs")
    async with db.session() as session:
        conv = await repo.create_conversation(session, user_id=theirs.id)
        await repo.add_message(session, conversation=conv, role="user", content="别人的机密对话")

    async with db.session() as session:
        assert await repo.search_messages(session, user_id=mine.id, query="机密") == []


@pytest.mark.parametrize("q", ["100%", "a_b", "back\\slash"])
async def test_search_escapes_like_wildcards(db, q):
    """搜索 `100%` 时 % 必须当字面量，不能当通配符。"""
    user = await _make_user(db)
    async with db.session() as session:
        conv = await repo.create_conversation(session, user_id=user.id)
        await repo.add_message(session, conversation=conv, role="user", content=f"比例是 {q}")
        await repo.add_message(session, conversation=conv, role="user", content="完全无关")

    async with db.session() as session:
        hits = await repo.search_messages(session, user_id=user.id, query=q)
        assert len(hits) == 1, f"{q} 被当成通配符了"


async def test_search_empty_query_returns_nothing(db):
    user = await _make_user(db)
    async with db.session() as session:
        conv = await repo.create_conversation(session, user_id=user.id)
        await repo.add_message(session, conversation=conv, role="user", content="内容")
        assert await repo.search_messages(session, user_id=user.id, query="   ") == []


# ---------------------------------------------------------------- 越权隔离
async def test_cannot_read_others_conversation(db):
    """拿到别人的会话 id 也读不到——授权写在查询里，不靠调用方记得检查。"""
    alice = await _make_user(db, "alice")
    bob = await _make_user(db, "bob")
    async with db.session() as session:
        conv = await repo.create_conversation(session, user_id=bob.id, title="Bob 的对话")
        conv_id = conv.id

    async with db.session() as session:
        assert (
            await repo.get_conversation(session, user_id=alice.id, conversation_id=conv_id) is None
        )
        assert await repo.list_messages(session, user_id=alice.id, conversation_id=conv_id) == []
        assert not await repo.delete_conversation(
            session, user_id=alice.id, conversation_id=conv_id
        )


async def test_list_only_returns_own_conversations(db):
    alice = await _make_user(db, "alice2")
    bob = await _make_user(db, "bob2")
    async with db.session() as session:
        await repo.create_conversation(session, user_id=bob.id, title="Bob 的")

    async with db.session() as session:
        assert await repo.list_conversations(session, user_id=alice.id) == []
        assert await repo.count_conversations(session, user_id=alice.id) == 0


# ---------------------------------------------------------------- 接口层
async def test_conversations_require_auth(api_client):
    for method, path in [("get", "/api/conversations"), ("get", "/api/conversations/search?q=x")]:
        resp = await getattr(api_client, method)(path)
        assert resp.status_code == 401, path


async def test_conversation_endpoints_reject_foreign_id(api_client):
    """端到端验证越权：Bob 建的会话，Alice 拿 id 去读必须 404。"""
    bob_token, _ = await register_user(api_client, username="bob_e2e")
    alice_token, _ = await register_user(api_client, username="alice_e2e")

    # 直接用仓储造一条 Bob 的会话（绕过聊天接口，避免依赖模型）
    import sqlalchemy as sa

    from bidding_qa_chatbot.db.models import Conversation, User

    async with api_client._transport.app.state.db.session() as session:  # type: ignore[attr-defined]
        bob = (
            await session.execute(sa.select(User).where(User.username == "bob_e2e"))
        ).scalar_one()
        conv = Conversation(user_id=bob.id, title="Bob 私有")
        session.add(conv)
        await session.flush()
        conv_id = str(conv.id)

    resp = await api_client.get(f"/api/conversations/{conv_id}", headers=auth_header(alice_token))
    assert resp.status_code == 404

    resp = await api_client.delete(
        f"/api/conversations/{conv_id}", headers=auth_header(alice_token)
    )
    assert resp.status_code == 404

    # 本人来读则正常
    resp = await api_client.get(f"/api/conversations/{conv_id}", headers=auth_header(bob_token))
    assert resp.status_code == 200
    assert resp.json()["title"] == "Bob 私有"


async def test_search_route_not_shadowed_by_id_route(api_client):
    """/search 必须声明在 /{conversation_id} 之前。

    反过来的话 "search" 会被当成会话 id 去解析，搜索接口会永远返回「id 格式非法」。
    """
    token, _ = await register_user(api_client, username="searcher")
    resp = await api_client.get("/api/conversations/search?q=保证金", headers=auth_header(token))
    assert resp.status_code == 200
    assert resp.json()["query"] == "保证金"


async def test_search_snippet_centers_on_keyword(api_client):
    """摘要要从关键词附近截取，从正文开头截会经常看不到命中的地方。"""
    token, _ = await register_user(api_client, username="snip")
    app = api_client._transport.app  # type: ignore[attr-defined]

    import sqlalchemy as sa

    from bidding_qa_chatbot.db.models import Conversation, User

    async with app.state.db.session() as session:
        user = (await session.execute(sa.select(User).where(User.username == "snip"))).scalar_one()
        conv = Conversation(user_id=user.id, title="长对话")
        session.add(conv)
        await session.flush()
        await repo.add_message(
            session,
            conversation=conv,
            role="user",
            content="前面" * 200 + "关键词中标" + "后面" * 200,
        )

    resp = await api_client.get("/api/conversations/search?q=中标", headers=auth_header(token))
    snippet = resp.json()["hits"][0]["snippet"]
    assert "中标" in snippet
    assert snippet.startswith("…")  # 说明是从中间截的，不是从头


async def test_invalid_conversation_id_returns_400(api_client):
    token, _ = await register_user(api_client, username="badid")
    resp = await api_client.get("/api/conversations/not-a-uuid", headers=auth_header(token))
    assert resp.status_code == 400


async def test_list_conversations_empty_for_new_user(api_client):
    token, _ = await register_user(api_client, username="fresh")
    resp = await api_client.get("/api/conversations", headers=auth_header(token))
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "total": 0}


async def test_recent_user_attachments_is_scoped_and_newest_first(db):
    """追问注入的查询：只属于本人、最新在前、忽略无附件的消息。"""
    user1 = await _make_user(db, "a")
    user2 = await _make_user(db, "b")
    payload = lambda text: [  # noqa: E731 — 测试内简单构造
        {
            "id": text,
            "filename": f"{text}.png",
            "content_type": "image/png",
            "size": 1,
            "kind": "image",
            "ocr_text": text,
        }
    ]

    async with db.session() as session:
        conv = await repo.create_conversation(session, user_id=user1.id)
        await repo.add_message(
            session,
            conversation=conv,
            role="user",
            content="q1",
            attachments=payload("第一份"),
        )
        await repo.add_message(session, conversation=conv, role="user", content="q2")
        await repo.add_message(
            session,
            conversation=conv,
            role="user",
            content="q3",
            attachments=payload("第二份"),
        )
        conv_id = conv.id

        other = await repo.create_conversation(session, user_id=user2.id)
        await repo.add_message(
            session,
            conversation=other,
            role="user",
            content="q4",
            attachments=payload("别人的附件"),
        )

    async with db.session() as session:
        rows = await repo.recent_user_attachments(
            session, user_id=user1.id, conversation_id=conv_id, limit=2
        )
    assert [row[0]["ocr_text"] for row in rows] == ["第二份", "第一份"]

    async with db.session() as session:
        assert (
            await repo.recent_user_attachments(
                session, user_id=user2.id, conversation_id=conv_id, limit=2
            )
            == []
        )
