"""会话历史自动摘要：窗口判定、增量摘要、失败降级、提示词注入。"""

from types import SimpleNamespace

from bidding_qa_chatbot.db import repository as repo
from bidding_qa_chatbot.llm.history_summary import ensure_history_summary
from bidding_qa_chatbot.llm.prompts import build_agent_messages, build_messages


class FakeChain:
    def __init__(self, text: str = "前情提要内容", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.requests: list = []

    async def complete(self, req):
        self.requests.append(req)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.text)


async def _seed(db, message_count: int, username: str = "u"):
    async with db.session() as session:
        user = await repo.create_user(
            session, username=username, email=f"{username}@e.com", password_hash="x"
        )
        conversation = await repo.create_conversation(session, user_id=user.id)
        for index in range(message_count):
            await repo.add_message(
                session,
                conversation=conversation,
                role="user" if index % 2 == 0 else "assistant",
                content=f"消息{index + 1}",
            )
        return user, conversation.id


async def test_summary_generated_over_window_and_is_incremental(db, settings_factory):
    user, conversation_id = await _seed(db, 6)
    settings = settings_factory(history_window_messages=2)
    chain = FakeChain("早前讨论了投标保证金。")

    summary = await ensure_history_summary(
        chain=chain,
        db=db,
        settings=settings,
        user_id=user.id,
        conversation_id=conversation_id,
    )

    assert summary == "早前讨论了投标保证金。"
    transcript = chain.requests[0].messages[-1].content
    assert "消息4" in transcript and "消息5" not in transcript  # 只摘要窗口之外
    async with db.session() as session:
        conversation = await repo.get_conversation(
            session, user_id=user.id, conversation_id=conversation_id
        )
        assert conversation is not None
        assert conversation.summary == "早前讨论了投标保证金。"
        assert conversation.summary_upto_seq == 4

    # 已是最新：不再调用 LLM
    again = await ensure_history_summary(
        chain=chain,
        db=db,
        settings=settings,
        user_id=user.id,
        conversation_id=conversation_id,
    )
    assert again == summary
    assert len(chain.requests) == 1


async def test_summary_merges_previous_summary_incrementally(db, settings_factory):
    user, conversation_id = await _seed(db, 6)
    settings = settings_factory(history_window_messages=2)
    chain = FakeChain("第一版摘要")

    await ensure_history_summary(
        chain=chain, db=db, settings=settings, user_id=user.id, conversation_id=conversation_id
    )

    async with db.session() as session:
        conversation = await repo.get_conversation(
            session, user_id=user.id, conversation_id=conversation_id
        )
        assert conversation is not None
        for index in range(2):
            await repo.add_message(
                session,
                conversation=conversation,
                role="user" if index % 2 == 0 else "assistant",
                content=f"新增消息{index + 1}",
            )
    chain.text = "第二版摘要"
    await ensure_history_summary(
        chain=chain, db=db, settings=settings, user_id=user.id, conversation_id=conversation_id
    )

    assert len(chain.requests) == 2
    second_prompt = chain.requests[1].messages[-1].content
    assert "第一版摘要" in second_prompt  # 旧摘要参与合并
    assert "消息5" in second_prompt  # 新纳入摘要的窗口外消息
    async with db.session() as session:
        conversation = await repo.get_conversation(
            session, user_id=user.id, conversation_id=conversation_id
        )
        assert conversation is not None
        assert conversation.summary == "第二版摘要"
        assert conversation.summary_upto_seq == 6


async def test_summary_failure_is_tolerated(db, settings_factory):
    user, conversation_id = await _seed(db, 6)
    settings = settings_factory(history_window_messages=2)
    chain = FakeChain(error=RuntimeError("provider down"))

    summary = await ensure_history_summary(
        chain=chain,
        db=db,
        settings=settings,
        user_id=user.id,
        conversation_id=conversation_id,
    )

    assert summary is None
    async with db.session() as session:
        conversation = await repo.get_conversation(
            session, user_id=user.id, conversation_id=conversation_id
        )
        assert conversation is not None
        assert conversation.summary is None


def test_summary_is_injected_into_prompts():
    agent_messages = build_agent_messages("新问题", history_summary="早前聊过保证金")
    assert "早前聊过保证金" in agent_messages[0].content
    assert "前情提要" in agent_messages[0].content

    legacy_messages = build_messages("新问题", "", history_summary="早前聊过保证金")
    assert "早前聊过保证金" in legacy_messages[0].content
    assert "前情提要" in legacy_messages[0].content
