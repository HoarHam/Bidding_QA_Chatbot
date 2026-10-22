"""GLM 工具调用文本标记：过滤、还原、跨 chunk、provider 集成。"""

import json
from types import SimpleNamespace

import pytest

from bidding_qa_chatbot.llm.tool_markup import (
    ZhipuToolCallFilter,
    strip_zhipu_tool_calls,
)

MARKUP = (
    "<tool_call>search_bidding_knowledge"
    "<arg_key>query</arg_key><arg_value>2023年 采购人 交易</arg_value>"
    "</tool_call>"
)


def _run(chunks: list[str]):
    stream_filter = ZhipuToolCallFilter()
    visible: list[str] = []
    calls = []
    for chunk in chunks:
        text, parsed = stream_filter.feed(chunk)
        visible.append(text)
        calls.extend(parsed)
    text, parsed = stream_filter.finish()
    visible.append(text)
    calls.extend(parsed)
    return "".join(visible), calls


def test_extracts_call_across_chunks():
    visible, calls = _run([MARKUP[:10], MARKUP[10:45], MARKUP[45:]])

    assert "<tool_call>" not in visible and "<arg_key>" not in visible
    assert calls and calls[0].name == "search_bidding_knowledge"
    assert json.loads(calls[0].arguments)["query"] == "2023年 采购人 交易"


def test_keeps_surrounding_text_and_multiple_calls():
    text = f"先查一下{MARKUP}再看看{MARKUP}"
    visible, calls = _run([text[:20], text[20:]])

    assert "先查一下" in visible and "再看看" in visible
    assert "<tool_call>" not in visible
    assert len(calls) == 2


def test_whitespace_and_json_values():
    markup = (
        "<tool_call>\n query_bidding_graph \n"
        "<arg_key> limit </arg_key><arg_value>5</arg_value>\n"
        "<arg_key>cypher</arg_key><arg_value>MATCH (n) RETURN n</arg_value>"
        "</tool_call>"
    )
    visible, calls = _run([markup])

    assert visible == ""
    args = json.loads(calls[0].arguments)
    assert args["limit"] == 5
    assert args["cypher"] == "MATCH (n) RETURN n"


def test_unclosed_markup_never_leaks():
    visible, calls = _run(["正常回答<tool_call>search_bidding_knowledge<arg_key>query"])

    assert visible == "正常回答"
    assert calls == []


def test_normal_text_is_untouched():
    text = "普通文本 <b>粗体</b> 与 a|b"
    visible, calls = _run([text[:6], text[6:]])
    assert visible == text
    assert calls == []


def test_strip_helper():
    assert strip_zhipu_tool_calls(f"前缀{MARKUP}后缀") == "前缀后缀"


# ---------------------------------------------------------------- provider 集成
class FakeZhipuClient:
    def __init__(self, chunks) -> None:
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: iter(chunks))
        )


def _chunk(delta: dict, finish: str | None = None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(**delta),
                finish_reason=finish,
            )
        ],
        usage=None,
    )


@pytest.mark.asyncio
async def test_provider_hides_markup_and_recovers_tool_call():
    from bidding_qa_chatbot.config import Settings
    from bidding_qa_chatbot.llm.base import LLMRequest, StreamEnd
    from bidding_qa_chatbot.llm.zhipu import ZhipuProvider

    settings = Settings(
        _env_file=None,
        qdrant_url="http://x",
        qdrant_api_key="k",
        zhipu_api_key="id.secret",
    )
    provider = ZhipuProvider(settings)
    provider._client = FakeZhipuClient(
        [
            _chunk({"content": MARKUP[:20]}),
            _chunk({"content": MARKUP[20:]}),
            _chunk({}, "stop"),
        ]
    )

    events, end = [], None
    async for item in provider.astream(LLMRequest(messages=[])):
        if isinstance(item, StreamEnd):
            end = item
        else:
            events.append(item)

    text = "".join(e.text for e in events if e.kind == "content")
    assert "<tool_call>" not in text and "<arg_key>" not in text
    assert end is not None and end.tool_calls
    assert end.tool_calls[0].name == "search_bidding_knowledge"
    assert json.loads(end.tool_calls[0].arguments)["query"] == "2023年 采购人 交易"
