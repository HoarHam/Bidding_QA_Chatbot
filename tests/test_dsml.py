"""DSML 过滤器：变体兼容、参数类型、普通文本不受影响。"""

import json

from bidding_qa_chatbot.llm.dsml import DSMLStreamFilter, strip_dsml


def _run(chunks: list[str]):
    stream_filter = DSMLStreamFilter()
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


def test_ascii_pipe_variant_and_json_parameter():
    block = (
        '<|DSML|tool_calls><|DSML|invoke name="query_bidding_graph">'
        '<|DSML|parameter name="limit" string="false">5</|DSML|parameter>'
        "</|DSML|invoke></|DSML|tool_calls>"
    )
    visible, calls = _run([block])

    assert visible == ""
    assert len(calls) == 1
    assert calls[0].name == "query_bidding_graph"
    assert json.loads(calls[0].arguments) == {"limit": 5}


def test_single_bar_with_space_after_dsml():
    block = (
        '<｜DSML｜ tool_calls><｜DSML｜ invoke name="search_bidding_knowledge">'
        '<｜DSML｜ parameter name="query" string="true">保证金</｜DSML｜ parameter>'
        "</｜DSML｜ invoke></｜DSML｜ tool_calls>"
    )
    visible, calls = _run([block[:15], block[15:60], block[60:]])

    assert visible == ""
    assert calls and calls[0].name == "search_bidding_knowledge"
    assert json.loads(calls[0].arguments)["query"] == "保证金"


def test_normal_text_is_not_altered():
    text = "1 < 2，结果见 a|b；这是 <b>普通标签</b>。"
    visible, calls = _run([text[:5], text[5:]])

    assert visible == text
    assert calls == []


def test_strip_dsml_keeps_surrounding_text():
    block = '<｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke name="x"></｜｜DSML｜｜ invoke></｜｜DSML｜｜ calls>'
    assert strip_dsml(f"前缀{block}后缀") == "前缀后缀"
