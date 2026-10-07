"""Provider 层测试（respx 拦截，零网络）。

**这组测试能成立，唯一原因是 `openai` 锁在 2.x。**
3.x 改用 `httpx2` 并不再依赖 `httpx`，respx/pytest-httpx 无法拦截，
mock 会静默放过、真的打到线上付费 API。`test_openai_pin.py` 里有一条守卫断言。

最有价值的断言是「请求体」：DeepSeek 的思考模式**默认开启**，
快速模式若忘记显式关闭，每次问答都会白白付出推理延迟与费用。
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from bidding_qa_chatbot.config import Settings
from bidding_qa_chatbot.llm.base import ChatMessage, LLMRequest, StreamEnd
from bidding_qa_chatbot.llm.openai_compat import OpenAICompatProvider

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


def _settings(**kw) -> Settings:
    base = {
        "qdrant_url": "http://x",
        "qdrant_api_key": "k",
        "deepseek_api_key": "sk-test-key",
        "zhipu_api_key": "id.secret",
        "deepseek_base_url": "https://api.deepseek.com",
    }
    base.update(kw)
    return Settings(_env_file=None, **base)


def _sse(*chunks: dict) -> bytes:
    lines = []
    for c in chunks:
        lines.append(f"data: {json.dumps(c, ensure_ascii=False)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def _chunk(delta: dict, finish: str | None = None) -> dict:
    return {
        "id": "1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "deepseek-flash",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _provider(settings: Settings | None = None) -> OpenAICompatProvider:
    s = settings or _settings()
    return OpenAICompatProvider(
        name="deepseek",
        base_url=s.deepseek_base_url,
        api_key="sk-test-key",
        settings=s,
        supports_thinking_toggle=True,
        thinking_style="deepseek",
    )


async def _drain(provider, req):
    events, end = [], None
    async for item in provider.astream(req):
        if isinstance(item, StreamEnd):
            end = item
        else:
            events.append(item)
    return events, end


@respx.mock
async def test_stream_emits_reasoning_then_content():
    respx.post(DEEPSEEK_URL).mock(
        return_value=httpx.Response(
            200,
            content=_sse(
                _chunk({"reasoning_content": "先分析法律依据"}),
                _chunk({"content": "根据《政府采购法》"}),
                _chunk({}, "stop"),
            ),
            headers={"content-type": "text/event-stream"},
        )
    )
    events, end = await _drain(_provider(), LLMRequest(messages=[ChatMessage("user", "问")]))

    assert [e.kind for e in events] == ["reasoning", "content"]
    assert events[0].text == "先分析法律依据"
    assert events[1].text == "根据《政府采购法》"
    assert end is not None and end.finish_reason == "stop"


@respx.mock
async def test_fast_mode_explicitly_disables_thinking():
    """最重要的断言：DeepSeek 默认开启思考，快速模式必须显式关掉。"""
    route = respx.post(DEEPSEEK_URL).mock(
        return_value=httpx.Response(
            200, content=_sse(_chunk({}, "stop")), headers={"content-type": "text/event-stream"}
        )
    )
    await _drain(_provider(), LLMRequest(messages=[ChatMessage("user", "问")], mode="fast"))

    body = json.loads(route.calls[0].request.content)
    # openai SDK 会把 extra_body 的键**平铺**到请求体顶层，所以断言顶层 thinking。
    assert body["thinking"]["type"] == "disabled"


@respx.mock
async def test_thinking_mode_enables_thinking_and_omits_temperature():
    route = respx.post(DEEPSEEK_URL).mock(
        return_value=httpx.Response(
            200, content=_sse(_chunk({}, "stop")), headers={"content-type": "text/event-stream"}
        )
    )
    await _drain(
        _provider(),
        LLMRequest(messages=[ChatMessage("user", "问")], mode="thinking", temperature=0.7),
    )

    body = json.loads(route.calls[0].request.content)
    assert body["thinking"]["type"] == "enabled"
    # 思考模式下 DeepSeek 会拒绝 temperature / top_p
    assert "temperature" not in body


@respx.mock
async def test_fast_mode_passes_temperature():
    route = respx.post(DEEPSEEK_URL).mock(
        return_value=httpx.Response(
            200, content=_sse(_chunk({}, "stop")), headers={"content-type": "text/event-stream"}
        )
    )
    await _drain(
        _provider(),
        LLMRequest(messages=[ChatMessage("user", "问")], mode="fast", temperature=0.3),
    )
    assert json.loads(route.calls[0].request.content)["temperature"] == 0.3


@respx.mock
async def test_server_error_makes_exactly_one_call():
    """max_retries=0 的证据：SDK 不该自己重试，重试策略只在 chain 里。"""
    route = respx.post(DEEPSEEK_URL).mock(return_value=httpx.Response(500, text="boom"))
    provider = _provider()

    from bidding_qa_chatbot.llm.base import RetryableProviderError

    with pytest.raises(RetryableProviderError):
        await _drain(provider, LLMRequest(messages=[ChatMessage("user", "问")]))
    assert route.call_count == 1


@respx.mock
async def test_unauthorized_is_fatal_and_redacts_message():
    respx.post(DEEPSEEK_URL).mock(return_value=httpx.Response(401, text="bad key"))
    provider = _provider()

    from bidding_qa_chatbot.llm.base import FatalProviderError

    with pytest.raises(FatalProviderError) as excinfo:
        await _drain(provider, LLMRequest(messages=[ChatMessage("user", "问")]))
    assert "401" in excinfo.value.message
    assert "sk-test-key" not in str(excinfo.value)
    assert "请检查 API key" in excinfo.value.message


@respx.mock
async def test_headers_carry_auth():
    route = respx.post(DEEPSEEK_URL).mock(
        return_value=httpx.Response(
            200, content=_sse(_chunk({}, "stop")), headers={"content-type": "text/event-stream"}
        )
    )
    await _drain(_provider(), LLMRequest(messages=[ChatMessage("user", "问")]))
    assert route.calls[0].request.headers["authorization"] == "Bearer sk-test-key"


@respx.mock
async def test_acomplete_returns_reasoning_and_usage():
    respx.post(DEEPSEEK_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "1",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "答案",
                            "reasoning_content": "想法",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )
    )
    result = await _provider().acomplete(LLMRequest(messages=[ChatMessage("user", "问")]))

    assert result.text == "答案"
    assert result.reasoning == "想法"
    assert result.usage is not None and result.usage.total_tokens == 15
    assert result.provider == "deepseek"
    assert result.model == "deepseek-flash"


def test_vllm_model_passed_verbatim():
    """vLLM 的 model 字段是服务端路径，必须原样透传。"""
    s = _settings(vllm_model="/root/autodl-tmp/DeepSeek-R1-0528-Qwen3-8B")
    provider = OpenAICompatProvider("vllm", s.vllm_base_url, "EMPTY", s)
    assert provider.model_for("fast") == "/root/autodl-tmp/DeepSeek-R1-0528-Qwen3-8B"


@respx.mock
async def test_vllm_provider_hits_configured_base_url():
    s = _settings(vllm_model="m", vllm_base_url="http://127.0.0.1:8000/v1")
    provider = OpenAICompatProvider("vllm", s.vllm_base_url, "EMPTY", s)
    route = respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=_sse(_chunk({"content": "本地"})),
            headers={"content-type": "text/event-stream"},
        )
    )
    events, _ = await _drain(provider, LLMRequest(messages=[ChatMessage("user", "问")]))
    assert route.called
    assert [e.text for e in events] == ["本地"]
