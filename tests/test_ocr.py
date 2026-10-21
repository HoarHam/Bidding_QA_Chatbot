"""Ollama GLM-OCR 客户端：原生 /api/generate 请求体、错误映射、并发限制。"""

import asyncio
import base64
import json

import httpx
import pytest
import respx

from bidding_qa_chatbot.llm.ocr import OCRUnavailable, OllamaOCR

NATIVE_URL = "http://localhost:11434/api/generate"


def _ocr(settings_factory, http=None, **settings_kw) -> OllamaOCR:
    return OllamaOCR(settings_factory(**settings_kw), http)


@respx.mock
async def test_sends_native_payload(settings_factory):
    route = respx.post(NATIVE_URL).mock(
        return_value=httpx.Response(200, json={"response": "  识别文本  "})
    )
    data = b"\x89PNG\r\n\x1a\nnot-a-real-png"

    async with httpx.AsyncClient() as http:
        text = await _ocr(settings_factory, http).describe_image(data, "image/png")

    assert text == "识别文本"
    payload = json.loads(route.calls[0].request.content)
    assert payload["model"] == "glm-ocr:latest"
    assert payload["prompt"] == "Text Recognition:"
    assert payload["images"] == [base64.b64encode(data).decode("ascii")]
    assert payload["stream"] is False
    assert payload["keep_alive"] == "30m"
    assert payload["options"] == {"temperature": 0, "num_predict": 8192, "num_ctx": 16384}


@respx.mock
async def test_trailing_v1_is_stripped(settings_factory):
    route = respx.post(NATIVE_URL).mock(return_value=httpx.Response(200, json={"response": "ok"}))
    ocr = _ocr(settings_factory, ollama_ocr_base_url="http://localhost:11434/v1")

    assert await ocr.describe_image(b"data") == "ok"
    assert route.called


@respx.mock
async def test_http_error_maps_to_ocr_unavailable(settings_factory):
    respx.post(NATIVE_URL).mock(return_value=httpx.Response(500))

    async with httpx.AsyncClient() as http:
        with pytest.raises(OCRUnavailable) as excinfo:
            await _ocr(settings_factory, http).describe_image(b"data")

    assert "HTTP 500" in str(excinfo.value)


@respx.mock
async def test_timeout_maps_to_ocr_unavailable(settings_factory):
    respx.post(NATIVE_URL).mock(side_effect=httpx.ReadTimeout("slow"))

    async with httpx.AsyncClient() as http:
        with pytest.raises(OCRUnavailable) as excinfo:
            await _ocr(settings_factory, http).describe_image(b"data")

    assert "超时" in str(excinfo.value)


@respx.mock
async def test_invalid_json_maps_to_ocr_unavailable(settings_factory):
    respx.post(NATIVE_URL).mock(return_value=httpx.Response(200, content=b"not json"))

    async with httpx.AsyncClient() as http:
        with pytest.raises(OCRUnavailable) as excinfo:
            await _ocr(settings_factory, http).describe_image(b"data")

    assert "格式异常" in str(excinfo.value)


@respx.mock
async def test_missing_response_field_maps_to_ocr_unavailable(settings_factory):
    respx.post(NATIVE_URL).mock(return_value=httpx.Response(200, json={"done": True}))

    async with httpx.AsyncClient() as http:
        with pytest.raises(OCRUnavailable):
            await _ocr(settings_factory, http).describe_image(b"data")


@respx.mock
async def test_empty_response_text_is_allowed(settings_factory):
    respx.post(NATIVE_URL).mock(return_value=httpx.Response(200, json={"response": "  "}))

    async with httpx.AsyncClient() as http:
        assert await _ocr(settings_factory, http).describe_image(b"data") == ""


async def test_empty_data_is_rejected_without_http(settings_factory):
    ocr = _ocr(settings_factory)
    with pytest.raises(OCRUnavailable):
        await ocr.describe_image(b"")


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"response": "ok"}


class _TrackingClient:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def post(self, url, json=None, timeout=None):  # noqa: A002 — 与 httpx 签名一致
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.05)
        self.active -= 1
        return _FakeResponse()


async def test_semaphore_limits_concurrency(settings_factory):
    client = _TrackingClient()
    ocr = _ocr(settings_factory, client, ollama_ocr_max_concurrency=1)

    await asyncio.gather(*(ocr.describe_image(b"data") for _ in range(3)))

    assert client.max_active == 1
