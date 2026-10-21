"""本地 Ollama GLM-OCR 客户端。

GLM-OCR 走 Ollama **原生** `/api/generate`：`images` 接收原始 base64（无 `data:`
前缀），识别结果在响应 JSON 的 `response` 字段。官方部署指南推荐该接口，
且模型的 RENDERER/PARSER 是为它定制的。

**不要接入 `ProviderChain`**：OCR 不是文本 provider 的回退候选，冷启动（首次
加载模型）约 25s，远超过首 token 超时；它也不该污染文本 provider 的熔断器。
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time

import httpx

from bidding_qa_chatbot.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: GLM-OCR 官方 SDK 的默认任务提示词
OCR_PROMPT = "Text Recognition:"


class OCRUnavailable(RuntimeError):
    """OCR 服务不可用或调用失败。消息可直接转述给用户。"""


class OllamaOCR:
    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        #: 由应用注入共享 httpx 客户端（继承 trust_env=False、连接池统一）；
        #: 为 None 时每次调用临时建一个客户端（测试/脚本场景）。
        self._client = client
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def base_url(self) -> str:
        """防御性去掉结尾 `/v1`：该配置可能被误填成 OpenAI 兼容地址。"""
        base = self.settings.ollama_ocr_base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return base

    def _get_semaphore(self) -> asyncio.Semaphore:
        # 懒创建：避免在 import 时绑定事件循环（测试会换循环）
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(max(1, self.settings.ollama_ocr_max_concurrency))
        return self._semaphore

    async def describe_image(self, data: bytes, media_type: str = "image/png") -> str:
        """识别一张图片，返回文本（可能为空，由调用方判定整体是否可用）。"""
        if not data:
            raise OCRUnavailable("附件内容为空，无法识别")

        payload = {
            "model": self.settings.ollama_ocr_model,
            "prompt": OCR_PROMPT,
            "images": [base64.b64encode(data).decode("ascii")],
            "stream": False,
            # 冷加载本机实测可达 ~2 分钟；让模型跨请求常驻，避免反复冷启动。
            "keep_alive": self.settings.ollama_ocr_keep_alive,
            "options": {
                "temperature": 0,
                "num_predict": self.settings.ollama_ocr_num_predict,
                "num_ctx": self.settings.ollama_ocr_num_ctx,
            },
        }
        # httpx.Timeout 必须给默认值或四个参数全给（仓库有源码扫描守卫）
        timeout = httpx.Timeout(self.settings.ollama_ocr_timeout, connect=5.0)
        url = f"{self.base_url}/api/generate"
        started = time.perf_counter()

        async with self._get_semaphore():
            try:
                if self._client is not None:
                    response = await self._client.post(url, json=payload, timeout=timeout)
                else:
                    async with httpx.AsyncClient(trust_env=False) as temp:
                        response = await temp.post(url, json=payload, timeout=timeout)
                response.raise_for_status()
            except (TimeoutError, httpx.TimeoutException) as exc:
                raise OCRUnavailable(
                    f"OCR 识别超时（超过 {self.settings.ollama_ocr_timeout:.0f} 秒）"
                ) from exc
            except httpx.HTTPStatusError as exc:
                raise OCRUnavailable(
                    f"OCR 服务返回错误（HTTP {exc.response.status_code}）"
                ) from exc
            except Exception as exc:  # noqa: BLE001 — 统一转成可转述错误
                raise OCRUnavailable(
                    "无法连接本地 Ollama 服务（请确认已启动并部署了 glm-ocr 模型）"
                ) from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise OCRUnavailable("OCR 服务返回格式异常") from exc
        text = body.get("response") if isinstance(body, dict) else None
        if not isinstance(text, str):
            raise OCRUnavailable("OCR 服务返回格式异常")

        # 只记录长度与耗时，绝不记录 OCR 文本或 base64
        logger.info(
            "附件 OCR 完成: %d 字符，%.1fs",
            len(text.strip()),
            time.perf_counter() - started,
        )
        return text.strip()

    async def aclose(self) -> None:
        """共享 http 客户端由 app 管理；这里无需关闭。"""
        return None
