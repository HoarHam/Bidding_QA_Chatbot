"""图片内容解析：`glm-4.6v-flashx`（需求 6）。

用途单一且关键：语料第 52 行的正文只有一句「…招采情况如下：」，
**真正的答案全在 3 张图里**。把图片转录成 Markdown 表格后追加进分块正文，
那一行才变得可检索、可回答。

输出固定要求为 Markdown 表格：语料的图片本身就是表格式的招采统计，
表格结构能保留行列对应关系，前端也能用 remark-gfm 直接渲染。
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path

from bidding_qa_chatbot.config import Settings, get_settings

logger = logging.getLogger(__name__)

VISION_PROMPT = (
    "请把这张图片的内容完整转录为 Markdown 表格。要求：\n"
    "1. 保留全部表头与每一行数据，不要遗漏、不要合并行；\n"
    "2. 数字、单位、百分号、括号等原样保留；\n"
    "3. 只输出 Markdown 表格本身，不要添加任何解释、前言或结语；\n"
    "4. 如果图中包含多个表格，依次输出，表格之间空一行。"
)


class VisionDescriber:
    """把图片字节转成 Markdown 文本。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None
        self._http_client = None

    def _ensure_client(self):
        if self._client is None:
            import httpx
            from zai import ZhipuAiClient

            # 图片转录较慢，读超时放宽到 120 秒；必须给默认值，否则 httpx 直接报错。
            self._http_client = httpx.Client(timeout=httpx.Timeout(120.0, connect=5.0))
            self._client = ZhipuAiClient(
                api_key=(
                    self.settings.zhipu_api_key.get_secret_value()
                    if self.settings.zhipu_api_key
                    else ""
                ),
                base_url=self.settings.zhipu_base_url,
                max_retries=1,
                http_client=self._http_client,
            )
        return self._client

    async def describe_image(self, data: bytes, media_type: str = "image/png") -> str:
        """返回 Markdown 表格文本；失败抛异常，由调用方决定是否忽略。"""
        import asyncio

        return await asyncio.to_thread(self._describe_sync, data, media_type)

    def _describe_sync(self, data: bytes, media_type: str) -> str:
        client = self._ensure_client()
        b64 = base64.b64encode(data).decode("ascii")

        resp = client.chat.completions.create(
            model=self.settings.zhipu_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{b64}"},
                        },
                        {"type": "text", "text": VISION_PROMPT},
                    ],
                }
            ],
            thinking={"type": "disabled"},
            max_tokens=2048,
        )
        return (resp.choices[0].message.content or "").strip()

    def describe_path(self, path: Path) -> str:
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        return self._describe_sync(path.read_bytes(), media_type)

    def close(self) -> None:
        self._client = None
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None
