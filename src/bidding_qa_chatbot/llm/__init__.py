"""LLM 接入层：可切换 provider + 自动回退（需求 4/5）。

对外只暴露 `build_chain()`；上层（API 层）不应该直接实例化任何具体 provider。
"""

from __future__ import annotations

import logging

from bidding_qa_chatbot.config import Settings, get_settings
from bidding_qa_chatbot.llm.base import (
    AllProvidersFailed,
    Attempt,
    FatalProviderError,
    LLMProvider,
    LLMRequest,
    LLMResult,
    MidStreamError,
    ProviderError,
    RetryableProviderError,
    StreamEnd,
    StreamEvent,
    Usage,
)
from bidding_qa_chatbot.llm.chain import ProviderChain

logger = logging.getLogger(__name__)

__all__ = [
    "AllProvidersFailed",
    "Attempt",
    "FatalProviderError",
    "LLMProvider",
    "LLMRequest",
    "LLMResult",
    "MidStreamError",
    "ProviderChain",
    "ProviderError",
    "RetryableProviderError",
    "StreamEnd",
    "StreamEvent",
    "Usage",
    "build_chain",
    "build_provider",
]


def build_provider(name: str, settings: Settings) -> LLMProvider:
    """按名字构造单个 provider。"""
    from bidding_qa_chatbot.llm.openai_compat import OpenAICompatProvider
    from bidding_qa_chatbot.llm.zhipu import ZhipuProvider

    if name == "deepseek":
        if settings.deepseek_api_key is None:
            raise FatalProviderError("deepseek", "DEEPSEEK_API_KEY 未配置")
        return OpenAICompatProvider(
            name="deepseek",
            base_url=settings.deepseek_base_url,
            api_key=settings.deepseek_api_key.get_secret_value(),
            settings=settings,
            supports_thinking_toggle=True,
            thinking_style="deepseek",
        )

    if name == "zhipu":
        if settings.zhipu_api_key is None:
            raise FatalProviderError("zhipu", "ZHIPU_API_KEY 未配置")
        return ZhipuProvider(settings)

    if name == "vllm":
        # 没有配 VLLM_MODEL 说明本地服务根本没部署，直接跳过这一环，
        # 而不是每次都去连一个必然超时的地址。
        if not settings.vllm_model:
            raise FatalProviderError("vllm", "VLLM_MODEL 未配置，跳过本地回退")
        return OpenAICompatProvider(
            name="vllm",
            base_url=settings.vllm_base_url,
            api_key=(
                settings.vllm_api_key.get_secret_value() if settings.vllm_api_key else "EMPTY"
            ),
            settings=settings,
            supports_thinking_toggle=False,
            connect_timeout=2.0,  # 本地连不上应当立刻失败
        )

    if name == "ollama":
        return OpenAICompatProvider(
            name="ollama",
            base_url=settings.ollama_base_url,
            api_key=(
                settings.ollama_api_key.get_secret_value() if settings.ollama_api_key else "ollama"
            ),
            settings=settings,
            supports_thinking_toggle=False,
            connect_timeout=2.0,
        )

    raise FatalProviderError(name, f"未知 provider: {name}")


def build_chain(settings: Settings | None = None) -> ProviderChain:
    settings = settings or get_settings()
    providers: list[LLMProvider] = []
    for name in settings.fallback_chain:
        try:
            providers.append(build_provider(name, settings))
        except FatalProviderError as exc:
            # 缺 key 或未部署的 provider 直接不进入链路，避免无谓的超时等待。
            logger.warning("跳过 provider %s: %s", name, exc.message)
    return ProviderChain(providers, settings)
