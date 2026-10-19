"""配置层测试。

覆盖两件容易出错的事：
1. `override=False` 的语义（真实环境变量必须压过 .env，这是 CI 注入配置的基础）；
2. 密钥绝不通过 `repr` / 日志泄漏。
"""

from __future__ import annotations

import os

import pytest

from bidding_qa_chatbot.config import Settings, load_dotenv, validate_runtime


def _settings(**overrides) -> Settings:
    """一份「配置完整」的基准：链路里的每个 provider 都给了 key。"""
    base = {
        "qdrant_url": "http://x",
        "qdrant_api_key": "k",
        "deepseek_api_key": "sk-x",
        "zhipu_api_key": "id.secret",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_override_false_keeps_existing_env_var(tmp_path, monkeypatch):
    """已存在的环境变量不被 .env 覆盖——这正是 override=False 的意义。"""
    env_file = tmp_path / ".env"
    env_file.write_text("API_PORT=1111\n", encoding="utf-8")

    monkeypatch.setenv("API_PORT", "2222")
    load_dotenv(dotenv_path=env_file, override=False)
    assert os.environ["API_PORT"] == "2222"

    monkeypatch.setenv("API_PORT", "3333")
    # 模拟错误用法：override=True 会把外部注入的值冲掉
    load_dotenv(dotenv_path=env_file, override=True)
    assert os.environ["API_PORT"] == "1111"


def test_secret_does_not_leak_in_repr():
    s = _settings(deepseek_api_key="sk-super-secret-value")
    assert "sk-super-secret-value" not in repr(s)
    assert s.deepseek_api_key is not None
    assert s.deepseek_api_key.get_secret_value() == "sk-super-secret-value"


def test_fallback_chain_puts_primary_first_and_dedupes():
    s = _settings(llm_provider="zhipu", llm_fallback_chain="deepseek,zhipu,vllm")
    assert s.fallback_chain == ["zhipu", "deepseek", "vllm"]


def test_fallback_chain_handles_whitespace_and_empties():
    s = _settings(llm_provider="deepseek", llm_fallback_chain=" zhipu , , vllm ")
    assert s.fallback_chain == ["deepseek", "zhipu", "vllm"]


def test_model_for_selects_thinking_model():
    s = _settings(
        deepseek_model="fast-m",
        deepseek_thinking_model="think-m",
        zhipu_model="z-fast",
        zhipu_thinking_model="z-think",
    )
    assert s.model_for("deepseek", "fast") == "fast-m"
    assert s.model_for("deepseek", "thinking") == "think-m"
    assert s.model_for("zhipu", "fast") == "z-fast"
    assert s.model_for("zhipu", "thinking") == "z-think"


def test_model_for_unknown_provider_raises():
    with pytest.raises(ValueError, match="未知 provider"):
        _settings().model_for("nope", "fast")


def test_validate_runtime_reports_all_problems_at_once():
    """一次性汇总所有问题，而不是抛第一个就停。"""
    s = Settings(_env_file=None, qdrant_url=None, qdrant_vector_size=768, llm_provider="deepseek")
    problems = validate_runtime(s)
    joined = " | ".join(problems)
    assert "QDRANT_URL" in joined
    assert "QDRANT_VECTOR_SIZE" in joined
    assert "DEEPSEEK_API_KEY" in joined


def test_validate_runtime_clean_config_passes():
    assert validate_runtime(_settings()) == []


def test_validate_runtime_rejects_bad_distance():
    problems = validate_runtime(_settings(qdrant_distance="bogus"))
    assert any("QDRANT_DISTANCE" in p for p in problems)


def test_vector_size_mismatch_is_flagged():
    """512 维是 bge-small-zh-v1.5 的输出维度，改成别的值必然不匹配。"""
    problems = validate_runtime(_settings(qdrant_vector_size=384))
    assert any("512" in p for p in problems)


def test_hf_endpoint_exported_to_environ(monkeypatch):
    """HF_ENDPOINT 必须写进 os.environ——huggingface_hub 只认环境变量。"""
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    s = _settings(hf_endpoint="https://hf-mirror.com")
    assert s.hf_endpoint == "https://hf-mirror.com"
    assert os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com"


def test_validate_runtime_requires_web_search_keys_when_enabled():
    problems = validate_runtime(
        _settings(enable_web_search=True, tavily_api_key=None, exa_api_key=None)
    )
    assert any("TAVILY_API_KEY" in p for p in problems)
    assert any("EXA_API_KEY" in p for p in problems)


def test_validate_runtime_ignores_web_keys_when_disabled():
    problems = validate_runtime(_settings(enable_web_search=False))
    assert not [p for p in problems if "WEB_SEARCH" in p]
