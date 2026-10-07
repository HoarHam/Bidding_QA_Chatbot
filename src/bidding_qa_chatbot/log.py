"""日志与密钥脱敏。

`.env` 里放的是真实可用的 API key，任何一条把 key 打进日志/异常/健康检查响应的代码
都是事故。这里用两层防护：

1. `redact()` —— 把已知密钥值以及常见密钥形态（`sk-` 开头、JWT）替换掉。
2. `RedactionFilter` —— 挂在 root logger 上，兜住所有遗漏的直接 `logger.info(...)`。
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

# 常见密钥形态：OpenAI/DeepSeek 的 sk-*，JWT（Qdrant key 就是 JWT），
# 智谱的 {32位hex}.{16位} 形式。
_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}"),
    re.compile(r"\b[0-9a-f]{32}\.[A-Za-z0-9_\-]{10,}"),
    re.compile(r"\btvly-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
]

_REDACTED = "***REDACTED***"


def redact(text: str) -> str:
    """抹掉文本中的密钥形态。用于异常消息、健康检查输出、日志。"""
    if not text:
        return text
    out = text
    for pat in _PATTERNS:
        out = pat.sub(_REDACTED, out)
    return out


def redact_obj(obj: Any) -> Any:
    """递归脱敏容器，便于直接塞进日志或 JSON 响应。"""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact_obj(v) for v in obj)
    return obj


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            record.args = redact_obj(record.args)  # type: ignore[assignment]
        return True


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
    )
    handler.addFilter(RedactionFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # 这几个库在 INFO 级别非常吵，且可能带出请求头。
    for noisy in ("httpx", "httpcore", "urllib3", "qdrant_client", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
