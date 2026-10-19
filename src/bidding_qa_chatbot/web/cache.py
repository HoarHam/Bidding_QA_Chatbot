"""进程内 TTL 缓存（异步安全）。

只用于联网搜索结果的短时复用：不落盘、不跨进程。`asyncio.Lock` 串行化读写，
避免并发请求同时改写字典。键由调用方拼装（引擎 + 条数 + 归一化查询）。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any


class TTLCache:
    def __init__(self, ttl: float, maxsize: int) -> None:
        self.ttl = ttl
        self.maxsize = max(1, maxsize)
        self._items: dict[str, tuple[float, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """命中返回缓存值；过期或不存在返回 None。"""
        async with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= time.monotonic():
                self._items.pop(key, None)
                return None
            return value

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            now = time.monotonic()
            expired = [k for k, (expires_at, _) in self._items.items() if expires_at <= now]
            for k in expired:
                self._items.pop(k, None)
            while len(self._items) >= self.maxsize:
                oldest = min(self._items, key=lambda k: self._items[k][0])
                self._items.pop(oldest, None)
            self._items[key] = (now + self.ttl, value)

    async def clear(self) -> None:
        async with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
