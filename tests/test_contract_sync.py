"""前后端线上格式一致性测试。

`frontend/lib/types.ts` 是 `src/bidding_qa_chatbot/schemas.py` 的**手工镜像**。
手工镜像会随时间漂移，而漂移的后果是运行时才暴露的类型错配，
所以这里用解析源码的方式把两边对齐起来。

不引入 codegen：只有 40 行左右的类型定义，生成器的维护成本高于收益。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from bidding_qa_chatbot import schemas

TYPES_TS = Path(__file__).resolve().parents[1] / "frontend" / "lib" / "types.ts"


def _ts_interface_fields(source: str, name: str) -> set[str]:
    """取出某个 TS interface 的字段名。"""
    match = re.search(rf"export interface {name} \{{(.*?)\n\}}", source, re.S)
    if match is None:
        return set()
    body = match.group(1)
    fields = set()
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith(("//", "/*", "*")):
            continue
        m = re.match(r"([A-Za-z_]\w*)\s*\??\s*:", line)
        if m:
            fields.add(m.group(1))
    return fields


@pytest.mark.skipif(not TYPES_TS.exists(), reason="前端尚未创建")
def test_frontend_types_mirror_backend_schemas():
    source = TYPES_TS.read_text(encoding="utf-8")

    # (TS 接口名, pydantic 模型)
    pairs = [
        ("ChatRequest", schemas.ChatRequest),
        ("ChatResponse", schemas.ChatResponse),
        ("SourceRef", schemas.SourceRef),
        ("SourceImage", schemas.SourceImage),
        ("SourceDetail", schemas.SourceDetail),
        ("Usage", schemas.Usage),
        ("AttemptModel", schemas.AttemptModel),
        ("SourcesEvent", schemas.SourcesEvent),
        ("ProviderEvent", schemas.ProviderEvent),
        ("DeltaEvent", schemas.DeltaEvent),
        ("ErrorEvent", schemas.ErrorEvent),
        ("DoneEvent", schemas.DoneEvent),
        ("RetrieveResponse", schemas.RetrieveResponse),
    ]

    problems: list[str] = []
    for ts_name, model in pairs:
        ts_fields = _ts_interface_fields(source, ts_name)
        assert ts_fields, f"types.ts 里找不到 interface {ts_name}"
        py_fields = set(model.model_fields)
        missing = py_fields - ts_fields
        extra = ts_fields - py_fields
        if missing:
            problems.append(f"{ts_name}: types.ts 缺少 {sorted(missing)}")
        if extra:
            problems.append(f"{ts_name}: types.ts 多出 {sorted(extra)}")

    assert not problems, "前后端类型不一致:\n  " + "\n  ".join(problems)


@pytest.mark.skipif(not TYPES_TS.exists(), reason="前端尚未创建")
def test_fallback_event_uses_wire_names():
    """FallbackEvent 在线上是 `from` / `to`（后端用 serialization_alias）。

    这条断言防止有人「顺手」把它们改成 from_provider / to_provider，
    那会让前端的 fallback 提示静默失效。
    """
    model = schemas.FallbackEvent(from_provider="a", to_provider="b", reason="r")
    wire = model.model_dump(by_alias=True, mode="json")
    assert "from" in wire and "to" in wire
    assert "from_provider" not in wire and "to_provider" not in wire

    source = TYPES_TS.read_text(encoding="utf-8")
    assert _ts_interface_fields(source, "FallbackEvent") == {"from", "to", "reason", "attempts"}


@pytest.mark.skipif(not TYPES_TS.exists(), reason="前端尚未创建")
def test_sse_event_names_match_documented_set():
    """SSE 事件名是前后端之间的硬约定，改动必须两边同步。"""
    source = TYPES_TS.read_text(encoding="utf-8")
    for ts_name in (
        "SourcesEvent",
        "ProviderEvent",
        "FallbackEvent",
        "DeltaEvent",
        "ErrorEvent",
        "DoneEvent",
    ):
        assert f"export interface {ts_name}" in source, f"types.ts 缺少 {ts_name}"

    # 后端实际发送的事件名
    from bidding_qa_chatbot.api import sse

    for event_name in ("sources", "provider", "fallback", "token", "reasoning", "error", "done"):
        frame = sse.encode_event(event_name, schemas.DeltaEvent(delta="x"))
        assert frame.startswith(f"event: {event_name}\n"), event_name
