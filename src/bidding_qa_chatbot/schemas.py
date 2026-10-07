"""API 与 SSE 的线上格式（wire format）唯一来源。

`frontend/lib/types.ts` 是这里的手工镜像。改动本文件后必须同步那边，
`tests/test_contract_sync.py` 会在两边字段名不一致时失败。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["fast", "thinking"]


# ---------------------------------------------------------------- 请求
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: Mode = "fast"
    top_k: int | None = Field(default=None, ge=1, le=50)
    # 只带最近若干轮；第一阶段的会话不落库，由前端持有。
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)
    session_id: str | None = None


# ---------------------------------------------------------------- 引用来源
class SourceImage(BaseModel):
    index: int
    url: str
    description: str | None = None


class SourceRef(BaseModel):
    chunk_id: str
    source_label: str
    question: str
    snippet: str
    excel_row: int
    # RRF 是排名融合的产物，量纲与余弦相似度不同，只能用来排序，
    # 不能当作置信度展示。UI 上展示的是 rerank_score。
    rrf_score: float
    rerank_score: float | None = None
    images: list[SourceImage] = Field(default_factory=list)


class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class RetrieveResponse(BaseModel):
    sources: list[SourceRef]
    elapsed_ms: int


class SourceDetail(BaseModel):
    """`GET /api/sources/{chunk_id}` 的完整记录（相比 SourceRef 多出完整答案）。"""

    chunk_id: str
    dataset: str
    source_label: str
    question: str
    answer: str
    excel_row: int
    images: list[SourceImage] = Field(default_factory=list)


# ---------------------------------------------------------------- 响应
class ChatResponse(BaseModel):
    answer: str
    reasoning: str | None = None
    sources: list[SourceRef]
    provider: str
    model: str
    finish_reason: str
    usage: Usage | None = None
    elapsed_ms: int
    request_id: str


class AttemptModel(BaseModel):
    provider: str
    model: str
    phase: Literal["connect", "first_token", "mid_stream"]
    error_type: str
    message: str
    elapsed_ms: int


class ApiError(BaseModel):
    code: str
    message: str
    attempts: list[AttemptModel] = Field(default_factory=list)


class ApiErrorResponse(BaseModel):
    error: ApiError


# ---------------------------------------------------------------- SSE 载荷
# 事件名 → data 载荷。前端按 event 名分发，不做字符串嗅探。
class SourcesEvent(BaseModel):
    sources: list[SourceRef]
    retrieved: int
    elapsed_ms: int


class ProviderEvent(BaseModel):
    provider: str
    model: str
    mode: Mode
    thinking_enabled: bool


class FallbackEvent(BaseModel):
    from_provider: str = Field(serialization_alias="from")
    to_provider: str = Field(serialization_alias="to")
    reason: str
    attempts: list[AttemptModel] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class DeltaEvent(BaseModel):
    delta: str


class ErrorEvent(BaseModel):
    code: str
    message: str
    # true 表示已有部分正文送达前端，不能再重试，否则会出现重复文本。
    partial: bool
    provider: str | None = None
    attempts: list[AttemptModel] = Field(default_factory=list)


class DoneEvent(BaseModel):
    finish_reason: str
    provider: str | None = None
    model: str | None = None
    usage: Usage | None = None
    elapsed_ms: int
    request_id: str


# ---------------------------------------------------------------- 健康检查
class DepHealth(BaseModel):
    name: str
    status: Literal["ok", "degraded", "unavailable", "not_configured"]
    latency_ms: int | None = None
    message: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    deps: list[DepHealth]


class StatsResponse(BaseModel):
    collection: str
    vector_size: int
    chunks: int | None = None
    provider_chain: list[str]
    embed_model: str
    rerank_model: str
    rerank_enabled: bool
