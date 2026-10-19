"""API 与 SSE 的线上格式（wire format）唯一来源。

`frontend/lib/types.ts` 是这里的手工镜像。改动本文件后必须同步那边，
`tests/test_contract_sync.py` 会在两边字段名不一致时失败。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field

Mode = Literal["fast", "thinking"]


# ---------------------------------------------------------------- 请求
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: Mode = "fast"
    #: 用户是否在本轮开启联网搜索（前端开关）。仅当服务端也启用时才真正生效。
    web_search: bool = False
    top_k: int | None = Field(default=None, ge=1, le=50)
    # 只带最近若干轮作为上下文；完整历史在服务端按 conversation_id 落库。
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)
    #: 续接已有会话时传入；为空则新建一个会话。
    conversation_id: str | None = None


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


class WebSource(BaseModel):
    """联网搜索的一条结果。

    与 `SourceRef`（本地知识库）刻意分开：联网结果没有 chunk_id / 精排分数，
    也不应该伪装成本地语料。前端「参考来源」面板按类型分区展示。
    """

    url: str
    title: str
    snippet: str
    #: 引擎给出的相关度（0..1 语义）。Exa 回退到 basic 工具时可能缺失。
    score: float | None = None
    engine: Literal["tavily", "exa"]
    published_date: str | None = None


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
    conversation_id: str | None = None


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


class WebSourcesEvent(BaseModel):
    """Agent 联网搜索结果。与 `sources` 事件分开推送，前端按类型分区渲染。"""

    sources: list[WebSource]
    retrieved: int
    elapsed_ms: int


class ProviderEvent(BaseModel):
    provider: str
    model: str
    mode: Mode
    thinking_enabled: bool


class ToolCallEvent(BaseModel):
    """Agent 决定调用某个工具。前端据此显示「正在检索知识库…」。"""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    display: str


class ToolResultEvent(BaseModel):
    """工具执行完毕。`sources` 是结构化的引用来源，供参考来源面板展示。"""

    name: str
    ok: bool
    elapsed_ms: int
    summary: str
    sources: list[SourceRef] = Field(default_factory=list)
    degraded: bool = False


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
    #: 本次问答所属会话。前端据此把后续提问接到同一会话上（跨会话恢复的关键）。
    conversation_id: str | None = None


# ---------------------------------------------------------------- 健康检查
class DepHealth(BaseModel):
    name: str
    status: Literal["ok", "degraded", "unavailable", "not_configured"]
    latency_ms: int | None = None
    message: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    deps: list[DepHealth]


# ---------------------------------------------------------------- 认证
class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_一-龥-]+$")
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str | None = Field(default=None, max_length=64)


class LoginRequest(BaseModel):
    #: 允许用用户名或邮箱登录
    username: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=128)


class UserOut(BaseModel):
    id: str
    username: str
    email: str
    display_name: str | None = None
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


# ---------------------------------------------------------------- 对话历史
class MessageOut(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    reasoning: str | None = None
    sources: list[SourceRef] = Field(default_factory=list)
    #: 联网搜索来源。落库时与本地来源同存于 messages.sources（带 kind="web" 标记），
    #: 读取时在这里分流，旧数据（无标记）仍按本地来源解析。
    web_sources: list[WebSource] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    created_at: datetime


class ConversationSummary(BaseModel):
    id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    items: list[ConversationSummary]
    total: int


class ConversationDetail(BaseModel):
    id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut]


class SearchHit(BaseModel):
    """一条命中搜索的消息，带上所属会话信息以便跳转。"""

    conversation_id: str
    conversation_title: str
    message_id: str
    role: Literal["user", "assistant"]
    snippet: str
    created_at: datetime


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    total: int


class StatsResponse(BaseModel):
    collection: str
    vector_size: int
    chunks: int | None = None
    provider_chain: list[str]
    embed_model: str
    rerank_model: str
    rerank_enabled: bool
    #: 服务端是否具备联网搜索能力（开关 + 至少一个引擎的密钥就绪）。
    #: 前端据此把「联网搜索」开关置灰，而不是让用户开了开关却静默无效。
    web_search_available: bool
