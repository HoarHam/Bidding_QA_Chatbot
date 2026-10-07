/**
 * 后端 `src/bidding_qa_chatbot/schemas.py` 的手工镜像。
 *
 * 字段名与线上一一对应（snake_case）。改动 schemas.py 后必须同步这里——
 * 后端有 `tests/test_contract_sync.py` 会比较两边字段名，漏改会失败。
 */

export type Mode = "fast" | "thinking";

// ---------------------------------------------------------------- 请求
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  query: string;
  mode?: Mode;
  top_k?: number | null;
  history?: ChatMessage[];
  session_id?: string | null;
}

// ---------------------------------------------------------------- 引用来源
export interface SourceImage {
  index: number;
  url: string;
  description: string | null;
}

export interface SourceRef {
  chunk_id: string;
  source_label: string;
  question: string;
  snippet: string;
  excel_row: number;
  /** RRF 是排名融合产物，量纲与余弦不同——只用于排序，不要当置信度展示。 */
  rrf_score: number;
  /** 展示给用户的相关性分数（CrossEncoder）。 */
  rerank_score: number | null;
  images: SourceImage[];
}

export interface SourceDetail {
  chunk_id: string;
  dataset: string;
  source_label: string;
  question: string;
  answer: string;
  excel_row: number;
  images: SourceImage[];
}

export interface Usage {
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
}

// ---------------------------------------------------------------- 响应
export interface RetrieveResponse {
  sources: SourceRef[];
  elapsed_ms: number;
}

export interface ChatResponse {
  answer: string;
  reasoning: string | null;
  sources: SourceRef[];
  provider: string;
  model: string;
  finish_reason: string;
  usage: Usage | null;
  elapsed_ms: number;
  request_id: string;
}

export interface AttemptModel {
  provider: string;
  model: string;
  phase: "connect" | "first_token" | "mid_stream";
  error_type: string;
  message: string;
  elapsed_ms: number;
}

export interface ApiError {
  code: string;
  message: string;
  attempts: AttemptModel[];
}

export interface ApiErrorResponse {
  error: ApiError;
}

// ---------------------------------------------------------------- 健康检查
export interface DepHealth {
  name: string;
  status: "ok" | "degraded" | "unavailable" | "not_configured";
  latency_ms: number | null;
  message: string | null;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  deps: DepHealth[];
}

export interface StatsResponse {
  collection: string;
  vector_size: number;
  /** 后端为 `int | None`：集合不存在或读取失败时为 null。 */
  chunks: number | null;
  provider_chain: string[];
  embed_model: string;
  rerank_model: string;
  rerank_enabled: boolean;
}

// ---------------------------------------------------------------- SSE 载荷
export interface SourcesEvent {
  sources: SourceRef[];
  retrieved: number;
  elapsed_ms: number;
}

export interface ProviderEvent {
  provider: string;
  model: string;
  mode: Mode;
  thinking_enabled: boolean;
}

/** 线上字段是 `from` / `to`（后端用 serialization_alias 输出）。 */
export interface FallbackEvent {
  from: string;
  to: string;
  reason: string;
  attempts: AttemptModel[];
}

export interface DeltaEvent {
  delta: string;
}

export interface ErrorEvent {
  code: string;
  message: string;
  /** true 表示已有部分正文送达，不能重试，否则会出现重复文本。 */
  partial: boolean;
  provider: string | null;
  attempts: AttemptModel[];
}

export interface DoneEvent {
  finish_reason: string;
  provider: string | null;
  model: string | null;
  usage: Usage | null;
  elapsed_ms: number;
  request_id: string;
}
