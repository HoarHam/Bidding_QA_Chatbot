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
  /** 用户是否在本轮开启联网搜索（前端开关）。仅当服务端也启用时才真正生效。 */
  web_search?: boolean;
  top_k?: number | null;
  history?: ChatMessage[];
  /** 续接已有会话时传入；为空则新建一个会话。 */
  conversation_id?: string | null;
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

/** 联网搜索的一条结果（与本地知识库来源分开，参考来源面板分区展示）。 */
export interface WebSource {
  url: string;
  title: string;
  snippet: string;
  /** 引擎给出的相关度（0..1 语义）；Exa 回退到 basic 工具时可能为 null。 */
  score: number | null;
  engine: "tavily" | "exa";
  published_date: string | null;
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
  conversation_id: string | null;
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
  /** 服务端是否具备联网搜索能力（开关 + 至少一个引擎的密钥就绪）。 */
  web_search_available: boolean;
}

// ---------------------------------------------------------------- SSE 载荷
export interface SourcesEvent {
  sources: SourceRef[];
  retrieved: number;
  elapsed_ms: number;
}

/** Agent 联网搜索结果。与 `sources` 事件分开推送，前端按类型分区渲染。 */
export interface WebSourcesEvent {
  sources: WebSource[];
  retrieved: number;
  elapsed_ms: number;
}

export interface ProviderEvent {
  provider: string;
  model: string;
  mode: Mode;
  thinking_enabled: boolean;
}

/** Agent 决定调用某个工具。前端据此显示「正在检索知识库…」。 */
export interface ToolCallEvent {
  name: string;
  arguments: Record<string, unknown>;
  display: string;
}

export interface ToolResultEvent {
  name: string;
  ok: boolean;
  elapsed_ms: number;
  summary: string;
  sources: SourceRef[];
  degraded: boolean;
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
  /** 本次问答所属会话。前端据此把后续提问接到同一会话上。 */
  conversation_id: string | null;
}

// ---------------------------------------------------------------- 账号
export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
  display_name?: string | null;
}

export interface LoginRequest {
  /** 用户名或邮箱 */
  username: string;
  password: string;
}

export interface UserOut {
  id: string;
  username: string;
  email: string;
  display_name: string | null;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: UserOut;
}

// ---------------------------------------------------------------- 对话历史
export interface MessageOut {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning: string | null;
  sources: SourceRef[];
  /** 联网搜索来源（与本地来源分开返回）。 */
  web_sources: WebSource[];
  provider: string | null;
  model: string | null;
  created_at: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface ConversationListResponse {
  items: ConversationSummary[];
  total: number;
}

export interface ConversationDetail {
  id: string;
  title: string;
  message_count: number;
  created_at: string;
  updated_at: string;
  messages: MessageOut[];
}

export interface SearchHit {
  conversation_id: string;
  conversation_title: string;
  message_id: string;
  role: "user" | "assistant";
  snippet: string;
  created_at: string;
}

export interface SearchResponse {
  query: string;
  hits: SearchHit[];
  total: number;
}
