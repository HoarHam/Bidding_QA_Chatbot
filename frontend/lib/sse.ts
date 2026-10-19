/**
 * SSE 解析。
 *
 * 为什么不用浏览器原生的 `EventSource`：它**只能发 GET，不能带请求体**，
 * 而这里是 POST + JSON。所以用 `fetch` + `ReadableStream` 手动解析。
 *
 * 手动解析要处理几件容易被忽略的事：
 * - 一次 `read()` 可能拿到半个帧，必须跨读取缓冲；
 * - 可能一次拿到多个帧，要按 `\n\n` 切开；
 * - `:` 开头是注释行（后端每 15 秒发一次 `: ping` 心跳），要忽略；
 * - 换行可能是 CRLF。
 */

import type {
  DeltaEvent,
  DoneEvent,
  ErrorEvent,
  FallbackEvent,
  ProviderEvent,
  SourcesEvent,
  ToolCallEvent,
  ToolResultEvent,
  WebSourcesEvent,
} from "./types";

export type SSEHandler = {
  onSources?: (e: SourcesEvent) => void;
  onWebSources?: (e: WebSourcesEvent) => void;
  onProvider?: (e: ProviderEvent) => void;
  onFallback?: (e: FallbackEvent) => void;
  onReasoning?: (e: DeltaEvent) => void;
  onToken?: (e: DeltaEvent) => void;
  onToolCall?: (e: ToolCallEvent) => void;
  onToolResult?: (e: ToolResultEvent) => void;
  onError?: (e: ErrorEvent) => void;
  onDone?: (e: DoneEvent) => void;
  /** HTTP 非 200 时收到的 JSON 错误体（不是 SSE）。 */
  onHttpError?: (status: number, body: unknown) => void;
};

interface Frame {
  event: string;
  data: string;
}

/** 解析单个 SSE 帧。返回 null 表示是心跳或空帧。 */
function parseFrame(raw: string): Frame | null {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of raw.split(/\r?\n/)) {
    if (line === "" || line.startsWith(":")) continue; // 空行 / 心跳注释
    const colon = line.indexOf(":");
    if (colon === -1) continue;
    const field = line.slice(0, colon);
    // 规范规定冒号后的单个空格要去掉
    const value = line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
  }

  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}

function dispatch(frame: Frame, handler: SSEHandler): void {
  let payload: unknown;
  try {
    payload = JSON.parse(frame.data);
  } catch {
    return; // 坏帧直接跳过，不影响后续
  }

  switch (frame.event) {
    case "sources":
      handler.onSources?.(payload as SourcesEvent);
      break;
    case "web_sources":
      handler.onWebSources?.(payload as WebSourcesEvent);
      break;
    case "provider":
      handler.onProvider?.(payload as ProviderEvent);
      break;
    case "fallback":
      handler.onFallback?.(payload as FallbackEvent);
      break;
    case "reasoning":
      handler.onReasoning?.(payload as DeltaEvent);
      break;
    case "tool_call":
      handler.onToolCall?.(payload as ToolCallEvent);
      break;
    case "tool_result":
      handler.onToolResult?.(payload as ToolResultEvent);
      break;
    case "token":
      handler.onToken?.(payload as DeltaEvent);
      break;
    case "error":
      handler.onError?.(payload as ErrorEvent);
      break;
    case "done":
      handler.onDone?.(payload as DoneEvent);
      break;
    default:
      break;
  }
}

/**
 * 消费一个 SSE 响应流。
 *
 * `signal.aborted` 为真时静默返回，不触发 onError——用户主动「停止」
 * 不应该看到一条错误。
 */
export async function consumeSSE(
  response: Response,
  handler: SSEHandler,
  signal?: AbortSignal,
): Promise<void> {
  // 非 200：后端返回的是 JSON 错误体，不能当 SSE 解析。
  if (!response.ok) {
    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      body = await response.text().catch(() => null);
    }
    handler.onHttpError?.(response.status, body);
    return;
  }

  if (!response.body) {
    handler.onHttpError?.(response.status, { error: { message: "响应没有正文流" } });
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;

      // stream:true —— 多字节 UTF-8 字符可能被切在两次读取之间，中文尤其常见。
      buffer += decoder.decode(value, { stream: true });

      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? ""; // 最后一段可能是半个帧，留到下次

      for (const raw of frames) {
        const frame = parseFrame(raw);
        if (frame) dispatch(frame, handler);
      }
    }

    // 收尾：处理没有以空行结束的最后一帧
    if (buffer.trim()) {
      const frame = parseFrame(buffer);
      if (frame) dispatch(frame, handler);
    }
  } catch (err) {
    if (signal?.aborted) return; // 主动取消，不算错误
    throw err;
  }
}
