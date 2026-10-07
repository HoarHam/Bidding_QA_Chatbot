/** 后端接口封装。所有路径都是相对的，经 Next 的开发代理转发，不需要 CORS。 */

import { consumeSSE, type SSEHandler } from "./sse";
import type {
  ChatRequest,
  RetrieveResponse,
  SourceDetail,
  StatsResponse,
} from "./types";

/**
 * 服务端渲染时没有「当前源」，必须用绝对地址。
 * 浏览器端一律走相对路径（同源 + 代理）。
 */
function baseUrl(): string {
  if (typeof window !== "undefined") return "";
  const origin =
    process.env.NEXT_PUBLIC_SITE_ORIGIN ??
    (process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : null) ??
    `http://127.0.0.1:${process.env.API_PORT ?? 3000}`;
  return origin;
}

export async function fetchStats(): Promise<StatsResponse | null> {
  try {
    const res = await fetch(`${baseUrl()}/api/stats`, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as StatsResponse;
  } catch {
    // 后端没起来时不能让 SSR 崩掉，页头降级显示即可。
    return null;
  }
}

export async function fetchSourceDetail(chunkId: string): Promise<SourceDetail | null> {
  try {
    const res = await fetch(`${baseUrl()}/api/sources/${encodeURIComponent(chunkId)}`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return (await res.json()) as SourceDetail;
  } catch {
    return null;
  }
}

export async function retrieveOnly(body: ChatRequest): Promise<RetrieveResponse | null> {
  try {
    const res = await fetch(`${baseUrl()}/api/retrieve`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) return null;
    return (await res.json()) as RetrieveResponse;
  } catch {
    return null;
  }
}

/** 发起流式问答。返回的 Promise 在流消费完毕后 resolve。 */
export async function streamChat(
  body: ChatRequest,
  handler: SSEHandler,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${baseUrl()}/api/chat/stream`, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  await consumeSSE(res, handler, signal);
}
