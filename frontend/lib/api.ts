/** 后端接口封装。所有路径都是相对的，经 Next 的开发代理转发，不需要 CORS。 */

import { consumeSSE, type SSEHandler } from "./sse";
import type {
  ChatRequest,
  RetrieveResponse,
  SourceDetail,
  StatsResponse,
} from "./types";

/**
 * 浏览器端一律走相对路径（同源，经 `app/api/[...path]/route.ts` 转发）。
 *
 * 服务端渲染时没有「当前源」，必须给绝对地址。这里**直接指向后端**而不是
 * 绕回 Next 自身：后者会形成自请求（Next 请求自己再转发一次），
 * 多一跳且在某些部署形态下可能挂住。
 *
 * 注意不要用 `API_PORT`——那是**后端**端口；早先误把它当前端端口用，
 * 只是碰巧没出问题。
 */
function baseUrl(): string {
  if (typeof window !== "undefined") return "";
  return process.env.API_ORIGIN ?? "http://127.0.0.1:8001";
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
