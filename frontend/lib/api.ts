/** 后端接口封装。所有路径都是相对的，经 Next 的开发代理转发，不需要 CORS。 */

import { authHeaders } from "./auth";
import { consumeSSE, type SSEHandler } from "./sse";
import type {
  AttachmentOut,
  ChatRequest,
  ConversationDetail,
  ConversationListResponse,
  RetrieveResponse,
  SearchResponse,
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
      headers: { "content-type": "application/json", ...authHeaders() },
      body: JSON.stringify(body),
    });
    if (!res.ok) return null;
    return (await res.json()) as RetrieveResponse;
  } catch {
    return null;
  }
}


// ---------------------------------------------------------------- 对话历史
export async function listConversations(): Promise<ConversationListResponse> {
  const res = await fetch(`${baseUrl()}/api/conversations`, {
    headers: authHeaders(),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`加载会话列表失败（HTTP ${res.status}）`);
  return (await res.json()) as ConversationListResponse;
}

export async function getConversation(id: string): Promise<ConversationDetail> {
  const res = await fetch(`${baseUrl()}/api/conversations/${encodeURIComponent(id)}`, {
    headers: authHeaders(),
    cache: "no-store",
  });
  if (res.status === 404) throw new Error("会话不存在或无权访问");
  if (!res.ok) throw new Error(`加载会话失败（HTTP ${res.status}）`);
  return (await res.json()) as ConversationDetail;
}

export async function deleteConversation(id: string): Promise<void> {
  const res = await fetch(`${baseUrl()}/api/conversations/${encodeURIComponent(id)}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  // 204 是正常返回；404 说明已经不存在，对调用方而言等同于删除成功
  if (!res.ok && res.status !== 404) {
    throw new Error(`删除失败（HTTP ${res.status}）`);
  }
}

export async function searchConversations(query: string): Promise<SearchResponse> {
  const res = await fetch(
    `${baseUrl()}/api/conversations/search?q=${encodeURIComponent(query)}`,
    { headers: authHeaders(), cache: "no-store" },
  );
  if (!res.ok) throw new Error(`搜索失败（HTTP ${res.status}）`);
  return (await res.json()) as SearchResponse;
}

/**
 * 上传附件（发送前暂存，OCR 在发送时由后端执行）。
 *
 * **不要手动设置 content-type**：multipart 的 boundary 必须由浏览器生成。
 */
export async function uploadAttachment(
  file: File,
  signal?: AbortSignal,
): Promise<AttachmentOut> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${baseUrl()}/api/attachments`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
    signal,
  });

  if (!res.ok) {
    let message = `上传失败（HTTP ${res.status}）`;
    try {
      const body = (await res.json()) as {
        error?: { message?: string };
        detail?: string;
      };
      message = body.error?.message ?? body.detail ?? message;
    } catch {
      // 非 JSON 错误体：保留默认文案
    }
    throw new Error(message);
  }
  return (await res.json()) as AttachmentOut;
}

/** 发起流式问答。返回的 Promise 在流消费完毕后 resolve。 */
export async function streamChat(
  body: ChatRequest,
  handler: SSEHandler,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${baseUrl()}/api/chat/stream`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      accept: "text/event-stream",
      ...authHeaders(),
    },
    body: JSON.stringify(body),
    signal,
  });
  await consumeSSE(res, handler, signal);
}
