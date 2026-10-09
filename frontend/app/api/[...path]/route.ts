/**
 * 后端代理（catch-all）。
 *
 * ## 为什么不用 next.config 的 rewrites
 *
 * 一度用 `rewrites` 把 `/api/*` 转发到后端，功能上没问题，**但它会缓冲 SSE 响应**：
 * 实测经 rewrites 的 44 个 token 帧全部在同一毫秒到达（跨度 0ms），
 * 而直连后端跨度 511ms、帧是散开到达的。用户看到的就是「答案憋到最后一次性蹦出来」，
 * 思考过程和正文一起出现——完全失去流式效果。
 *
 * 所以改为 route handler：把上游的 `ReadableStream` **原样**交给 Response，
 * 不做任何读取/拼接，字节到达即转发。
 *
 * Next 16 注意：`params` 现在是 Promise，必须 await；`dynamic` / `revalidate` 等
 * 段配置在启用 Cache Components 后已被移除，这里不依赖它们。
 */

import type { NextRequest } from "next/server";

/** 后端地址。服务端渲染时也走这里，避免绕回 Next 自身形成自请求。 */
export const API_ORIGIN = process.env.API_ORIGIN ?? "http://127.0.0.1:8001";

export const runtime = "nodejs";

//: 逐跳首部，不应转发（RFC 9110）。特别是 content-length：
//: 流式响应长度未知，透传会让客户端一直等一个永远凑不满的响应体。
const HOP_BY_HOP = [
  "connection",
  "keep-alive",
  "transfer-encoding",
  "te",
  "trailer",
  "upgrade",
  "content-length",
  "content-encoding",
  "host",
];

async function proxy(request: NextRequest, params: { path: string[] }): Promise<Response> {
  const target = `${API_ORIGIN}/api/${params.path.join("/")}${request.nextUrl.search}`;

  const headers = new Headers(request.headers);
  for (const h of HOP_BY_HOP) headers.delete(h);

  const hasBody = !["GET", "HEAD"].includes(request.method);

  const upstream = await fetch(target, {
    method: request.method,
    headers,
    // 原样转发请求体流，不先读进内存（对话历史可能很大）
    body: hasBody ? request.body : undefined,
    // Node 的 fetch 要求流式请求体显式声明 duplex
    duplex: "half",
    redirect: "manual",
    cache: "no-store",
  } as RequestInit);

  const responseHeaders = new Headers(upstream.headers);
  for (const h of HOP_BY_HOP) responseHeaders.delete(h);
  // 双保险：禁止中间层缓冲（nginx 等认这个头）
  responseHeaders.set("X-Accel-Buffering", "no");
  if (upstream.headers.get("content-type")?.includes("text/event-stream")) {
    responseHeaders.set("Cache-Control", "no-cache");
    responseHeaders.set("Connection", "keep-alive");
  }

  // 关键：直接把上游的 body 流转出去，不 await、不 concat、不 transform。
  // 一旦写成 `await upstream.arrayBuffer()` 之类的缓冲式读取，
  // 流式就会退化成「憋到最后一次性显示」——UI 验证脚本会报
  // 「中间态样本 0 个」把它拦下来（已用负向测试确认该检查有效）。
  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: responseHeaders,
  });
}

type Ctx = { params: Promise<{ path: string[] }> };

export async function GET(request: NextRequest, ctx: Ctx) {
  return proxy(request, await ctx.params);
}

export async function POST(request: NextRequest, ctx: Ctx) {
  return proxy(request, await ctx.params);
}

export async function PUT(request: NextRequest, ctx: Ctx) {
  return proxy(request, await ctx.params);
}

export async function PATCH(request: NextRequest, ctx: Ctx) {
  return proxy(request, await ctx.params);
}

export async function DELETE(request: NextRequest, ctx: Ctx) {
  return proxy(request, await ctx.params);
}
