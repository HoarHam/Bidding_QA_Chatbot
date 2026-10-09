import type { NextConfig } from "next";

/**
 * 开发代理**不在这里**配置。
 *
 * `/api/*` 由路由处理器 `app/api/[...path]/route.ts` 转发到后端。
 * 曾经用这里的 `rewrites` 转发，但它会**缓冲 SSE 响应**——
 * 实测经 rewrites 的 token 帧全部在同一毫秒到达（跨度 0ms），
 * 直连后端跨度 511ms。表现就是答案不再流式、憋到最后一次性显示。
 * 路由处理器把上游 ReadableStream 原样透传，不做缓冲，才能保住流式效果。
 *
 * 注意 `.env` 里的 `API_HOST=0.0.0.0` 是**监听**地址，
 * 代理必须拨 127.0.0.1（见 route.ts 的 API_ORIGIN）。
 */
const nextConfig: NextConfig = {
  // Next 16 默认拦截跨来源对开发资源（/_next/hmr 等）的请求，以防未授权访问。
  // 它只放行 localhost 及其子域、以及 dev server 启动时使用的主机名；
  // 用 http://127.0.0.1:3000 打开时，127.0.0.1 不在放行列表里，
  // 于是客户端 JS 被拦下 → 整个前端失去交互（按钮、Enter、模式切换全部失效），
  // 而服务端渲染的 HTML 完全正常，页面「能打开」但是死的。
  //
  // 官方约定：条目只写**主机名**，不带协议、端口、路径（见
  // node_modules/next/dist/docs/.../allowedDevOrigins.md）。
  // localhost 本就在默认放行列表内，这里显式列出只是为了表明意图。
  allowedDevOrigins: ["127.0.0.1", "localhost"],
};

export default nextConfig;
