import type { NextConfig } from "next";

/**
 * 开发代理：把 /api/* 转发到 Python 后端。
 *
 * 保留 `/api` 前缀，这样前端所有请求都是相对路径，浏览器看来是同源，
 * 不需要任何 CORS 配置。
 *
 * 注意 `.env` 里的 `API_HOST=0.0.0.0` 是**监听**地址，代理必须拨 127.0.0.1
 * （localhost 会解析到它）。所以这里不使用 API_HOST。
 */
const API_ORIGIN = process.env.API_ORIGIN ?? "http://127.0.0.1:8001";

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

  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${API_ORIGIN}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
