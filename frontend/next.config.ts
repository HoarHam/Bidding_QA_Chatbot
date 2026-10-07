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
