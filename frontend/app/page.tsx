import { Database, Cpu, Layers } from "lucide-react";

import ChatShell from "@/components/ChatShell";
import { fetchStats } from "@/lib/api";

/**
 * Server Component：真正走一次 App Router 的 SSR。
 *
 * `fetchStats` 失败时返回 null（后端没起、集合不存在等），
 * 页头降级展示——SSR 不该因为后端不可用而整页崩溃。
 */
export default async function Page() {
  const stats = await fetchStats();

  return (
    <main className="mx-auto flex h-screen max-w-7xl flex-col px-4 py-4">
      <header className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-3">
        <div>
          <h1 className="text-lg font-semibold">招投标智能问答机器人</h1>
          <p className="text-xs text-slate-500">
            混合检索（稠密 + BM25）+ 精排 + 可切换大模型
          </p>
        </div>

        {stats ? (
          <div className="flex flex-wrap items-center gap-3 text-xs text-slate-600">
            <span className="inline-flex items-center gap-1">
              <Database className="size-3.5" />
              {stats.collection} · {stats.chunks ?? "?"} 条 · {stats.vector_size} 维
            </span>
            <span className="inline-flex items-center gap-1">
              <Layers className="size-3.5" />
              {stats.provider_chain.join(" → ")}
            </span>
            <span className="inline-flex items-center gap-1">
              <Cpu className="size-3.5" />
              {stats.rerank_enabled ? "已启用精排" : "未启用精排"}
            </span>
          </div>
        ) : (
          <span className="text-xs text-amber-600">
            未连接到后端（请确认 API 服务已在 8001 端口启动）
          </span>
        )}
      </header>

      <ChatShell webSearchAvailable={stats?.web_search_available ?? false} />
    </main>
  );
}
