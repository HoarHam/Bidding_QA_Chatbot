"use client";

import { BookOpen, Globe } from "lucide-react";

import SourceCard from "./SourceCard";
import WebSourceCard from "./WebSourceCard";
import type { SourceRef, WebSource } from "@/lib/types";

/**
 * 参考来源面板（需求 8 的硬性要求）。
 *
 * 本地知识库与联网搜索**分区展示**：两者来源类型不同（语料 chunk vs 外部网页），
 * 评分含义也不同（CrossEncoder rerank vs 搜索引擎相关度），混在一列会让用户误解。
 *
 * `sources` / `web_sources` 事件在答案 token 之前或工具调用后到达，
 * 面板会在答案还在流式输出时就已填充——即使 LLM 全部挂掉，用户依然看得到检索结果。
 */
export default function SourcesPanel({
  sources,
  webSources,
  pending,
}: {
  sources: SourceRef[];
  webSources: WebSource[];
  pending: boolean;
}) {
  const total = sources.length + webSources.length;

  return (
    <aside className="flex h-full min-h-0 flex-col">
      <div className="mb-2 flex items-center gap-2 text-sm font-medium text-slate-700">
        <BookOpen className="size-4" />
        参考来源
        {total > 0 && (
          <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-normal text-slate-600">
            {total}
          </span>
        )}
      </div>

      <div className="scrollbar-thin min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
        {total === 0 && (
          <p className="rounded-lg border border-dashed border-slate-200 p-4 text-center text-xs text-slate-400">
            {pending ? "检索中…" : "未检索到相关知识"}
          </p>
        )}

        {sources.length > 0 && (
          <section className="space-y-2">
            <h3 className="flex items-center gap-1 px-1 text-[11px] font-medium text-slate-500">
              <BookOpen className="size-3" />
              本地知识库
              <span className="rounded-full bg-slate-100 px-1.5 text-[10px] font-normal text-slate-500">
                {sources.length}
              </span>
            </h3>
            {sources.map((s, i) => (
              <SourceCard key={s.chunk_id} source={s} rank={i + 1} />
            ))}
          </section>
        )}

        {webSources.length > 0 && (
          <section className="space-y-2 rounded-lg border border-sky-100 bg-sky-50/50 p-2">
            <h3 className="flex items-center gap-1 px-1 text-[11px] font-medium text-sky-700">
              <Globe className="size-3" />
              联网搜索
              <span className="rounded-full bg-sky-100 px-1.5 text-[10px] font-normal text-sky-700">
                {webSources.length}
              </span>
            </h3>
            {webSources.map((s, i) => (
              <WebSourceCard key={s.url} source={s} rank={i + 1} />
            ))}
          </section>
        )}
      </div>
    </aside>
  );
}
