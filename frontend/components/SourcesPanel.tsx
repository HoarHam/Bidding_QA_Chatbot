"use client";

import { BookOpen } from "lucide-react";

import SourceCard from "./SourceCard";
import type { SourceRef } from "@/lib/types";

/**
 * 参考来源面板（需求 8 的硬性要求）。
 *
 * `sources` 事件在**任何回答 token 之前**到达，所以这个面板会在答案还在流式输出时
 * 就已经填充完毕——这是刻意的体验设计：即使 LLM 全部挂掉，用户依然看得到检索结果。
 */
export default function SourcesPanel({
  sources,
  retrieved,
  pending,
}: {
  sources: SourceRef[];
  retrieved: number;
  pending: boolean;
}) {
  return (
    <aside className="flex h-full min-h-0 flex-col">
      <div className="mb-2 flex items-center gap-2 text-sm font-medium text-slate-700">
        <BookOpen className="size-4" />
        参考来源
        {retrieved > 0 && (
          <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-normal text-slate-600">
            {retrieved}
          </span>
        )}
      </div>

      <div className="scrollbar-thin min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
        {sources.length === 0 && (
          <p className="rounded-lg border border-dashed border-slate-200 p-4 text-center text-xs text-slate-400">
            {pending ? "检索中…" : "未检索到相关知识"}
          </p>
        )}
        {sources.map((s, i) => (
          <SourceCard key={s.chunk_id} source={s} rank={i + 1} />
        ))}
      </div>
    </aside>
  );
}
