"use client";

import { ExternalLink, Globe } from "lucide-react";

import type { WebSource } from "@/lib/types";

function hostname(url: string): string {
  try {
    return new URL(url).hostname;
  } catch {
    return url;
  }
}

/**
 * 单条联网搜索结果。
 *
 * 与本地 `SourceCard` 的区别：标题本身是可跳转外链，展示引擎给出的相关度
 * （不是 CrossEncoder 分数），并标出引擎（Tavily 关键词 / Exa 语义）。
 */
export default function WebSourceCard({
  source,
  rank,
}: {
  source: WebSource;
  rank: number;
}) {
  const score = source.score;

  return (
    <div
      data-testid="web-source-card"
      className="rounded-lg border border-sky-200 bg-white p-3 transition hover:border-sky-400/70 hover:shadow-sm"
    >
      <div className="mb-1 flex items-start gap-2">
        <span className="mt-0.5 inline-flex size-5 shrink-0 items-center justify-center rounded bg-sky-100 text-[11px] font-medium text-sky-700">
          {rank}
        </span>
        <a
          href={source.url}
          target="_blank"
          rel="noreferrer"
          title={source.url}
          className="flex-1 text-[13px] font-medium text-blue-600 underline underline-offset-2 hover:text-blue-700"
        >
          <span className="line-clamp-2">{source.title || source.url}</span>
        </a>
        <ExternalLink className="mt-0.5 size-3.5 shrink-0 text-slate-400" />
      </div>

      <p className="mb-2 truncate pl-7 text-[11px] text-slate-500">{hostname(source.url)}</p>

      <p className="line-clamp-3 pl-7 text-xs text-slate-600">{source.snippet}</p>

      <div className="mt-2 flex items-center gap-2 pl-7">
        <span
          className={`inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 text-[10px] ${
            source.engine === "tavily"
              ? "bg-emerald-50 text-emerald-700"
              : "bg-violet-50 text-violet-700"
          }`}
        >
          <Globe className="size-2.5" />
          {source.engine === "tavily" ? "Tavily 关键词" : "Exa 语义"}
        </span>
        {source.published_date && (
          <span className="text-[10px] text-slate-400">{source.published_date}</span>
        )}
        <span className="flex-1" />
        {score !== null && (
          <span className="inline-flex items-center gap-1.5">
            <span className="h-1 w-10 overflow-hidden rounded-full bg-slate-100">
              <span
                className="block h-full rounded-full bg-sky-500"
                style={{ width: `${Math.max(0, Math.min(1, score)) * 100}%` }}
              />
            </span>
            <span
              className="text-[10px] text-slate-400 tabular-nums"
              title="相关度（搜索引擎返回）"
            >
              {score.toFixed(3)}
            </span>
          </span>
        )}
      </div>
    </div>
  );
}
