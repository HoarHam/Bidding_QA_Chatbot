"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp, ExternalLink, ImageIcon } from "lucide-react";

import type { SourceRef } from "@/lib/types";

/**
 * 单条参考来源。
 *
 * 展示的是 `rerank_score`（CrossEncoder 相关性），**不是** `rrf_score`：
 * RRF 是排名融合的产物，量纲与余弦相似度不同，把它显示成百分比或置信度
 * 会误导用户。
 */
export default function SourceCard({ source, rank }: { source: SourceRef; rank: number }) {
  const [expanded, setExpanded] = useState(false);
  const [lightbox, setLightbox] = useState<string | null>(null);

  const score = source.rerank_score;

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3 transition hover:border-brand-500/50 hover:shadow-sm">
      <div className="mb-1.5 flex items-start gap-2">
        <span className="mt-0.5 inline-flex size-5 shrink-0 items-center justify-center rounded bg-slate-100 text-[11px] font-medium text-slate-600">
          {rank}
        </span>
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="flex-1 text-left"
        >
          <p className="text-[13px] font-medium text-slate-800">{source.question}</p>
        </button>
        {expanded ? (
          <ChevronUp className="mt-0.5 size-3.5 shrink-0 text-slate-400" />
        ) : (
          <ChevronDown className="mt-0.5 size-3.5 shrink-0 text-slate-400" />
        )}
      </div>

      <p className="mb-2 pl-7 text-[11px] text-slate-500">{source.source_label}</p>

      <p className={`pl-7 text-xs text-slate-600 ${expanded ? "" : "line-clamp-3"}`}>
        {source.snippet}
      </p>

      {source.images.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2 pl-7">
          {source.images.map((img) => (
            <button
              key={img.index}
              type="button"
              onClick={() => setLightbox(img.url)}
              className="group inline-flex items-center gap-1 rounded border border-slate-200 bg-slate-50 px-2 py-1 text-[11px] text-slate-600 hover:border-brand-500/60 hover:text-brand-700"
              title={img.description ?? "查看原图"}
            >
              <ImageIcon className="size-3" />
              图片 {img.index}
              <ExternalLink className="size-2.5 opacity-0 transition group-hover:opacity-100" />
            </button>
          ))}
        </div>
      )}

      {score !== null && (
        <div className="mt-2 flex items-center gap-2 pl-7">
          <div className="h-1 flex-1 overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full rounded-full bg-brand-500"
              style={{ width: `${Math.max(0, Math.min(1, score)) * 100}%` }}
            />
          </div>
          <span
            className="text-[10px] text-slate-400 tabular-nums"
            title="相关性得分（CrossEncoder）"
          >
            {score.toFixed(3)}
          </span>
        </div>
      )}

      {lightbox && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6"
          onClick={() => setLightbox(null)}
          role="presentation"
        >
          {/* 原图来自后端静态路由，用原生 img 而非 next/image（本地文件，无需优化） */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={lightbox}
            alt="来源原图"
            className="max-h-full max-w-full rounded-lg bg-white object-contain p-2"
          />
        </div>
      )}
    </div>
  );
}
