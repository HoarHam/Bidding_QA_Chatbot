"use client";

import { useState } from "react";
import { Brain, ChevronDown, ChevronRight } from "lucide-react";

/**
 * 深度思考面板，默认折叠。
 *
 * 思考文本**按纯文本渲染**，不走 Markdown：模型的思考过程经常包含
 * 未闭合的代码围栏或半截表格，用 Markdown 渲染会把整个布局吃掉。
 */
export default function ThinkingPanel({ text, streaming }: { text: string; streaming: boolean }) {
  const [open, setOpen] = useState(false);
  if (!text) return null;

  return (
    <div className="mb-2 overflow-hidden rounded-lg border border-amber-200 bg-amber-50/60">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1.5 px-3 py-2 text-left text-xs font-medium text-amber-800 hover:bg-amber-100/60"
      >
        {open ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
        <Brain className="size-3.5" />
        深度思考
        {streaming && <span className="ml-1 animate-pulse text-amber-600">思考中…</span>}
        <span className="ml-auto text-[10px] font-normal text-amber-600/80">
          {text.length} 字
        </span>
      </button>
      {open && (
        <pre className="scrollbar-thin max-h-64 overflow-auto border-t border-amber-200 bg-white/70 px-3 py-2 text-[11px] leading-relaxed whitespace-pre-wrap text-slate-600">
          {text}
        </pre>
      )}
    </div>
  );
}
