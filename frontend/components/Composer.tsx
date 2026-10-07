"use client";

import { useRef, useState } from "react";
import { Brain, CornerDownLeft, Send, Square, Zap } from "lucide-react";

import type { Mode } from "@/lib/types";

/**
 * 输入区。
 *
 * 一个中文 UI 必须处理的细节：**输入法组字状态**。
 * 用中文输入法打字时，回车是「选词确认」，不应该触发发送。
 * 靠 `e.nativeEvent.isComposing` 判断。
 */
export default function Composer({
  onSend,
  onStop,
  busy,
  mode,
  onModeChange,
}: {
  onSend: (text: string) => void;
  onStop: () => void;
  busy: boolean;
  mode: Mode;
  onModeChange: (m: Mode) => void;
}) {
  const [text, setText] = useState("");
  const composingRef = useRef(false);

  function submit() {
    const value = text.trim();
    if (!value || busy) return;
    onSend(value);
    setText("");
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-2 shadow-sm">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onCompositionStart={() => (composingRef.current = true)}
        onCompositionEnd={() => (composingRef.current = false)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            // 组字状态下回车是选词，不能当发送
            if (composingRef.current || e.nativeEvent.isComposing) return;
            e.preventDefault();
            submit();
          }
        }}
        rows={2}
        placeholder="请输入招投标相关问题，例如：供应商针对单一来源异议被驳回，可以再次异议吗？（Enter 发送，Shift+Enter 换行）"
        className="scrollbar-thin max-h-40 w-full resize-none bg-transparent px-2 py-1.5 text-sm outline-none placeholder:text-slate-400"
      />

      <div className="mt-1 flex items-center justify-between gap-2 px-1">
        <div className="flex items-center gap-1 rounded-lg bg-slate-100 p-0.5">
          <button
            type="button"
            onClick={() => onModeChange("fast")}
            className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] transition ${
              mode === "fast"
                ? "bg-white text-slate-800 shadow-sm"
                : "text-slate-500 hover:text-slate-700"
            }`}
          >
            <Zap className="size-3" />
            快速
          </button>
          <button
            type="button"
            onClick={() => onModeChange("thinking")}
            className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] transition ${
              mode === "thinking"
                ? "bg-white text-slate-800 shadow-sm"
                : "text-slate-500 hover:text-slate-700"
            }`}
          >
            <Brain className="size-3" />
            深度思考
          </button>
        </div>

        <div className="flex items-center gap-2">
          <span className="hidden text-[10px] text-slate-400 sm:inline">
            <CornerDownLeft className="inline size-3" /> 发送
          </span>
          {busy ? (
            <button
              type="button"
              onClick={onStop}
              className="inline-flex items-center gap-1 rounded-lg bg-slate-700 px-3 py-1.5 text-xs text-white hover:bg-slate-800"
            >
              <Square className="size-3" />
              停止
            </button>
          ) : (
            <button
              type="button"
              onClick={submit}
              disabled={!text.trim()}
              className="inline-flex items-center gap-1 rounded-lg bg-brand-600 px-3 py-1.5 text-xs text-white hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              <Send className="size-3" />
              发送
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
