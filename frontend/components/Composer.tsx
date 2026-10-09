"use client";

import { useState } from "react";
import { Brain, Send, Square, Zap } from "lucide-react";

import type { Mode } from "@/lib/types";

/** 示例问题。以真实文本渲染（不是 placeholder），因此**可以被选中复制**。 */
const EXAMPLES = [
  "供应商针对单一来源异议被驳回，可以再次异议吗？",
  "投标保证金的退还期限是多久？",
  "质疑、询问的答复期限各是多少?",
];

/**
 * 输入区。
 *
 * 三个曾经踩过的坑，改动时注意：
 *
 * 1. **示例问题不能放在 `placeholder` 里**——placeholder 按浏览器规范不可选中，
 *    用户没法复制。这里用可点击的文本块，既可选中复制，点一下也能直接填入。
 * 2. **不要再加第二个写着「发送」的元素**。之前有一个纯提示文字「⏎ 发送」，
 *    外观像按钮却点不动，和真按钮并排极易混淆。提示语并进 placeholder。
 * 3. **Enter 的组字判断不要用「自己维护的 ref」**。之前用 onCompositionStart/End
 *    维护一个 ref，一旦 compositionend 没触发（中文输入法下并不罕见），
 *    这个 ref 会永远停在 true，导致 Enter 再也不能发送——只能换行。
 *    改用原生事件上的 `isComposing`（权威且不会卡死），并兼容 keyCode 229。
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
  const canSend = text.trim().length > 0 && !busy;

  function submit() {
    const value = text.trim();
    if (!value || busy) return;
    onSend(value);
    setText("");
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-2 shadow-sm">
      {/* 示例问题：真实文本，可选中复制；点击填入输入框 */}
      {text.length === 0 && (
        <div className="mb-1.5 flex flex-wrap items-center gap-1.5 px-1">
          <span className="text-[11px] text-slate-400">示例：</span>
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => setText(ex)}
              title="点击填入输入框（也可以直接选中复制）"
              className="cursor-pointer rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] text-slate-600 select-text transition hover:border-brand-500/60 hover:bg-brand-50 hover:text-brand-700"
            >
              {ex}
            </button>
          ))}
        </div>
      )}

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key !== "Enter" || e.shiftKey) return; // Shift+Enter 换行
          // 组字中（中文输入法选词）时回车不算发送。
          // isComposing 由浏览器维护，比自建 ref 可靠；229 是部分浏览器组字时的 keyCode。
          if (e.nativeEvent.isComposing || e.nativeEvent.keyCode === 229) return;
          e.preventDefault();
          submit();
        }}
        rows={2}
        placeholder="输入招投标相关问题…（Enter 发送，Shift+Enter 换行）"
        className="scrollbar-thin max-h-40 w-full resize-none bg-transparent px-2 py-1.5 text-sm outline-none placeholder:text-slate-400"
      />

      <div className="mt-1 flex items-center justify-between gap-2 px-1">
        <div className="flex items-center gap-1 rounded-lg bg-slate-100 p-0.5">
          <button
            type="button"
            onClick={() => onModeChange("fast")}
            aria-pressed={mode === "fast"}
            className={`cursor-pointer rounded-md px-2 py-1 text-[11px] transition ${
              mode === "fast"
                ? "bg-white font-medium text-slate-800 shadow-sm"
                : "text-slate-500 hover:text-slate-700"
            }`}
          >
            <span className="inline-flex items-center gap-1">
              <Zap className="size-3" />
              快速
            </span>
          </button>
          <button
            type="button"
            onClick={() => onModeChange("thinking")}
            aria-pressed={mode === "thinking"}
            className={`cursor-pointer rounded-md px-2 py-1 text-[11px] transition ${
              mode === "thinking"
                ? "bg-white font-medium text-slate-800 shadow-sm"
                : "text-slate-500 hover:text-slate-700"
            }`}
          >
            <span className="inline-flex items-center gap-1">
              <Brain className="size-3" />
              深度思考
            </span>
          </button>
        </div>

        {busy ? (
          <button
            type="button"
            onClick={onStop}
            className="inline-flex cursor-pointer items-center gap-1 rounded-lg bg-slate-700 px-3 py-1.5 text-xs text-white hover:bg-slate-800"
          >
            <Square className="size-3" />
            停止
          </button>
        ) : (
          <button
            type="button"
            onClick={submit}
            disabled={!canSend}
            title={canSend ? "发送（Enter）" : "请先输入问题"}
            className="inline-flex cursor-pointer items-center gap-1 rounded-lg bg-brand-600 px-3 py-1.5 text-xs text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            <Send className="size-3" />
            发送
          </button>
        )}
      </div>
    </div>
  );
}
