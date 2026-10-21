"use client";

import { useRef, useState } from "react";
import { Brain, FileText, Globe, Loader2, Paperclip, Send, Square, X } from "lucide-react";

import type { Mode } from "@/lib/types";

/** 输入区展示用的附件草稿（上传状态 + 元数据）。 */
export interface PendingAttachment {
  id?: string;
  filename: string;
  content_type?: string;
  size: number;
  status: "uploading" | "ready" | "error";
  error?: string;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

/** 示例问题。以真实文本渲染（不是 placeholder），因此**可以被选中复制**。
 *  前三条面向知识库检索（法规/期限类），后三条面向 Neo4j 知识图谱（交易事实/统计）。 */
const EXAMPLES = [
  "供应商针对单一来源异议被驳回，可以再次异议吗？",
  "投标保证金的退还期限是多久？",
  "质疑、询问的答复期限各是多少?",
  "有哪些关于空调的采购?",
  "采购最多的标的是什么?",
  "空调的供应商和采购人有哪些?",
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
  webSearch,
  onWebSearchChange,
  webSearchAvailable,
  attachment,
  onPickFile,
  onRemoveAttachment,
}: {
  onSend: (text: string, attachmentIds: string[]) => void;
  onStop: () => void;
  busy: boolean;
  mode: Mode;
  onModeChange: (m: Mode) => void;
  webSearch: boolean;
  onWebSearchChange: (enabled: boolean) => void;
  webSearchAvailable: boolean;
  attachment: PendingAttachment | null;
  onPickFile: (file: File) => void;
  onRemoveAttachment: () => void;
}) {
  const [text, setText] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const readyIds =
    attachment?.status === "ready" && attachment.id ? [attachment.id] : [];
  // 上传中 / 上传失败时禁止发送（避免把不存在的附件 id 发给后端）
  const attachmentBlocksSend = Boolean(attachment && attachment.status !== "ready");
  const canSend = text.trim().length > 0 && !busy && !attachmentBlocksSend;

  function submit() {
    const value = text.trim();
    if (!canSend) return;
    onSend(value, readyIds);
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

      {/* 隐藏文件选择器：上传只存储，OCR 在发送时由后端执行 */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".png,.jpg,.jpeg,.webp,.pdf,.docx,image/png,image/jpeg,image/webp,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = ""; // 允许再次选择同一个文件
          if (file) onPickFile(file);
        }}
      />

      {attachment && (
        <div
          data-testid="attachment-chip"
          title={attachment.error}
          className={`mb-1 flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] ${
            attachment.status === "error"
              ? "border-red-200 bg-red-50 text-red-700"
              : "border-slate-200 bg-slate-50 text-slate-600"
          }`}
        >
          <FileText className="size-3.5 shrink-0" />
          <span className="max-w-[16rem] truncate" title={attachment.filename}>
            {attachment.filename}
          </span>
          <span className="shrink-0 text-slate-400">{formatSize(attachment.size)}</span>
          {attachment.status === "uploading" && (
            <Loader2 className="size-3 shrink-0 animate-spin text-brand-500" />
          )}
          {attachment.status === "error" && (
            <span className="shrink-0 text-red-600">上传失败</span>
          )}
          <button
            type="button"
            data-testid="attachment-remove"
            aria-label="移除附件"
            title="移除附件"
            onClick={onRemoveAttachment}
            className="ml-0.5 shrink-0 cursor-pointer text-slate-400 hover:text-red-500"
          >
            <X className="size-3" />
          </button>
        </div>
      )}

      <div className="mt-1 flex items-center justify-between gap-2 px-1">
        <div className="flex items-center gap-1.5">
          {/* 深度思考开关：默认快速模式（未选中、置灰）；点一下切到深度思考，再点回快速 */}
          <button
            type="button"
            onClick={() => onModeChange(mode === "thinking" ? "fast" : "thinking")}
            aria-pressed={mode === "thinking"}
            title={
              mode === "thinking"
                ? "当前为深度思考模式，点击切回快速"
                : "开启深度思考（模型先推理再作答，更慢但更稳）"
            }
            className={`inline-flex cursor-pointer items-center gap-1 rounded-md border px-2 py-1 text-[11px] transition ${
              mode === "thinking"
                ? "border-sky-300 bg-sky-50 font-medium text-sky-700"
                : "border-slate-200 bg-white text-slate-500 hover:border-sky-300 hover:text-sky-700"
            }`}
          >
            <Brain className="size-3" />
            深度思考
          </button>

          {/* 联网搜索开关：服务端不具备能力时置灰并提示，避免「开了却静默无效」 */}
          <button
            type="button"
            onClick={() => onWebSearchChange(!webSearch)}
            aria-pressed={webSearch}
            disabled={!webSearchAvailable}
            title={
              webSearchAvailable
                ? "开启后允许模型联网检索公开信息（仅在需要时调用）"
                : "服务端未启用联网搜索"
            }
            className={`inline-flex items-center gap-1 rounded-md border px-2 py-1 text-[11px] transition disabled:cursor-not-allowed disabled:opacity-50 ${
              webSearch && webSearchAvailable
                ? "border-sky-300 bg-sky-50 font-medium text-sky-700"
                : webSearchAvailable
                  ? "border-slate-200 bg-white text-slate-500 hover:border-sky-300 hover:text-sky-700"
                  : "border-slate-200 bg-white text-slate-400"
            }`}
          >
            <Globe className="size-3" />
            联网搜索
          </button>
        </div>

        <div className="flex items-center gap-1.5">
          {/* 上传附件：仅图标，位于发送按钮左侧 */}
          <button
            type="button"
            aria-label="上传附件"
            title="上传附件（PNG/JPEG/WebP/PDF/DOCX，≤10MB）"
            disabled={busy || attachment?.status === "uploading"}
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex size-[30px] cursor-pointer items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-500 transition hover:border-brand-500/60 hover:text-brand-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {attachment?.status === "uploading" ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <Paperclip className="size-3.5" />
            )}
          </button>

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
              title={
                canSend
                  ? "发送（Enter）"
                  : attachment?.status === "uploading"
                    ? "附件上传中，请稍候"
                    : attachment?.status === "error"
                      ? "请先移除上传失败的附件"
                      : "请先输入问题"
              }
              className="inline-flex cursor-pointer items-center gap-1 rounded-lg bg-brand-600 px-3 py-1.5 text-xs text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
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
