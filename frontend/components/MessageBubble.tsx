"use client";

import { useState } from "react";
import {
  AlertCircle,
  Bot,
  Check,
  Copy,
  FileText,
  Loader2,
  RefreshCw,
  Search,
  ThumbsDown,
  ThumbsUp,
  User,
} from "lucide-react";

import ErrorBanner from "./ErrorBanner";
import Markdown from "./Markdown";
import ThinkingPanel from "./ThinkingPanel";
import type { AttemptModel, AttachmentOut } from "@/lib/types";

/** 一次工具调用及其结果，用于在回答上方展示「正在检索…」。 */
export interface ToolActivity {
  key: string;
  display: string;
  summary?: string;
  ok?: boolean;
  elapsedMs?: number;
  pending?: boolean;
}

export interface ChatTurn {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  tools?: ToolActivity[];
  streaming?: boolean;
  /** 用户消息携带的附件（元数据；OCR 文本只在服务端）。 */
  attachments?: AttachmentOut[];
  /** 本轮带附件、正在等待后端 OCR 解析。 */
  preparing?: boolean;
  /** 持久化后的消息 id（反馈/重新生成需要；新消息由 done 事件带回）。 */
  messageId?: string;
  /** 助手消息的用户反馈。 */
  feedback?: "up" | "down" | null;
  /** 是否为该会话最后一条助手消息（只有它可以重新生成）。 */
  isLastAssistant?: boolean;
  onRegenerate?: () => void;
  onFeedback?: (rating: "up" | "down") => void;
  error?: { message: string; partial: boolean; attempts: AttemptModel[] };
  provider?: string | null;
  model?: string | null;
  fallbackFrom?: string | null;
  onRetry?: () => void;
}

/**
 * 工具调用过程。
 *
 * 展示「模型正在做什么」——Agent 在给出答案前会先检索，这段过程如果完全不可见，
 * 用户会以为界面卡住了。进行中的调用带转圈图标，完成后显示结果条数。
 */
function ToolActivityList({ items }: { items: ToolActivity[] }) {
  if (!items.length) return null;
  return (
    <ul className="mb-2 space-y-1">
      {items.map((t) => (
        <li
          key={t.key}
          data-testid="tool-activity"
          className="flex items-center gap-1.5 rounded-md bg-slate-50 px-2 py-1 text-[11px] text-slate-500"
        >
          {t.pending ? (
            <Loader2 className="size-3 shrink-0 animate-spin text-brand-500" />
          ) : t.ok ? (
            <Search className="size-3 shrink-0 text-emerald-600" />
          ) : (
            <AlertCircle className="size-3 shrink-0 text-red-500" />
          )}
          <span className="truncate">{t.display}</span>
          {t.summary && (
            <span className={t.ok ? "ml-auto shrink-0 text-slate-400" : "ml-auto shrink-0 text-red-500"}>
              {t.summary}
            </span>
          )}
        </li>
      ))}
    </ul>
  );
}

export default function MessageBubble({ turn }: { turn: ChatTurn }) {
  const isUser = turn.role === "user";
  const [copied, setCopied] = useState(false);

  async function copyAnswer() {
    try {
      await navigator.clipboard.writeText(turn.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // 剪贴板不可用（非安全上下文等）时静默失败，不打断对话
    }
  }

  const actionButton =
    "inline-flex size-6 cursor-pointer items-center justify-center rounded-md text-slate-400 transition hover:bg-slate-100 hover:text-slate-700";

  return (
    <div
      // 供 UI 验证脚本稳定选取（比按文本匹配可靠），见 scripts/verify-ui.mjs
      data-role={turn.role}
      data-streaming={turn.streaming ? "true" : "false"}
      className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}
    >
      <div
        className={`mt-0.5 inline-flex size-7 shrink-0 items-center justify-center rounded-full ${
          isUser ? "bg-slate-200 text-slate-600" : "bg-brand-500 text-white"
        }`}
      >
        {isUser ? <User className="size-4" /> : <Bot className="size-4" />}
      </div>

      <div className={`min-w-0 flex-1 ${isUser ? "flex flex-col items-end" : ""}`}>
        <div
          className={`max-w-full rounded-xl px-3.5 py-2.5 ${
            isUser ? "bg-brand-500 text-white" : "border border-slate-200 bg-white"
          }`}
        >
          {isUser ? (
            <div>
              <p className="text-sm whitespace-pre-wrap">{turn.content}</p>
              {turn.attachments && turn.attachments.length > 0 && (
                <div className="mt-2 flex flex-wrap justify-end gap-1.5">
                  {turn.attachments.map((att) => (
                    <span
                      key={att.id}
                      data-testid="attachment-chip"
                      title={att.filename}
                      className="inline-flex max-w-full items-center gap-1 rounded-md bg-white/20 px-2 py-0.5 text-[11px] text-white/90"
                    >
                      <FileText className="size-3 shrink-0" />
                      <span className="truncate">{att.filename}</span>
                    </span>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <>
              <ThinkingPanel text={turn.reasoning ?? ""} streaming={Boolean(turn.streaming)} />
              <ToolActivityList items={turn.tools ?? []} />
              {turn.content ? (
                <Markdown content={turn.content} />
              ) : turn.streaming ? (
                turn.preparing ? (
                  <span className="inline-flex items-center gap-1.5 text-xs text-slate-400">
                    <Loader2 className="size-3 animate-spin" />
                    正在解析附件…
                  </span>
                ) : (
                  <span className="inline-block h-3.5 w-1.5 animate-pulse bg-slate-400 align-middle" />
                )
              ) : null}
            </>
          )}
        </div>

        {!isUser && !turn.streaming && turn.content && (
          <div
            data-testid="message-actions"
            className="mt-1 flex items-center gap-0.5 pl-1"
          >
            <button
              type="button"
              onClick={() => void copyAnswer()}
              title={copied ? "已复制" : "复制回答"}
              aria-label="复制回答"
              className={actionButton}
            >
              {copied ? (
                <Check className="size-3 text-emerald-600" />
              ) : (
                <Copy className="size-3" />
              )}
            </button>

            {turn.isLastAssistant && turn.messageId && (
              <button
                type="button"
                data-testid="regenerate"
                onClick={turn.onRegenerate}
                title="重新生成"
                aria-label="重新生成"
                className={actionButton}
              >
                <RefreshCw className="size-3" />
              </button>
            )}

            {turn.messageId && (
              <>
                <button
                  type="button"
                  data-testid="feedback-up"
                  aria-pressed={turn.feedback === "up"}
                  onClick={() => turn.onFeedback?.("up")}
                  title="有帮助"
                  aria-label="有帮助"
                  className={`${actionButton} ${
                    turn.feedback === "up" ? "bg-emerald-50 text-emerald-600" : ""
                  }`}
                >
                  <ThumbsUp className="size-3" />
                </button>
                <button
                  type="button"
                  data-testid="feedback-down"
                  aria-pressed={turn.feedback === "down"}
                  onClick={() => turn.onFeedback?.("down")}
                  title="没有帮助"
                  aria-label="没有帮助"
                  className={`${actionButton} ${
                    turn.feedback === "down" ? "bg-red-50 text-red-600" : ""
                  }`}
                >
                  <ThumbsDown className="size-3" />
                </button>
              </>
            )}

            {copied && <span className="text-[10px] text-slate-400">已复制</span>}
          </div>
        )}

        {!isUser && (turn.provider || turn.fallbackFrom) && (
          <div className="mt-1 flex flex-wrap items-center gap-2 pl-1 text-[10px] text-slate-400">
            {turn.provider && <span>由 {turn.model ?? turn.provider} 生成</span>}
            {turn.fallbackFrom && (
              <span className="rounded bg-amber-100 px-1.5 py-0.5 text-amber-700">
                已从 {turn.fallbackFrom} 切换到 {turn.provider}
              </span>
            )}
          </div>
        )}

        {turn.error && (
          <ErrorBanner
            message={turn.error.message}
            partial={turn.error.partial}
            attempts={turn.error.attempts}
            onRetry={turn.onRetry}
          />
        )}
      </div>
    </div>
  );
}
