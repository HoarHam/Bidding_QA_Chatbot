"use client";

import { Bot, User } from "lucide-react";

import ErrorBanner from "./ErrorBanner";
import Markdown from "./Markdown";
import ThinkingPanel from "./ThinkingPanel";
import type { AttemptModel } from "@/lib/types";

export interface ChatTurn {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  streaming?: boolean;
  error?: { message: string; partial: boolean; attempts: AttemptModel[] };
  provider?: string | null;
  model?: string | null;
  fallbackFrom?: string | null;
  onRetry?: () => void;
}

export default function MessageBubble({ turn }: { turn: ChatTurn }) {
  const isUser = turn.role === "user";

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
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
            <p className="text-sm whitespace-pre-wrap">{turn.content}</p>
          ) : (
            <>
              <ThinkingPanel text={turn.reasoning ?? ""} streaming={Boolean(turn.streaming)} />
              {turn.content ? (
                <Markdown content={turn.content} />
              ) : turn.streaming ? (
                <span className="inline-block h-3.5 w-1.5 animate-pulse bg-slate-400 align-middle" />
              ) : null}
            </>
          )}
        </div>

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
