"use client";

import { useCallback, useRef, useState } from "react";
import { BookOpen } from "lucide-react";

import Composer from "./Composer";
import MessageList from "./MessageList";
import SourcesPanel from "./SourcesPanel";
import type { ChatTurn } from "./MessageBubble";
import { streamChat } from "@/lib/api";
import type { AttemptModel, Mode, SourceRef } from "@/lib/types";

let turnSeq = 0;
const nextId = () => `turn-${++turnSeq}`;

/**
 * 聊天状态机。
 *
 * 两个刻意的设计：
 *
 * 1. **流一定从事件处理器里发起，绝不在 `useEffect` 里。**
 *    React 严格模式会双调用副作用，从 effect 里开流会导致同一条消息被请求两次。
 * 2. **中途失败保留已产出的正文。** 用户已经读过的文字不能被清掉，
 *    错误以提示条的形式出现在正文下方。
 */
export default function ChatShell() {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [sources, setSources] = useState<SourceRef[]>([]);
  const [retrieved, setRetrieved] = useState(0);
  const [mode, setMode] = useState<Mode>("fast");
  const [busy, setBusy] = useState(false);
  const [sourcesPending, setSourcesPending] = useState(false);
  const [mobileSourcesOpen, setMobileSourcesOpen] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const lastQueryRef = useRef<string>("");

  const patchTurn = useCallback((id: string, patch: Partial<ChatTurn>) => {
    setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t)));
  }, []);

  const send = useCallback(
    async (query: string, history: ChatTurn[]) => {
      lastQueryRef.current = query;

      const assistantId = nextId();
      setSources([]);
      setRetrieved(0);
      setSourcesPending(true);
      setTurns((prev) => [
        ...prev,
        ...history,
        { id: nextId(), role: "user", content: query },
        { id: assistantId, role: "assistant", content: "", streaming: true },
      ]);

      const controller = new AbortController();
      abortRef.current = controller;
      setBusy(true);

      let content = "";
      let reasoning = "";

      try {
        await streamChat(
          {
            query,
            mode,
            history: history
              .filter((t) => !t.error)
              .map((t) => ({ role: t.role, content: t.content }))
              .slice(-6),
          },
          {
            onSources: (e) => {
              setSources(e.sources);
              setRetrieved(e.retrieved);
              setSourcesPending(false);
            },
            onProvider: (e) => {
              patchTurn(assistantId, { provider: e.provider, model: e.model });
            },
            onFallback: (e) => {
              // 诚实告知用户答案来自备选模型
              patchTurn(assistantId, { fallbackFrom: e.from });
            },
            onReasoning: (e) => {
              reasoning += e.delta;
              patchTurn(assistantId, { reasoning });
            },
            onToken: (e) => {
              content += e.delta;
              patchTurn(assistantId, { content });
            },
            onError: (e) => {
              // 保留已产出的正文，错误显示在下方
              patchTurn(assistantId, {
                error: {
                  message: e.message,
                  partial: e.partial,
                  attempts: e.attempts ?? [],
                },
              });
            },
            onDone: (e) => {
              patchTurn(assistantId, {
                streaming: false,
                provider: e.provider,
                model: e.model,
              });
            },
            onHttpError: (status, body) => {
              const err = (body as { error?: { message?: string; attempts?: AttemptModel[] } })
                ?.error;
              patchTurn(assistantId, {
                streaming: false,
                error: {
                  message: err?.message ?? `请求失败（HTTP ${status}）`,
                  partial: false,
                  attempts: err?.attempts ?? [],
                },
              });
            },
          },
          controller.signal,
        );
      } catch (err) {
        // 主动取消不算错误
        if (!controller.signal.aborted) {
          patchTurn(assistantId, {
            streaming: false,
            error: {
              message: err instanceof Error ? err.message : "网络错误",
              partial: content.length > 0,
              attempts: [],
            },
          });
        }
      } finally {
        setSourcesPending(false);
        setBusy(false);
        abortRef.current = null;
        setTurns((prev) =>
          prev.map((t) => (t.id === assistantId ? { ...t, streaming: false } : t)),
        );
      }
    },
    [mode, patchTurn],
  );

  const handleSend = useCallback(
    (text: string) => {
      // 只带已完成且无错误的轮次作为历史
      const history = turns.filter((t) => t.role === "user" || t.content);
      void send(text, history);
    },
    [send, turns],
  );

  const handleRetry = useCallback(() => {
    const query = lastQueryRef.current;
    if (!query) return;
    const history = turns.filter((t) => t.role === "user" && t.content !== query);
    setTurns((prev) => prev.filter((t) => !t.error));
    void send(query, history);
  }, [send, turns]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setBusy(false);
  }, []);

  const turnsWithRetry = turns.map((t) =>
    t.error ? { ...t, onRetry: handleRetry } : t,
  );

  return (
    <div className="flex min-h-0 flex-1 gap-4">
      <section className="flex min-h-0 min-w-0 flex-1 flex-col">
        <MessageList turns={turnsWithRetry} />
        <div className="mt-3">
          <Composer
            onSend={handleSend}
            onStop={handleStop}
            busy={busy}
            mode={mode}
            onModeChange={setMode}
          />
        </div>
      </section>

      {/* 桌面端：右侧固定面板 */}
      <div className="hidden w-80 shrink-0 lg:block">
        <SourcesPanel sources={sources} retrieved={retrieved} pending={sourcesPending} />
      </div>

      {/* 移动端：折叠抽屉 */}
      <div className="lg:hidden">
        <button
          type="button"
          onClick={() => setMobileSourcesOpen(true)}
          className="fixed right-4 bottom-24 z-40 inline-flex items-center gap-1 rounded-full bg-brand-600 px-3 py-2 text-xs text-white shadow-lg"
        >
          <BookOpen className="size-3.5" />
          参考来源 {retrieved > 0 && `· ${retrieved}`}
        </button>

        {mobileSourcesOpen && (
          <div className="fixed inset-0 z-50 flex flex-col bg-white p-4 lg:hidden">
            <button
              type="button"
              onClick={() => setMobileSourcesOpen(false)}
              className="mb-2 self-end text-xs text-slate-500"
            >
              关闭
            </button>
            <SourcesPanel sources={sources} retrieved={retrieved} pending={sourcesPending} />
          </div>
        )}
      </div>
    </div>
  );
}
