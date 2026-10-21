"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { BookOpen, Loader2 } from "lucide-react";

import Composer, { type PendingAttachment } from "./Composer";
import ConversationSidebar from "./ConversationSidebar";
import MessageList from "./MessageList";
import SourcesPanel from "./SourcesPanel";
import type { ChatTurn, ToolActivity, TurnTimings } from "./MessageBubble";
import {
  getConversation,
  setMessageFeedback,
  streamChat,
  uploadAttachment,
} from "@/lib/api";
import { clearSession, fetchMe, getToken } from "@/lib/auth";
import type {
  AttemptModel,
  AttachmentOut,
  MessageOut,
  MessageTimings,
  Mode,
  SourceRef,
  UserOut,
  WebSource,
} from "@/lib/types";

let turnSeq = 0;
const nextId = () => `turn-${++turnSeq}`;

function toTurnTimings(timings: MessageTimings | null | undefined): TurnTimings | undefined {
  if (!timings) return undefined;
  return {
    totalMs: timings.total_ms,
    thinkingMs: timings.thinking_ms,
    toolMs: timings.tool_ms,
    generationMs: timings.generation_ms,
  };
}

function toTurn(m: MessageOut): ChatTurn {
  return {
    id: m.id || nextId(),
    role: m.role,
    content: m.content,
    reasoning: m.reasoning ?? undefined,
    attachments: m.attachments ?? [],
    messageId: m.id,
    feedback: m.feedback,
    timings: toTurnTimings(m.timings),
    provider: m.provider,
    model: m.model,
  };
}

/**
 * 聊天状态机（需登录）。
 *
 * 几个刻意的设计：
 *
 * 1. **流一定从事件处理器里发起，绝不在 `useEffect` 里。**
 *    React 严格模式会双调用副作用，从 effect 里开流会导致同一条消息被请求两次。
 * 2. **中途失败保留已产出的正文。** 用户已经读过的文字不能被清掉。
 * 3. **会话 id 来自后端的 `done` 事件**，而不是前端自己生成。
 *    首轮对话的会话是在服务端建的（用户消息要先落库才能跑 Agent），
 *    前端拿回 id 后，后续提问才接得上同一个会话。
 */
export default function ChatShell({
  webSearchAvailable = false,
}: {
  /** 服务端是否具备联网搜索能力（由 page.tsx 从 /api/stats 注入）。 */
  webSearchAvailable?: boolean;
}) {
  const router = useRouter();

  const [user, setUser] = useState<UserOut | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [sidebarRefresh, setSidebarRefresh] = useState(0);

  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [sources, setSources] = useState<SourceRef[]>([]);
  const [webSources, setWebSources] = useState<WebSource[]>([]);
  const [mode, setMode] = useState<Mode>("fast");
  // 联网搜索开关：跨发送保持，直到用户主动关闭
  const [webSearch, setWebSearch] = useState(false);
  const [busy, setBusy] = useState(false);
  const [sourcesPending, setSourcesPending] = useState(false);
  const [mobileSourcesOpen, setMobileSourcesOpen] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  // 当前输入区里的附件草稿（发送前上传完成；OCR 在发送时由后端执行）
  const [attachment, setAttachment] = useState<PendingAttachment | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const lastQueryRef = useRef<string>("");
  const uploadAbortRef = useRef<AbortController | null>(null);
  const uploadTokenRef = useRef(0);
  const lastAttachmentIdsRef = useRef<string[]>([]);
  const lastAttachmentMetaRef = useRef<AttachmentOut[]>([]);

  // ---- 鉴权：没有 token 或 token 失效就回登录页 ----
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!getToken()) {
        router.replace("/login");
        return;
      }
      const me = await fetchMe();
      if (cancelled) return;
      if (!me) {
        router.replace("/login");
        return;
      }
      setUser(me);
      setAuthChecked(true);
    })();
    return () => {
      cancelled = true;
    };
  }, [router]);

  const patchTurn = useCallback((id: string, patch: Partial<ChatTurn>) => {
    setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t)));
  }, []);

  const handleLogout = useCallback(() => {
    clearSession();
    router.replace("/login");
  }, [router]);

  const clearAttachment = useCallback(() => {
    uploadAbortRef.current?.abort();
    uploadAbortRef.current = null;
    uploadTokenRef.current += 1;
    setAttachment(null);
  }, []);

  /** 选择文件后立即上传（只存储；解析在发送时进行）。 */
  const handlePickFile = useCallback(async (file: File) => {
    uploadAbortRef.current?.abort();
    const controller = new AbortController();
    uploadAbortRef.current = controller;
    const token = ++uploadTokenRef.current;

    setAttachment({ filename: file.name, size: file.size, status: "uploading" });
    try {
      const uploaded = await uploadAttachment(file, controller.signal);
      if (token !== uploadTokenRef.current) return;
      setAttachment({
        id: uploaded.id,
        filename: uploaded.filename,
        content_type: uploaded.content_type,
        size: uploaded.size,
        status: "ready",
      });
    } catch (err) {
      if (controller.signal.aborted || token !== uploadTokenRef.current) return;
      setAttachment({
        filename: file.name,
        size: file.size,
        status: "error",
        error: err instanceof Error ? err.message : "附件上传失败",
      });
    } finally {
      if (uploadAbortRef.current === controller) uploadAbortRef.current = null;
    }
  }, []);

  const handleRemoveAttachment = useCallback(() => {
    clearAttachment();
  }, [clearAttachment]);

  const handleNewConversation = useCallback(() => {
    abortRef.current?.abort();
    clearAttachment();
    lastAttachmentIdsRef.current = [];
    lastAttachmentMetaRef.current = [];
    setConversationId(null);
    setTurns([]);
    setSources([]);
    setWebSources([]);
  }, [clearAttachment]);

  const handleSelectConversation = useCallback(
    async (id: string) => {
      if (id === conversationId) return;
      abortRef.current?.abort();
      clearAttachment();
      lastAttachmentIdsRef.current = [];
      lastAttachmentMetaRef.current = [];
      setLoadingHistory(true);
      try {
        const detail = await getConversation(id);
        setConversationId(detail.id);
        const loaded = detail.messages.map(toTurn);
        setTurns(loaded);
        // 参考来源取最后一条助手消息的，与对话末尾展示的内容一致
        const lastAssistant = [...detail.messages].reverse().find((m) => m.role === "assistant");
        setSources(lastAssistant?.sources ?? []);
        setWebSources(lastAssistant?.web_sources ?? []);
      } catch {
        setTurns([]);
        setSources([]);
        setWebSources([]);
      } finally {
        setLoadingHistory(false);
      }
    },
    [conversationId, clearAttachment],
  );

  const streamInto = useCallback(
    async (
      assistantId: string,
      query: string,
      history: ChatTurn[],
      options: { attachmentIds?: string[]; regenerateMessageId?: string } = {},
    ) => {
      const attachmentIds = options.attachmentIds ?? [];
      const startedIn = conversationId;

      // 分阶段用时（浏览器侧测量，包含网络与排队）
      const startedAt = performance.now();
      let firstTokenAt: number | null = null;
      let toolElapsedMs = 0;

      const controller = new AbortController();
      abortRef.current = controller;
      setBusy(true);

      let content = "";
      let reasoning = "";
      let tools: ToolActivity[] = [];

      try {
        await streamChat(
          {
            query,
            mode,
            web_search: webSearch,
            attachment_ids: attachmentIds,
            conversation_id: startedIn,
            regenerate_message_id: options.regenerateMessageId,
            history: history
              .filter((t) => !t.error)
              .map((t) => ({ role: t.role, content: t.content }))
              .slice(-6),
          },
          {
            onSources: (e) => {
              setSources(e.sources);
              setSourcesPending(false);
            },
            onWebSources: (e) => {
              setWebSources(e.sources);
              setSourcesPending(false);
            },
            onProvider: (e) => {
              // 能走到 provider 事件，说明附件 OCR 已完成
              patchTurn(assistantId, { provider: e.provider, model: e.model, preparing: false });
            },
            onFallback: (e) => {
              patchTurn(assistantId, { fallbackFrom: e.from });
            },
            onReasoning: (e) => {
              reasoning += e.delta;
              patchTurn(assistantId, { reasoning });
            },
            onToolCall: (e) => {
              tools = [
                ...tools,
                { key: `tool-${tools.length}`, display: e.display, pending: true },
              ];
              patchTurn(assistantId, { tools });
            },
            onToolResult: (e) => {
              toolElapsedMs += Math.max(0, e.elapsed_ms);
              tools = tools.map((t, i) =>
                i === tools.length - 1
                  ? { ...t, summary: e.summary, ok: e.ok, elapsedMs: e.elapsed_ms, pending: false }
                  : t,
              );
              patchTurn(assistantId, { tools });
            },
            onToken: (e) => {
              if (firstTokenAt === null) firstTokenAt = performance.now();
              content += e.delta;
              patchTurn(assistantId, { content, preparing: false });
            },
            onError: (e) => {
              patchTurn(assistantId, {
                error: { message: e.message, partial: e.partial, attempts: e.attempts ?? [] },
              });
            },
            onDone: (e) => {
              const doneAt = performance.now();
              const totalMs = Math.max(0, doneAt - startedAt);
              const generationMs =
                firstTokenAt === null ? 0 : Math.max(0, doneAt - firstTokenAt);
              const thinkingMs = Math.max(
                0,
                totalMs - toolElapsedMs - generationMs,
              );
              patchTurn(assistantId, {
                streaming: false,
                provider: e.provider,
                model: e.model,
                messageId: e.message_id ?? undefined,
                // 优先使用后端落库的用时（与历史回载一致）；旧后端没有时退回本地测量
                timings: toTurnTimings(e.timings) ?? {
                  totalMs,
                  thinkingMs,
                  toolMs: toolElapsedMs,
                  generationMs,
                },
              });
              // 会话 id 由后端在建会话时生成，这里接住它，后续提问才能接上同一会话
              if (e.conversation_id && e.conversation_id !== startedIn) {
                setConversationId(e.conversation_id);
              }
              // 消息已落库，让侧栏刷新出这条会话
              setSidebarRefresh((n) => n + 1);
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
          prev.map((t) =>
            t.id === assistantId ? { ...t, streaming: false, preparing: false } : t,
          ),
        );
      }
    },
    [conversationId, mode, webSearch, patchTurn],
  );

  const send = useCallback(
    async (
      query: string,
      history: ChatTurn[],
      attachmentIds: string[] = [],
      attachmentMeta: AttachmentOut[] = [],
    ) => {
      lastQueryRef.current = query;
      lastAttachmentIdsRef.current = attachmentIds;
      lastAttachmentMetaRef.current = attachmentMeta;
      const assistantId = nextId();

      setSources([]);
      setWebSources([]);
      setSourcesPending(true);
      setTurns((prev) => [
        ...prev,
        { id: nextId(), role: "user", content: query, attachments: attachmentMeta },
        {
          id: assistantId,
          role: "assistant",
          content: "",
          streaming: true,
          preparing: attachmentIds.length > 0,
        },
      ]);

      await streamInto(assistantId, query, history, { attachmentIds });
    },
    [streamInto],
  );

  const handleSend = useCallback(
    (text: string, attachmentIds: string[]) => {
      const history = turns.filter((t) => t.role === "user" || t.content);
      const attachmentMeta: AttachmentOut[] =
        attachment?.status === "ready" && attachment.id
          ? [
              {
                id: attachment.id,
                filename: attachment.filename,
                content_type: attachment.content_type ?? "",
                size: attachment.size,
              },
            ]
          : [];
      void send(text, history, attachmentIds, attachmentMeta);
      // 发送后清空输入区 chip；重试仍可通过 lastAttachment*Ref 找回
      if (attachmentIds.length > 0) setAttachment(null);
    },
    [send, turns, attachment],
  );

  /** 重新生成：只允许最后一条助手消息，原地替换（后端更新同一条消息）。 */
  const handleRegenerate = useCallback(
    (turn: ChatTurn) => {
      if (busy || turn.role !== "assistant" || !turn.messageId) return;
      const index = turns.findIndex((t) => t.id === turn.id);
      if (index < 0) return;
      const precedingUser = [...turns.slice(0, index)]
        .reverse()
        .find((t) => t.role === "user" && t.content.trim());
      if (!precedingUser) return;

      const history = turns
        .slice(0, index)
        .filter((t) => t.role === "user" || t.content);
      setSources([]);
      setWebSources([]);
      setSourcesPending(true);
      patchTurn(turn.id, {
        content: "",
        reasoning: undefined,
        tools: [],
        error: undefined,
        feedback: undefined,
        timings: undefined,
        streaming: true,
        preparing: false,
      });
      void streamInto(turn.id, precedingUser.content, history, {
        regenerateMessageId: turn.messageId,
      });
    },
    [busy, turns, patchTurn, streamInto],
  );

  const handleFeedback = useCallback(
    (turn: ChatTurn, rating: "up" | "down") => {
      if (!turn.messageId) return;
      const previous = turn.feedback ?? null;
      // 再次点击同一评价 = 清除
      const next = previous === rating ? null : rating;
      patchTurn(turn.id, { feedback: next });
      void setMessageFeedback(turn.messageId, next).catch(() => {
        patchTurn(turn.id, { feedback: previous });
      });
    },
    [patchTurn],
  );

  const handleRetry = useCallback(() => {
    const query = lastQueryRef.current;
    if (!query) return;
    const history = turns.filter((t) => t.role === "user" && t.content !== query);
    setTurns((prev) => prev.filter((t) => !t.error));
    void send(query, history, lastAttachmentIdsRef.current, lastAttachmentMetaRef.current);
  }, [send, turns]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setBusy(false);
  }, []);

  if (!authChecked || !user) {
    return (
      <div className="flex flex-1 items-center justify-center text-sm text-slate-400">
        <Loader2 className="mr-2 size-4 animate-spin" />
        正在校验登录状态…
      </div>
    );
  }

  const lastAssistantId = [...turns].reverse().find((t) => t.role === "assistant")?.id;
  const turnsWithRetry = turns.map((t) => {
    const decorated: ChatTurn = { ...t, isLastAssistant: t.id === lastAssistantId };
    if (t.error) decorated.onRetry = handleRetry;
    if (t.role === "assistant") {
      decorated.onRegenerate = () => handleRegenerate(t);
      decorated.onFeedback = (rating) => handleFeedback(t, rating);
    }
    return decorated;
  });
  const sourceCount = sources.length + webSources.length;

  return (
    <div className="flex min-h-0 flex-1 gap-4">
      <div className="hidden md:flex">
        <ConversationSidebar
          user={user}
          activeId={conversationId}
          onSelect={(id) => void handleSelectConversation(id)}
          onNew={handleNewConversation}
          onLogout={handleLogout}
          refreshKey={sidebarRefresh}
        />
      </div>

      <section className="flex min-h-0 min-w-0 flex-1 flex-col">
        {loadingHistory ? (
          <div className="flex flex-1 items-center justify-center text-sm text-slate-400">
            <Loader2 className="mr-2 size-4 animate-spin" />
            正在加载历史对话…
          </div>
        ) : (
          <MessageList turns={turnsWithRetry} />
        )}
        <div className="mt-3">
          <Composer
            onSend={handleSend}
            onStop={handleStop}
            busy={busy}
            mode={mode}
            onModeChange={setMode}
            webSearch={webSearch}
            onWebSearchChange={setWebSearch}
            webSearchAvailable={webSearchAvailable}
            attachment={attachment}
            onPickFile={handlePickFile}
            onRemoveAttachment={handleRemoveAttachment}
          />
        </div>
      </section>

      <div className="hidden w-80 shrink-0 lg:block">
        <SourcesPanel sources={sources} webSources={webSources} pending={sourcesPending} />
      </div>

      <div className="lg:hidden">
        <button
          type="button"
          onClick={() => setMobileSourcesOpen(true)}
          className="fixed right-4 bottom-24 z-40 inline-flex cursor-pointer items-center gap-1 rounded-full bg-brand-600 px-3 py-2 text-xs text-white shadow-lg"
        >
          <BookOpen className="size-3.5" />
          参考来源 {sourceCount > 0 && `· ${sourceCount}`}
        </button>

        {mobileSourcesOpen && (
          <div className="fixed inset-0 z-50 flex flex-col bg-white p-4 lg:hidden">
            <button
              type="button"
              onClick={() => setMobileSourcesOpen(false)}
              className="mb-2 cursor-pointer self-end text-xs text-slate-500"
            >
              关闭
            </button>
            <SourcesPanel sources={sources} webSources={webSources} pending={sourcesPending} />
          </div>
        )}
      </div>
    </div>
  );
}
