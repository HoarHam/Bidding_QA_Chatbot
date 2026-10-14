"use client";

import { useCallback, useEffect, useState } from "react";
import { Loader2, MessageSquarePlus, Search, Trash2, X } from "lucide-react";

import { deleteConversation, listConversations, searchConversations } from "@/lib/api";
import type { ConversationSummary, SearchHit, UserOut } from "@/lib/types";

/**
 * 会话侧栏：历史列表 + 搜索 + 删除。
 *
 * 两条独立的列表状态：正常浏览时显示会话列表，一旦输入搜索词就切换成命中结果。
 * 没有把两者合并成一条，因为它们的语义不同——一个是「有哪些对话」，
 * 一个是「哪句话里提到过这个词」，混在一起用户分不清点进去会看到什么。
 */
export default function ConversationSidebar({
  user,
  activeId,
  onSelect,
  onNew,
  onLogout,
  refreshKey,
}: {
  user: UserOut;
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onLogout: () => void;
  /** 变化时重新拉取列表（新会话产生、消息落库后由父组件触发） */
  refreshKey: number;
}) {
  const [items, setItems] = useState<ConversationSummary[]>([]);
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listConversations();
      setItems(data.items);
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load, refreshKey]);

  // 搜索加 300ms 防抖：每敲一个字就请求一次既浪费也会让结果闪烁
  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setHits(null);
      return;
    }
    const timer = setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await searchConversations(q);
        setHits(data.hits);
      } catch (e) {
        setError(e instanceof Error ? e.message : "搜索失败");
      } finally {
        setLoading(false);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [query]);

  async function handleDelete(id: string, e: React.MouseEvent) {
    e.stopPropagation(); // 别触发选中
    if (!window.confirm("删除这个对话？该操作不可恢复。")) return;
    try {
      await deleteConversation(id);
      setItems((prev) => prev.filter((c) => c.id !== id));
      setHits((prev) => prev?.filter((h) => h.conversation_id !== id) ?? null);
      if (id === activeId) onNew();
    } catch (err) {
      setError(err instanceof Error ? err.message : "删除失败");
    }
  }

  const searching = query.trim().length > 0;

  return (
    <aside className="flex h-full min-h-0 w-64 shrink-0 flex-col border-r border-slate-200 pr-3">
      <button
        type="button"
        onClick={onNew}
        className="mb-3 flex w-full cursor-pointer items-center justify-center gap-1.5 rounded-lg bg-brand-600 px-3 py-2 text-xs text-white transition hover:bg-brand-700"
      >
        <MessageSquarePlus className="size-3.5" />
        新对话
      </button>

      <div className="relative mb-2">
        <Search className="absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-slate-400" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="搜索历史对话…"
          className="w-full rounded-lg border border-slate-200 py-1.5 pr-7 pl-7 text-xs outline-none focus:border-brand-500"
        />
        {query && (
          <button
            type="button"
            onClick={() => setQuery("")}
            className="absolute top-1/2 right-2 -translate-y-1/2 cursor-pointer text-slate-400 hover:text-slate-600"
            aria-label="清空搜索"
          >
            <X className="size-3.5" />
          </button>
        )}
      </div>

      <div className="scrollbar-thin min-h-0 flex-1 space-y-1 overflow-y-auto">
        {error && (
          <p className="rounded-md bg-red-50 px-2 py-1.5 text-[11px] text-red-700">{error}</p>
        )}

        {loading && (
          <p className="flex items-center justify-center gap-1 py-3 text-[11px] text-slate-400">
            <Loader2 className="size-3 animate-spin" />
            加载中…
          </p>
        )}

        {!loading && searching && hits?.length === 0 && (
          <p className="px-2 py-3 text-center text-[11px] text-slate-400">没有匹配的历史记录</p>
        )}

        {!loading && !searching && items.length === 0 && (
          <p className="px-2 py-3 text-center text-[11px] text-slate-400">还没有历史对话</p>
        )}

        {searching
          ? hits?.map((h) => (
              <button
                key={h.message_id}
                type="button"
                onClick={() => onSelect(h.conversation_id)}
                data-testid="search-hit"
                className={`group w-full cursor-pointer rounded-lg px-2 py-1.5 text-left transition hover:bg-slate-100 ${
                  h.conversation_id === activeId ? "bg-slate-100" : ""
                }`}
              >
                <p className="truncate text-[11px] font-medium text-slate-700">
                  {h.conversation_title}
                </p>
                <p className="mt-0.5 line-clamp-2 text-[11px] text-slate-500">{h.snippet}</p>
              </button>
            ))
          : items.map((c) => (
              <div
                key={c.id}
                role="button"
                tabIndex={0}
                onClick={() => onSelect(c.id)}
                onKeyDown={(e) => e.key === "Enter" && onSelect(c.id)}
                data-testid="conversation-item"
                className={`group flex cursor-pointer items-center gap-1 rounded-lg px-2 py-1.5 transition hover:bg-slate-100 ${
                  c.id === activeId ? "bg-slate-100" : ""
                }`}
              >
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[11px] text-slate-700">{c.title}</p>
                  <p className="text-[10px] text-slate-400">{c.message_count} 条消息</p>
                </div>
                <button
                  type="button"
                  onClick={(e) => void handleDelete(c.id, e)}
                  aria-label="删除对话"
                  className="cursor-pointer text-slate-300 opacity-0 transition group-hover:opacity-100 hover:text-red-500"
                >
                  <Trash2 className="size-3.5" />
                </button>
              </div>
            ))}
      </div>

      <div className="mt-2 flex items-center justify-between border-t border-slate-200 pt-2">
        <span className="truncate text-[11px] text-slate-500" title={user.email}>
          {user.display_name || user.username}
        </span>
        <button
          type="button"
          onClick={onLogout}
          className="shrink-0 cursor-pointer text-[11px] text-slate-400 hover:text-slate-700"
        >
          退出
        </button>
      </div>
    </aside>
  );
}
