"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowDown } from "lucide-react";

import MessageBubble, { type ChatTurn } from "./MessageBubble";

export default function MessageList({ turns }: { turns: ChatTurn[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [stuck, setStuck] = useState(false);

  // 用户往上翻看历史时不要强行拉回底部
  useEffect(() => {
    if (stuck) return;
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, stuck]);

  function onScroll() {
    const el = containerRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    setStuck(distance > 120);
  }

  return (
    <div className="relative min-h-0 flex-1">
      <div
        ref={containerRef}
        onScroll={onScroll}
        className="scrollbar-thin h-full space-y-4 overflow-y-auto px-1 py-2"
      >
        {turns.length === 0 && (
          <div className="flex h-full items-center justify-center">
            <p className="text-sm text-slate-400">
              请输入招投标相关问题开始对话
            </p>
          </div>
        )}
        {turns.map((t) => (
          <MessageBubble key={t.id} turn={t} />
        ))}
        <div ref={endRef} />
      </div>

      {stuck && (
        <button
          type="button"
          onClick={() => setStuck(false)}
          className="absolute right-4 bottom-3 inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-[11px] text-slate-600 shadow-md hover:bg-slate-50"
        >
          <ArrowDown className="size-3" />
          回到底部
        </button>
      )}
    </div>
  );
}
