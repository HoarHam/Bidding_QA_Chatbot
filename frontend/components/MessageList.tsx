"use client";

import { useEffect, useRef, useState, type WheelEvent } from "react";
import { ArrowDown } from "lucide-react";

import MessageBubble, { type ChatTurn } from "./MessageBubble";

/** 距离底部小于该值视为「仍在底部」，继续自动跟随。 */
const BOTTOM_THRESHOLD = 40;

export default function MessageList({ turns }: { turns: ChatTurn[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  /**
   * 是否跟随底部。用 ref 而不是 state：流式 token 每几十毫秒就更新一次，
   * 必须同步读取最新值，避免 effect 拿到过期的 state 把用户拽回底部。
   */
  const stickToBottomRef = useRef(true);
  const [stuck, setStuck] = useState(false);

  // 只在用户没有向上翻看时跟随底部。
  // 刻意用容器自身的 scrollTop **即时**滚动，而不是 scrollIntoView + smooth：
  // 平滑动画进行中会不断产生「离底部很近」的滚动事件，把用户的向上滚动覆盖掉，
  // 表现为向上滚动后无法停留、页面被强制拉到底部。
  useEffect(() => {
    const el = containerRef.current;
    if (!el || !stickToBottomRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [turns]);

  function onScroll() {
    const el = containerRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    const atBottom = distance <= BOTTOM_THRESHOLD;
    stickToBottomRef.current = atBottom;
    setStuck(!atBottom);
  }

  function onWheel(event: WheelEvent<HTMLDivElement>) {
    // 明确的向上滚轮/触控板手势：立即停止跟随，即使还在阈值内
    if (event.deltaY < 0) {
      stickToBottomRef.current = false;
      setStuck(true);
    }
  }

  function scrollToBottom() {
    const el = containerRef.current;
    if (!el) return;
    stickToBottomRef.current = true;
    setStuck(false);
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }

  return (
    <div className="relative min-h-0 flex-1">
      <div
        ref={containerRef}
        onScroll={onScroll}
        onWheel={onWheel}
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
        <div />
      </div>

      {stuck && (
        <button
          type="button"
          onClick={scrollToBottom}
          className="absolute right-4 bottom-3 inline-flex cursor-pointer items-center gap-1 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-[11px] text-slate-600 shadow-md hover:bg-slate-50"
        >
          <ArrowDown className="size-3" />
          回到底部
        </button>
      )}
    </div>
  );
}
