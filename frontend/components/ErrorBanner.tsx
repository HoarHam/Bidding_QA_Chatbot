"use client";

import { AlertTriangle, RotateCcw } from "lucide-react";

import type { AttemptModel } from "@/lib/types";

/**
 * 错误提示条。
 *
 * 当 `partial` 为真时，意思是「回答输出到一半断了」。此时**已显示的部分会保留**
 * （由 ChatShell 保证），错误条显示在正文下方并提供重试——
 * 直接清空用户已经读过的内容，比给一个不完整的答案更糟。
 */
export default function ErrorBanner({
  message,
  partial,
  attempts,
  onRetry,
}: {
  message: string;
  partial: boolean;
  attempts: AttemptModel[];
  onRetry?: () => void;
}) {
  return (
    <div className="mt-2 rounded-lg border border-red-200 bg-red-50 p-3 text-xs">
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-red-600" />
        <div className="flex-1">
          <p className="font-medium text-red-800">
            {partial ? "回答中断" : "请求失败"}
          </p>
          <p className="mt-0.5 text-red-700">{message}</p>

          {attempts.length > 0 && (
            <details className="mt-1.5">
              <summary className="cursor-pointer text-[11px] text-red-600 hover:underline">
                查看 {attempts.length} 次尝试详情
              </summary>
              <ul className="mt-1 space-y-0.5 pl-4 text-[11px] text-red-600">
                {attempts.map((a, i) => (
                  <li key={i} className="list-disc">
                    {a.provider} · {a.phase} · {a.error_type}（{a.elapsed_ms}ms）
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="inline-flex shrink-0 items-center gap-1 rounded border border-red-300 bg-white px-2 py-1 text-[11px] text-red-700 hover:bg-red-100"
          >
            <RotateCcw className="size-3" />
            重试
          </button>
        )}
      </div>
    </div>
  );
}
