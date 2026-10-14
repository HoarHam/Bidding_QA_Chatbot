"use client";

import { useState } from "react";
import { AlertCircle, Loader2, LogIn, UserPlus } from "lucide-react";

import { AuthError, login, register } from "@/lib/auth";
import type { UserOut } from "@/lib/types";

type Tab = "login" | "register";

/**
 * 登录 / 注册表单。
 *
 * 注意密码框不能做 trim：前后空格是密码的一部分，静默去掉会导致
 * 「注册时能用、登录时登不上」这种极难排查的问题。
 */
export default function LoginForm({ onSuccess }: { onSuccess: (user: UserOut) => void }) {
  const [tab, setTab] = useState<Tab>("login");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (busy) return;
    setError(null);
    setBusy(true);
    try {
      const data =
        tab === "login"
          ? await login({ username: username.trim(), password })
          : await register({ username: username.trim(), email: email.trim(), password });
      onSuccess(data.user);
    } catch (err) {
      setError(err instanceof AuthError ? err.message : "网络异常，请稍后重试");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="w-full max-w-sm rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="mb-5 flex rounded-lg bg-slate-100 p-0.5">
        {(["login", "register"] as Tab[]).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => {
              setTab(t);
              setError(null);
            }}
            className={`flex-1 cursor-pointer rounded-md px-3 py-1.5 text-sm transition ${
              tab === t ? "bg-white font-medium text-slate-800 shadow-sm" : "text-slate-500"
            }`}
          >
            {t === "login" ? "登录" : "注册"}
          </button>
        ))}
      </div>

      <form onSubmit={submit} className="space-y-3">
        <label className="block">
          <span className="mb-1 block text-xs text-slate-600">
            {tab === "login" ? "用户名或邮箱" : "用户名"}
          </span>
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            autoComplete="username"
            className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-brand-500"
          />
        </label>

        {tab === "register" && (
          <label className="block">
            <span className="mb-1 block text-xs text-slate-600">邮箱</span>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="email"
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-brand-500"
            />
          </label>
        )}

        <label className="block">
          <span className="mb-1 block text-xs text-slate-600">
            密码{tab === "register" && "（至少 8 位）"}
          </span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            minLength={tab === "register" ? 8 : undefined}
            autoComplete={tab === "login" ? "current-password" : "new-password"}
            className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-brand-500"
          />
        </label>

        {error && (
          <p className="flex items-start gap-1.5 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700">
            <AlertCircle className="mt-0.5 size-3.5 shrink-0" />
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={busy}
          className="flex w-full cursor-pointer items-center justify-center gap-1.5 rounded-lg bg-brand-600 px-3 py-2 text-sm text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {busy ? (
            <Loader2 className="size-3.5 animate-spin" />
          ) : tab === "login" ? (
            <LogIn className="size-3.5" />
          ) : (
            <UserPlus className="size-3.5" />
          )}
          {tab === "login" ? "登录" : "注册并登录"}
        </button>
      </form>

      <p className="mt-4 text-center text-[11px] text-slate-400">
        对话历史会保存在你的账号下，可跨会话恢复与搜索
      </p>
    </div>
  );
}
