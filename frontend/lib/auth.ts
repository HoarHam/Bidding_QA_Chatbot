/**
 * 登录状态管理。
 *
 * token 存在 localStorage 里：简单直接，但**只能在浏览器端访问**——
 * 服务端渲染时 `window` 不存在，所有读写都要判空，否则页面会在 SSR 阶段崩掉。
 *
 * 安全说明：localStorage 里的 token 可被同源 JS 读取，因此对 XSS 没有防护。
 * 更稳妥的做法是 httpOnly cookie（配合 CSRF 防护）。当前阶段选择前者，
 * 是为了让前后端分离部署更简单；如果将来要对外开放，应重新评估。
 */

import type { LoginRequest, RegisterRequest, TokenResponse, UserOut } from "./types";

const TOKEN_KEY = "bidding_qa_token";
const USER_KEY = "bidding_qa_user";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function getStoredUser(): UserOut | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as UserOut;
  } catch {
    return null; // 存坏了就当未登录
  }
}

function storeSession(data: TokenResponse): void {
  window.localStorage.setItem(TOKEN_KEY, data.access_token);
  window.localStorage.setItem(USER_KEY, JSON.stringify(data.user));
}

export function clearSession(): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(TOKEN_KEY);
  window.localStorage.removeItem(USER_KEY);
}

export function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/** 把后端返回的错误体转成一句能展示给用户的话。 */
async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    // pydantic 校验错误是数组，取第一条的说明
    if (Array.isArray(detail) && detail[0]?.msg) return String(detail[0].msg);
    if (body?.error?.message) return String(body.error.message);
  } catch {
    /* 非 JSON 响应，走下面的兜底 */
  }
  return `请求失败（HTTP ${res.status}）`;
}

export class AuthError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AuthError";
  }
}

async function postAuth(path: string, body: unknown): Promise<TokenResponse> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new AuthError(await errorMessage(res));
  const data = (await res.json()) as TokenResponse;
  storeSession(data);
  return data;
}

export function login(body: LoginRequest): Promise<TokenResponse> {
  return postAuth("/api/auth/login", body);
}

export function register(body: RegisterRequest): Promise<TokenResponse> {
  return postAuth("/api/auth/register", body);
}

/** 校验当前 token 是否仍有效，并返回最新的用户信息。 */
export async function fetchMe(): Promise<UserOut | null> {
  const token = getToken();
  if (!token) return null;
  try {
    const res = await fetch("/api/auth/me", { headers: authHeaders(), cache: "no-store" });
    if (res.status === 401) {
      // token 过期或被撤销：清掉，让调用方跳登录页
      clearSession();
      return null;
    }
    if (!res.ok) return getStoredUser();
    const user = (await res.json()) as UserOut;
    window.localStorage.setItem(USER_KEY, JSON.stringify(user));
    return user;
  } catch {
    // 网络问题时退回本地缓存的用户信息，不把用户直接踢下线
    return getStoredUser();
  }
}

export function logout(): void {
  clearSession();
}
