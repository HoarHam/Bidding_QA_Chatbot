"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { FileSearch } from "lucide-react";

import LoginForm from "@/components/LoginForm";
import { getToken } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();

  // 已登录的话没必要停留在这个页面
  useEffect(() => {
    if (getToken()) router.replace("/");
  }, [router]);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-6 px-4">
      <div className="text-center">
        <div className="mb-2 inline-flex size-11 items-center justify-center rounded-xl bg-brand-600 text-white">
          <FileSearch className="size-6" />
        </div>
        <h1 className="text-lg font-semibold">招投标智能问答机器人</h1>
        <p className="mt-1 text-xs text-slate-500">
          登录后即可提问，对话历史会自动保存
        </p>
      </div>

      <LoginForm
        onSuccess={() => {
          // 用 replace 而非 push：登录页不该留在后退历史里
          router.replace("/");
        }}
      />
    </main>
  );
}
