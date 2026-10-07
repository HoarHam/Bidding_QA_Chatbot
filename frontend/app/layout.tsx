import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "招投标智能问答机器人",
  description: "基于 RAG 的招投标与政府采购问答助手",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body className="bg-slate-50 text-slate-900 antialiased">{children}</body>
    </html>
  );
}
