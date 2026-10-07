"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * 回答渲染（需求 12）。
 *
 * ⚠️ **不要添加 `rehype-raw`。**
 * 渲染的内容来自大模型输出，属于不可信输入。加 `rehype-raw` 会让模型输出的
 * 任意 HTML 被真实渲染（XSS 风险）。不加的话原始 HTML 只会作为文本显示。
 *
 * 表格样式是这里的关键：语料里的图片转录、采购规则对比都是表格，
 * 宽表必须能横向滚动，否则会把聊天栏撑破。
 */
export default function Markdown({ content }: { content: string }) {
  return (
    <div className="text-sm leading-relaxed">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ ...props }) => <p className="my-2 first:mt-0 last:mb-0" {...props} />,
          ul: ({ ...props }) => <ul className="my-2 list-disc space-y-1 pl-5" {...props} />,
          ol: ({ ...props }) => <ol className="my-2 list-decimal space-y-1 pl-5" {...props} />,
          li: ({ ...props }) => <li className="leading-relaxed" {...props} />,
          h1: ({ ...props }) => <h1 className="mt-3 mb-2 text-base font-semibold" {...props} />,
          h2: ({ ...props }) => <h2 className="mt-3 mb-2 text-base font-semibold" {...props} />,
          h3: ({ ...props }) => <h3 className="mt-3 mb-1.5 text-sm font-semibold" {...props} />,
          strong: ({ ...props }) => <strong className="font-semibold" {...props} />,
          blockquote: ({ ...props }) => (
            <blockquote
              className="my-2 border-l-4 border-slate-300 bg-slate-50 py-1 pl-3 text-slate-600"
              {...props}
            />
          ),
          code: ({ className, children, ...props }) => {
            const inline = !className;
            if (inline) {
              return (
                <code
                  className="rounded bg-slate-100 px-1 py-0.5 font-mono text-[0.85em]"
                  {...props}
                >
                  {children}
                </code>
              );
            }
            return (
              <code className={`${className ?? ""} font-mono text-xs`} {...props}>
                {children}
              </code>
            );
          },
          pre: ({ ...props }) => (
            <pre
              className="my-2 overflow-x-auto rounded-lg bg-slate-900 p-3 text-slate-100"
              {...props}
            />
          ),
          a: ({ ...props }) => (
            <a
              className="text-blue-600 underline underline-offset-2 hover:text-blue-700"
              target="_blank"
              rel="noreferrer"
              {...props}
            />
          ),
          // 宽表横向滚动，不然会把聊天栏撑破
          table: ({ ...props }) => (
            <div className="my-3 overflow-x-auto rounded-lg border border-slate-200">
              <table className="w-full border-collapse text-xs" {...props} />
            </div>
          ),
          thead: ({ ...props }) => <thead className="bg-slate-100" {...props} />,
          th: ({ ...props }) => (
            <th
              className="border-b border-slate-200 px-3 py-2 text-left font-semibold whitespace-nowrap"
              {...props}
            />
          ),
          td: ({ ...props }) => (
            <td className="border-b border-slate-100 px-3 py-2 align-top" {...props} />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
