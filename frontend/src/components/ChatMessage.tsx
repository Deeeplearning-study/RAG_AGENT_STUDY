import { Bot, UserRound } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { SourceCard } from './SourceCard';
import type { ChatMessage as ChatMessageType } from '../types/api';

type ChatMessageProps = {
  message: ChatMessageType;
  showTrace: boolean;
};

export function ChatMessage({ message, showTrace }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <article
      className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      {!isUser ? (
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-teal-700 text-white">
          <Bot className="h-5 w-5" aria-hidden="true" />
        </div>
      ) : null}

      <div
        className={`max-w-[min(100%,44rem)] rounded-md px-4 py-3 shadow-sm ${
          isUser
            ? 'bg-slate-900 text-white'
            : 'border border-slate-200 bg-white text-slate-900'
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap break-words text-sm leading-7">
            {message.content}
          </p>
        ) : (
          <div className="space-y-3 text-sm leading-7">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
            components={{
              a: ({ ...props }) => (
                <a
                  className="font-medium text-teal-700 underline underline-offset-2"
                  target="_blank"
                  rel="noreferrer"
                  {...props}
                />
              ),
              blockquote: ({ ...props }) => (
                <blockquote
                  className="border-l-4 border-slate-200 pl-3 text-slate-700"
                  {...props}
                />
              ),
              code: ({ children, ...props }) => (
                <code
                  className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[0.85em] text-slate-900"
                  {...props}
                >
                  {children}
                </code>
              ),
              h1: ({ ...props }) => (
                <h1 className="text-lg font-semibold leading-7" {...props} />
              ),
              h2: ({ ...props }) => (
                <h2 className="text-base font-semibold leading-7" {...props} />
              ),
              h3: ({ ...props }) => (
                <h3 className="text-sm font-semibold leading-7" {...props} />
              ),
              li: ({ ...props }) => <li className="pl-1" {...props} />,
              ol: ({ ...props }) => (
                <ol className="list-decimal space-y-1 pl-5" {...props} />
              ),
              p: ({ ...props }) => (
                <p className="break-words text-sm leading-7" {...props} />
              ),
              pre: ({ ...props }) => (
                <pre
                  className="overflow-x-auto rounded-md bg-slate-950 p-3 text-sm leading-6 text-slate-50"
                  {...props}
                />
              ),
              table: ({ ...props }) => (
                <div className="overflow-x-auto">
                  <table
                    className="min-w-full border-collapse text-sm"
                    {...props}
                  />
                </div>
              ),
              td: ({ ...props }) => (
                <td
                  className="border border-slate-200 px-2 py-1 align-top"
                  {...props}
                />
              ),
              th: ({ ...props }) => (
                <th
                  className="border border-slate-200 bg-slate-50 px-2 py-1 text-left font-semibold"
                  {...props}
                />
              ),
              ul: ({ ...props }) => (
                <ul className="list-disc space-y-1 pl-5" {...props} />
              ),
            }}
            >
            {message.content}
            </ReactMarkdown>
          </div>
        )}

        {!isUser && message.sources && message.sources.length > 0 ? (
          <div className="mt-4 space-y-2">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              참조한 출처
            </h3>
            <div className="space-y-2">
              {message.sources.map((source, index) => (
                <SourceCard
                  key={`${source.file_name}-${source.page_start}-${index}`}
                  source={source}
                  index={index}
                />
              ))}
            </div>
          </div>
        ) : null}

        {!isUser && showTrace && message.agentTrace ? (
          <div className="mt-4 rounded-md border border-cyan-100 bg-cyan-50 px-3 py-2 text-xs leading-5 text-cyan-950">
            <p className="font-semibold">Agent trace</p>
            <p>선택 청크: {message.agentTrace.selected_chunk_count}개</p>
            {message.agentTrace.query_variants.length > 0 ? (
              <p className="break-words">
                검색 질의: {message.agentTrace.query_variants.join(' / ')}
              </p>
            ) : null}
          </div>
        ) : null}
      </div>

      {isUser ? (
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-slate-200 text-slate-700">
          <UserRound className="h-5 w-5" aria-hidden="true" />
        </div>
      ) : null}
    </article>
  );
}
