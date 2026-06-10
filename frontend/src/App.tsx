import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, Bot, Loader2 } from 'lucide-react';
import { sendChatMessage } from './api/chat';
import { getDocuments, getHealth, ingestDocuments } from './api/documents';
import { ChatInput } from './components/ChatInput';
import { ChatMessage } from './components/ChatMessage';
import { DocumentStatus } from './components/DocumentStatus';
import type {
  ChatMessage as ChatMessageType,
  DocumentSummary,
  HealthResponse,
  IngestResponse,
} from './types/api';

const DEFAULT_TOP_K = 8;

function createId() {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }

  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function resolveError(error: unknown) {
  if (error instanceof Error) {
    return error.message;
  }

  return '요청 중 알 수 없는 오류가 발생했습니다.';
}

function LoadingMessage() {
  return (
    <article className="flex gap-3">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-teal-700 text-white">
        <Bot className="h-5 w-5" aria-hidden="true" />
      </div>
      <div className="rounded-md border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600 shadow-sm">
        <span className="inline-flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          답변을 생성하는 중입니다.
        </span>
      </div>
    </article>
  );
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [ingestResult, setIngestResult] = useState<IngestResponse | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [chatError, setChatError] = useState<string | null>(null);
  const [isStatusLoading, setIsStatusLoading] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const showTrace = useMemo(() => {
    return (
      import.meta.env.DEV ||
      import.meta.env.VITE_ENABLE_AGENT_TRACE === 'true'
    );
  }, []);

  const refreshStatus = useCallback(async () => {
    setIsStatusLoading(true);
    setStatusError(null);

    try {
      const [nextHealth, nextDocuments] = await Promise.all([
        getHealth(),
        getDocuments(),
      ]);

      setHealth(nextHealth);
      setDocuments(nextDocuments);
    } catch (error) {
      setStatusError(resolveError(error));
    } finally {
      setIsStatusLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
  }, [refreshStatus]);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [messages, isChatLoading]);

  const handleReindex = async () => {
    setIsIndexing(true);
    setStatusError(null);
    setIngestResult(null);

    try {
      const result = await ingestDocuments({ force: true });
      setIngestResult(result);
      await refreshStatus();
    } catch (error) {
      setStatusError(resolveError(error));
    } finally {
      setIsIndexing(false);
    }
  };

  const handleSubmit = async (message: string) => {
    const userMessage: ChatMessageType = {
      id: createId(),
      role: 'user',
      content: message,
      createdAt: new Date(),
    };

    setMessages((current) => [...current, userMessage]);
    setChatError(null);
    setIsChatLoading(true);

    try {
      const response = await sendChatMessage({
        message,
        top_k: DEFAULT_TOP_K,
      });

      const assistantMessage: ChatMessageType = {
        id: createId(),
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
        agentTrace: response.agent_trace,
        createdAt: new Date(),
      };

      setMessages((current) => [...current, assistantMessage]);
    } catch (error) {
      setChatError(resolveError(error));
    } finally {
      setIsChatLoading(false);
    }
  };

  return (
    <div className="flex h-dvh overflow-hidden bg-slate-100 text-slate-950">
      <main className="grid h-full min-h-0 w-full grid-cols-1 overflow-hidden lg:grid-cols-[minmax(0,1fr)_22rem]">
        <section className="flex h-full min-h-0 flex-col overflow-hidden">
          <header className="border-b border-slate-200 bg-white px-4 py-4 md:px-6">
            <div className="mx-auto flex max-w-5xl flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="text-xs font-semibold text-teal-700">
                  Local PDF RAG Agent
                </p>
                <h1 className="mt-1 text-xl font-semibold tracking-normal text-slate-950 md:text-2xl">
                  문서 기반 질의응답
                </h1>
              </div>
              <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
                <span className="rounded-md bg-white px-2.5 py-1.5 ring-1 ring-slate-200">
                  문서 {health?.indexed_documents ?? documents.length}개
                </span>
                <span className="rounded-md bg-white px-2.5 py-1.5 ring-1 ring-slate-200">
                  Top-K {DEFAULT_TOP_K}
                </span>
              </div>
            </div>
          </header>

          <div
            ref={scrollRef}
            className="min-h-0 flex-1 overflow-y-auto px-4 py-5 md:px-6"
          >
            <div className="mx-auto flex max-w-5xl flex-col gap-4">
              {messages.length === 0 ? (
                <div className="rounded-md border border-dashed border-slate-300 bg-white px-5 py-10 text-center shadow-sm">
                  <p className="text-base font-semibold text-slate-900">
                    아직 대화가 없습니다.
                  </p>
                  <p className="mt-2 text-sm leading-6 text-slate-500">
                    인덱싱된 PDF 근거를 바탕으로 한국어 답변과 출처가 표시됩니다.
                  </p>
                </div>
              ) : (
                messages.map((message) => (
                  <ChatMessage
                    key={message.id}
                    message={message}
                    showTrace={showTrace}
                  />
                ))
              )}

              {isChatLoading ? <LoadingMessage /> : null}

              {chatError ? (
                <div className="flex items-start gap-2 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm leading-6 text-red-800">
                  <AlertCircle
                    className="mt-0.5 h-4 w-4 shrink-0"
                    aria-hidden="true"
                  />
                  <p>{chatError}</p>
                </div>
              ) : null}
            </div>
          </div>

          <ChatInput disabled={isChatLoading} onSubmit={handleSubmit} />
        </section>

        <div className="hidden h-full min-h-0 overflow-hidden lg:block">
          <DocumentStatus
            documents={documents}
            health={health}
            error={statusError}
            ingestResult={ingestResult}
            isLoading={isStatusLoading}
            isIndexing={isIndexing}
            onRefresh={refreshStatus}
            onReindex={handleReindex}
          />
        </div>
      </main>
    </div>
  );
}
