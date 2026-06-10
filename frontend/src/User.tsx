import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, Bot, Loader2, MessageSquare, Plus } from "lucide-react";
import { sendChatMessage } from "./api/chat";
import { ChatInput } from "./components/ChatInput";
import { ChatMessage } from "./components/ChatMessage";
import type { ChatMessage as ChatMessageType } from "./types/api";

const DEFAULT_TOP_K = 8;

type ChatSession = {
  id: string;
  title: string;
  messages: ChatMessageType[];
  createdAt: Date;
};

const SUGGESTED_QUESTIONS = [
  "당뇨병의 초기 증상은 무엇인가요?",
  "고혈압을 낮추는 생활 습관은?",
  "감기와 독감의 차이점은?",
  "만성 피로 증후군이란 무엇인가요?",
];

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
  return "요청 중 알 수 없는 오류가 발생했습니다.";
}

function LoadingMessage() {
  return (
    <article className="flex gap-3">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-teal-600 text-white">
        <Bot className="h-5 w-5" aria-hidden="true" />
      </div>
      <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600 shadow-sm">
        <span className="inline-flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          답변을 생성하는 중입니다.
        </span>
      </div>
    </article>
  );
}

export default function User() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [chatError, setChatError] = useState<string | null>(null);
  const [loadingSessionId, setLoadingSessionId] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const isAnyLoading = loadingSessionId !== null;
  const isCurrentSessionLoading = loadingSessionId === activeSessionId;

  const showTrace = useMemo(
    () => import.meta.env.VITE_ENABLE_AGENT_TRACE === "true",
    []
  );

  const activeSession = sessions.find((s) => s.id === activeSessionId) ?? null;
  const messages = useMemo(
    () => activeSession?.messages ?? [],
    [activeSession]
  );

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, isCurrentSessionLoading]);

  const startNewChat = useCallback(() => {
    setActiveSessionId(null);
    setChatError(null);
  }, []);

  const handleSubmit = async (message: string) => {
    const userMessage: ChatMessageType = {
      id: createId(),
      role: "user",
      content: message,
      createdAt: new Date(),
    };

    let sessionId = activeSessionId;

    if (!sessionId) {
      const newSession: ChatSession = {
        id: createId(),
        title: message.length > 28 ? message.slice(0, 28) + "…" : message,
        messages: [userMessage],
        createdAt: new Date(),
      };
      setSessions((prev) => [newSession, ...prev]);
      sessionId = newSession.id;
      setActiveSessionId(sessionId);
    } else {
      setSessions((prev) =>
        prev.map((s) =>
          s.id === sessionId
            ? { ...s, messages: [...s.messages, userMessage] }
            : s
        )
      );
    }

    setChatError(null);
    const capturedId = sessionId;
    setLoadingSessionId(capturedId);

    try {
      const response = await sendChatMessage({ message, top_k: DEFAULT_TOP_K });

      const assistantMessage: ChatMessageType = {
        id: createId(),
        role: "assistant",
        content: response.answer,
        sources: response.sources,
        agentTrace: response.agent_trace,
        createdAt: new Date(),
      };

      setSessions((prev) =>
        prev.map((s) =>
          s.id === capturedId
            ? { ...s, messages: [...s.messages, assistantMessage] }
            : s
        )
      );
    } catch (error) {
      setChatError(resolveError(error));
    } finally {
      setLoadingSessionId(null);
    }
  };

  return (
    <div className="flex h-dvh overflow-hidden bg-white text-slate-950">
      {/* Sidebar */}
      <aside className="flex w-60 shrink-0 flex-col border-r border-slate-200 bg-slate-50">
        <div className="flex items-center justify-between px-4 py-4">
          <span className="text-sm font-bold text-teal-700">진단 IN</span>
          <button
            onClick={startNewChat}
            className="flex h-7 w-7 items-center justify-center rounded-md text-slate-500 transition hover:bg-slate-200 hover:text-slate-900"
            title="새 채팅"
          >
            <Plus className="h-4 w-4" />
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto px-2 pb-4">
          {sessions.length === 0 ? (
            <p className="px-2 py-2 text-xs text-slate-400">
              채팅 내역이 없습니다.
            </p>
          ) : (
            <ul className="space-y-0.5">
              {sessions.map((session) => (
                <li key={session.id}>
                  <button
                    onClick={() => {
                      setActiveSessionId(session.id);
                      setChatError(null);
                    }}
                    className={`flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm transition ${
                      session.id === activeSessionId
                        ? "bg-teal-50 font-medium text-teal-900"
                        : "text-slate-600 hover:bg-slate-200 hover:text-slate-900"
                    }`}
                  >
                    <MessageSquare className="h-3.5 w-3.5 shrink-0 text-slate-400" />
                    <span className="truncate">{session.title}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </nav>
      </aside>

      {/* Main */}
      <main className="flex h-full min-h-0 flex-1 flex-col overflow-hidden">
        <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center gap-6 px-4">
              <div className="text-center">
                <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-teal-600 text-white shadow-md">
                  <Bot className="h-8 w-8" />
                </div>
                <h1 className="text-2xl font-bold text-slate-900">
                  무엇이든 물어보세요
                </h1>
                <p className="mt-2 text-sm text-slate-500">
                  건강과 질병에 관한 궁금한 점을 전문 문서 기반으로 답변해
                  드립니다.
                </p>
              </div>
              <div className="grid w-full max-w-lg grid-cols-2 gap-2">
                {SUGGESTED_QUESTIONS.map((q) => (
                  <button
                    key={q}
                    onClick={() => void handleSubmit(q)}
                    disabled={isAnyLoading}
                    className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-left text-sm text-slate-700 shadow-sm transition hover:border-teal-300 hover:bg-teal-50 hover:text-teal-900 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto flex max-w-3xl flex-col gap-5 px-6 py-6">
              {messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  message={message}
                  showTrace={showTrace}
                />
              ))}
              {isCurrentSessionLoading ? <LoadingMessage /> : null}
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
          )}
        </div>

        <ChatInput disabled={isAnyLoading} onSubmit={handleSubmit} />
      </main>
    </div>
  );
}
