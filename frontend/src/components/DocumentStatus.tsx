import {
  AlertCircle,
  CheckCircle2,
  Database,
  FileText,
  Loader2,
  RefreshCw,
  Server,
} from 'lucide-react';
import { apiBaseUrl } from '../api/client';
import type {
  DocumentSummary,
  HealthResponse,
  IngestResponse,
} from '../types/api';

type DocumentStatusProps = {
  documents: DocumentSummary[];
  health: HealthResponse | null;
  error: string | null;
  ingestResult: IngestResponse | null;
  isLoading: boolean;
  isIndexing: boolean;
  onRefresh: () => void;
  onReindex: () => void;
};

function formatDate(value?: string | null) {
  if (!value) {
    return '기록 없음';
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat('ko-KR', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date);
}

function latestIndexedAt(documents: DocumentSummary[]) {
  const indexedDates = documents
    .map((document) => document.indexed_at)
    .filter(Boolean)
    .sort();

  return indexedDates.length > 0 ? indexedDates[indexedDates.length - 1] : undefined;
}

function StatusBadge({
  ok,
  label,
}: {
  ok: boolean | null;
  label: string;
}) {
  const Icon = ok ? CheckCircle2 : AlertCircle;

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium ${
        ok
          ? 'bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200'
          : 'bg-amber-50 text-amber-800 ring-1 ring-amber-200'
      }`}
    >
      <Icon className="h-3.5 w-3.5" aria-hidden="true" />
      {label}
    </span>
  );
}

export function DocumentStatus({
  documents,
  health,
  error,
  ingestResult,
  isLoading,
  isIndexing,
  onRefresh,
  onReindex,
}: DocumentStatusProps) {
  const chunksTotal = documents.reduce(
    (total, document) => total + document.chunks,
    0,
  );
  const lastIndexedAt = latestIndexedAt(documents);
  const apiLabel = apiBaseUrl || '현재 호스트';

  return (
    <aside className="flex h-full min-h-0 flex-col border-l border-slate-200 bg-white">
      <div className="border-b border-slate-200 px-5 py-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-xs font-semibold text-slate-500">문서 상태</p>
            <h2 className="mt-1 text-lg font-semibold text-slate-950">
              인덱싱 현황
            </h2>
          </div>
          <button
            type="button"
            onClick={onRefresh}
            disabled={isLoading || isIndexing}
            className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-slate-300 bg-white text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
            aria-label="상태 새로고침"
            title="상태 새로고침"
          >
            <RefreshCw
              className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`}
              aria-hidden="true"
            />
          </button>
        </div>
        <p className="mt-2 break-words text-xs text-slate-500">
          API: {apiLabel}
        </p>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <div className="flex items-center gap-2 text-xs font-medium text-slate-500">
              <FileText className="h-4 w-4" aria-hidden="true" />
              문서
            </div>
            <p className="mt-2 text-2xl font-semibold text-slate-950">
              {health?.indexed_documents ?? documents.length}
            </p>
          </div>
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <div className="flex items-center gap-2 text-xs font-medium text-slate-500">
              <Database className="h-4 w-4" aria-hidden="true" />
              청크
            </div>
            <p className="mt-2 text-2xl font-semibold text-slate-950">
              {chunksTotal.toLocaleString('ko-KR')}
            </p>
          </div>
        </div>

        <div className="mt-4 rounded-md border border-slate-200 bg-white p-3">
          <div className="flex flex-wrap gap-2">
            <StatusBadge ok={health?.ollama ?? null} label="Ollama" />
            <StatusBadge ok={health?.chroma ?? null} label="Chroma" />
            <StatusBadge ok={health?.status === 'ok'} label="API" />
          </div>
          <div className="mt-3 space-y-1 text-xs leading-5 text-slate-600">
            <p>메인 모델: {health?.main_model ?? '확인 중'}</p>
            <p>서브 모델: {health?.sub_model ?? '확인 중'}</p>
            <p>임베딩: {health?.embedding_model ?? '확인 중'}</p>
            <p>마지막 인덱싱: {formatDate(lastIndexedAt)}</p>
          </div>
        </div>

        {error ? (
          <div className="mt-4 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm leading-6 text-red-800">
            {error}
          </div>
        ) : null}

        {ingestResult ? (
          <div className="mt-4 rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm leading-6 text-emerald-800">
            재인덱싱 완료: 문서 {ingestResult.documents_indexed}/
            {ingestResult.documents_total}개, 청크{' '}
            {ingestResult.chunks_total.toLocaleString('ko-KR')}개
          </div>
        ) : null}

        <button
          type="button"
          onClick={onReindex}
          disabled={isIndexing}
          className="mt-4 inline-flex h-10 w-full items-center justify-center gap-2 rounded-md bg-slate-900 px-4 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {isIndexing ? (
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          ) : (
            <RefreshCw className="h-4 w-4" aria-hidden="true" />
          )}
          {isIndexing ? '재인덱싱 중' : '강제 재인덱싱'}
        </button>

        <div className="mt-5">
          <h3 className="text-sm font-semibold text-slate-950">문서 목록</h3>
          <div className="mt-3 space-y-2">
            {documents.length === 0 ? (
              <div className="rounded-md border border-dashed border-slate-300 px-3 py-6 text-center text-sm text-slate-500">
                인덱싱된 문서가 없습니다.
              </div>
            ) : (
              documents.map((document) => (
                <article
                  key={document.document_id}
                  className="rounded-md border border-slate-200 bg-slate-50 px-3 py-3"
                >
                  <div className="flex items-start gap-2">
                    <Server
                      className="mt-0.5 h-4 w-4 shrink-0 text-slate-500"
                      aria-hidden="true"
                    />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-slate-900">
                        {document.file_name}
                      </p>
                      <p className="mt-1 text-xs text-slate-500">
                        {document.pages}쪽 · {document.chunks}청크
                      </p>
                    </div>
                  </div>
                </article>
              ))
            )}
          </div>
        </div>
      </div>
    </aside>
  );
}
