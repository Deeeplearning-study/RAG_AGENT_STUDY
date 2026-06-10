import { apiFetch } from './client';
import type {
  DocumentSummary,
  HealthResponse,
  IngestRequest,
  IngestResponse,
} from '../types/api';

export function getHealth() {
  return apiFetch<HealthResponse>('/api/health');
}

export async function getDocuments() {
  const payload = await apiFetch<unknown>('/api/documents');
  return normalizeDocuments(payload);
}

export function ingestDocuments(request: IngestRequest) {
  return apiFetch<IngestResponse>('/api/ingest', {
    method: 'POST',
    body: request,
  });
}

function normalizeDocuments(payload: unknown): DocumentSummary[] {
  const items =
    Array.isArray(payload)
      ? payload
      : payload &&
          typeof payload === 'object' &&
          'documents' in payload &&
          Array.isArray(payload.documents)
        ? payload.documents
        : [];

  return items
    .filter((item): item is Record<string, unknown> => {
      return Boolean(item && typeof item === 'object');
    })
    .map((item) => ({
      document_id: String(item.document_id ?? ''),
      file_name: String(item.file_name ?? ''),
      pages: Number(item.pages ?? item.page_count ?? 0),
      chunks: Number(item.chunks ?? item.chunk_count ?? 0),
      indexed_at: String(item.indexed_at ?? ''),
    }));
}
