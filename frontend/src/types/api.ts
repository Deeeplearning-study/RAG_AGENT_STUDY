export type HealthResponse = {
  status: string;
  ollama: boolean;
  chroma: boolean;
  main_model: string;
  sub_model: string;
  embedding_model: string;
  indexed_documents: number;
};

export type DocumentSummary = {
  document_id: string;
  file_name: string;
  pages: number;
  chunks: number;
  indexed_at: string | null;
};

export type IngestRequest = {
  force: boolean;
};

export type IngestResponse = {
  status: string;
  documents_total: number;
  documents_indexed: number;
  chunks_total: number;
};

export type ChatRequest = {
  message: string;
  top_k: number;
};

export type Source = {
  file_name: string;
  page_start: number;
  page_end: number;
  snippet: string;
};

export type AgentTrace = {
  query_variants: string[];
  selected_chunk_count: number;
};

export type ChatResponse = {
  answer: string;
  sources: Source[];
  agent_trace?: AgentTrace;
};

export type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  agentTrace?: AgentTrace;
  createdAt: Date;
};
