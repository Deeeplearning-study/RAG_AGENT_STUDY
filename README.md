# PDF Multi-Agent RAG Agent

Local RAG application for answering questions from the PDFs in `pdf/`.

The planned stack is:

- Frontend: React, Vite, TailwindCSS, TypeScript
- Backend: FastAPI, uv, ChromaDB, PyMuPDF, CrewAI
- Local AI runtime: Ollama
- Main answer model: `gemma4:26b`
- Sub-agent model: `gemma4:e4b`
- Embedding model: `bge-m3`
- Reranker model: `BAAI/bge-reranker-v2-m3`

The API contract and runtime flow are defined in `Plan.md`. This repository currently includes the project plan, persistent agent briefs, the `pdf/` source documents, a frontend Vite scaffold, partial backend scaffolding, and environment examples. Backend implementation files should be completed by the implementation agents following the structure in `Plan.md`.

## Prerequisites

Install these before running the app:

- Python 3.11 or newer
- `uv`
- Node.js 20 or newer
- npm
- Ollama

Check local tools:

```bash
python --version
uv --version
node --version
npm --version
ollama --version
```

## Ollama Setup

Start Ollama:

```bash
ollama serve
```

In another terminal, pull the required models:

```bash
ollama pull gemma4:26b
ollama pull gemma4:e4b
ollama pull bge-m3
```

If your local Ollama registry uses different tags, update `backend/.env` after copying `backend/.env.example`. The environment variable names are `MAIN_AGENT_MODEL`, `SUB_AGENT_MODEL`, and `EMBEDDING_MODEL`.

You can verify installed models with:

```bash
ollama list
```

## Environment Files

Create local env files from the examples:

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

Backend variables:

```env
PDF_DIR=../pdf
CHROMA_DIR=./data/chroma
PROCESSED_DIR=./data/processed
OLLAMA_BASE_URL=http://localhost:11434
MAIN_AGENT_MODEL=gemma4:26b
SUB_AGENT_MODEL=gemma4:e4b
EMBEDDING_MODEL=bge-m3
DEFAULT_TOP_K=8
ENABLE_AGENT_TRACE=true
RERANKER_ENABLED=true
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=auto
RERANKER_BATCH_SIZE=8
RERANKER_MAX_LENGTH=512
RERANKER_CANDIDATE_MULTIPLIER=3
RERANKER_MAX_CANDIDATES=40
```

Frontend variables:

```env
VITE_API_BASE_URL=http://localhost:8888
```

## Backend Setup

Expected backend location:

```bash
cd backend
```

After `backend/pyproject.toml` is created by the backend implementation, install dependencies with:

```bash
uv sync
```

Expected local run command for the FastAPI app:

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8888
```

The backend should expose these endpoints from `Plan.md`:

- `GET /api/health`: checks Ollama, Chroma, configured models, and indexed document count
- `POST /api/ingest`: indexes PDFs from `PDF_DIR`; request body: `{"force": false}`
- `GET /api/documents`: lists indexed PDF metadata
- `POST /api/chat`: answers a question using the RAG flow; request body: `{"message": "...", "top_k": 8}`

## Frontend Setup

Expected frontend location:

```bash
cd frontend
```

`frontend/package.json` defines `dev`, `build`, `preview`, and `lint` scripts. Install dependencies and run the Vite dev server:

```bash
npm install
npm run dev
```

The frontend should call `VITE_API_BASE_URL` for all backend requests. The default assumes the backend is running at `http://localhost:8888`, and Vite is configured to serve the frontend on port `6000`.

## Ingestion Flow

The ingestion pipeline uses files from `pdf/` and stores generated local data under `backend/data/`.

Expected flow:

1. Start Ollama and confirm the embedding model is installed.
2. Start the backend.
3. Call `GET /api/health` and confirm Ollama and Chroma are available.
4. Run ingestion:

```bash
curl -X POST http://localhost:8888/api/ingest \
  -H 'Content-Type: application/json' \
  -d '{"force": false}'
```

5. Confirm indexed documents:

```bash
curl http://localhost:8888/api/documents
```

Use `{"force": true}` only when you need to rebuild the index even if PDF hashes have not changed.

## Chat Flow

After ingestion completes, send a chat request:

```bash
curl -X POST http://localhost:8888/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "문서에서 권고하는 치료 기준을 요약해줘", "top_k": 8}'
```

Expected response shape:

```json
{
  "answer": "string",
  "sources": [
    {
      "file_name": "example.pdf",
      "page_start": 12,
      "page_end": 13,
      "snippet": "관련 원문 일부..."
    }
  ],
  "agent_trace": {
    "query_variants": ["string"],
    "selected_chunk_count": 5
  }
}
```

`ENABLE_AGENT_TRACE=true` is intended for local debugging. Set it to `false` when the UI or API should hide agent trace details.

`RERANKER_ENABLED=true` enables a local Cross-Encoder reranker after Chroma retrieval and before evidence selection. The default `BAAI/bge-reranker-v2-m3` model is loaded by `sentence-transformers` on first use. If model loading or inference fails, the backend falls back to the original vector-score order and records a warning in the internal agent trace.

## Troubleshooting

Ollama is not running:

- Start it with `ollama serve`.
- Confirm the backend env uses `OLLAMA_BASE_URL=http://localhost:11434`.
- Retry `GET /api/health`.

Required model is not installed:

- Run `ollama list`.
- Pull missing models with the `ollama pull` commands above.
- If your machine uses different model tags, update `backend/.env`.

Ingestion has not been run:

- `POST /api/chat` should not be expected to answer from PDFs before `POST /api/ingest` succeeds.
- Run `GET /api/documents` to confirm indexed documents exist.

PDF text extraction returns little or no text:

- The MVP uses PyMuPDF text extraction and does not include OCR.
- Scanned PDFs may produce empty or low-quality text.
- OCR with Tesseract or a separate OCR pipeline is a later phase from `Plan.md`.

Chroma contains stale data:

- Run ingestion with `{"force": true}`.
- If the implementation supports manual reset, stop the backend and remove the local Chroma and processed metadata directories configured by `CHROMA_DIR` and `PROCESSED_DIR`.
- Restart the backend and re-run ingestion.

Frontend cannot reach backend:

- Confirm the backend is running on `http://localhost:8888`.
- Confirm `frontend/.env` has `VITE_API_BASE_URL=http://localhost:8888`.
- Restart the Vite dev server after changing frontend env variables.
- If the browser reports CORS errors, check that the FastAPI CORS settings allow the Vite dev server origin.

Backend dependency files are missing:

- This repository may still be in the planning/scaffold phase.
- Add backend files following the `backend/` structure in `Plan.md`, then run `uv sync`.

Frontend source files are missing:

- The Vite project manifest exists, but `frontend/src/` may still need implementation files.
- Add frontend source files following the `frontend/` structure in `Plan.md`, then run `npm install`.
