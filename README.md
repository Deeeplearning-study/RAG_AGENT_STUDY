# PDF Multi-Agent RAG Agent

`pdf/` 디렉터리에 있는 PDF 문서를 기반으로 질문에 답하는 로컬 RAG 애플리케이션입니다.

사용 기술 스택은 다음과 같습니다.

- Frontend: React, Vite, TailwindCSS, TypeScript
- Backend: FastAPI, uv, ChromaDB, PyMuPDF, CrewAI
- Local AI runtime: Ollama
- Main answer model: `gemma4:26b`
- Sub-agent model: `gemma4:e4b`
- Embedding model: `bge-m3`
- Reranker model: `BAAI/bge-reranker-v2-m3`

API 계약과 런타임 흐름은 `Plan.md`에 정의되어 있습니다. 이 저장소에는 프로젝트 계획, 개발용 persistent agent brief, `pdf/` 원본 문서, Vite 기반 frontend, FastAPI backend, 환경 변수 예시 파일이 포함되어 있습니다.

## 사전 준비

앱을 실행하기 전에 아래 도구를 설치하세요.

- Python 3.11 이상
- `uv`
- Node.js 20 이상
- npm
- Ollama

로컬 도구가 설치되어 있는지 확인합니다.

```bash
python --version
uv --version
node --version
npm --version
ollama --version
```

## Ollama 설정

Ollama를 실행합니다.

```bash
ollama serve
```

다른 터미널에서 필요한 모델을 내려받습니다.

```bash
ollama pull gemma4:26b
ollama pull gemma4:e4b
ollama pull bge-m3
```

로컬 Ollama registry에서 다른 태그를 사용한다면 `backend/.env.example`을 복사해 `backend/.env`를 만든 뒤 모델명을 수정하세요. 관련 환경 변수는 `MAIN_AGENT_MODEL`, `SUB_AGENT_MODEL`, `EMBEDDING_MODEL`입니다.

설치된 모델은 다음 명령으로 확인할 수 있습니다.

```bash
ollama list
```

## 환경 변수 파일

예시 파일을 복사해 로컬 환경 변수 파일을 만듭니다.

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

Backend 환경 변수:

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

Frontend 환경 변수:

```env
VITE_API_BASE_URL=http://localhost:8888
```

## Backend 실행

Backend 디렉터리로 이동합니다.

```bash
cd backend
```

의존성을 설치합니다.

```bash
uv sync
```

FastAPI 앱을 실행합니다.

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8888
```

Backend는 다음 API를 제공합니다.

- `GET /api/health`: Ollama, Chroma, 설정된 모델, 인덱싱된 문서 수를 확인합니다.
- `POST /api/ingest`: `PDF_DIR`의 PDF를 인덱싱합니다. 요청 본문: `{ "force": false }`
- `GET /api/documents`: 인덱싱된 PDF 메타데이터를 조회합니다.
- `POST /api/chat`: RAG 흐름으로 질문에 답합니다. 요청 본문: `{ "message": "...", "top_k": 8 }`

## Frontend 실행

Frontend 디렉터리로 이동합니다.

```bash
cd frontend
```

`frontend/package.json`에는 `dev`, `build`, `preview`, `lint` 스크립트가 정의되어 있습니다. 의존성을 설치하고 Vite 개발 서버를 실행합니다.

```bash
npm install
npm run dev
```

Frontend는 모든 backend 요청에 `VITE_API_BASE_URL`을 사용합니다. 기본 설정은 backend가 `http://localhost:8888`에서 실행되는 것을 가정합니다. Vite 개발 서버는 포트 `6000`에서 실행되도록 설정되어 있습니다.

파일 watcher 한도 문제로 Vite가 `ENOSPC: System limit for number of file watchers reached` 에러를 내면 polling 모드로 실행할 수 있습니다.

```bash
CHOKIDAR_USEPOLLING=true npm run dev -- --host 0.0.0.0
```

## 인덱싱 흐름

인덱싱 파이프라인은 `pdf/`의 파일을 읽고 생성된 로컬 데이터를 `backend/data/` 아래에 저장합니다.

기본 실행 순서는 다음과 같습니다.

1. Ollama를 실행하고 embedding 모델이 설치되어 있는지 확인합니다.
2. Backend를 실행합니다.
3. `GET /api/health`를 호출해 Ollama와 Chroma가 사용 가능한지 확인합니다.
4. 인덱싱을 실행합니다.

```bash
curl -X POST http://localhost:8888/api/ingest \
  -H 'Content-Type: application/json' \
  -d '{"force": false}'
```

5. 인덱싱된 문서를 확인합니다.

```bash
curl http://localhost:8888/api/documents
```

PDF hash가 바뀌지 않았더라도 인덱스를 다시 만들고 싶을 때만 `{ "force": true }`를 사용하세요.

## 채팅 흐름

인덱싱이 끝나면 채팅 요청을 보낼 수 있습니다.

```bash
curl -X POST http://localhost:8888/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "문서에서 권고하는 치료 기준을 요약해줘", "top_k": 8}'
```

응답 형태는 다음과 같습니다.

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

`ENABLE_AGENT_TRACE=true`는 로컬 디버깅용입니다. UI 또는 API 응답에서 agent trace 세부 정보를 숨기려면 `false`로 설정하세요.

`RERANKER_ENABLED=true`는 Chroma 검색 이후, evidence selection 이전에 로컬 Cross-Encoder reranker를 활성화합니다. 기본 모델 `BAAI/bge-reranker-v2-m3`는 첫 사용 시 `sentence-transformers`가 로드합니다. 모델 로딩이나 추론에 실패하면 backend는 기존 vector score 순서로 fallback하고 내부 agent trace에 warning을 기록합니다.

## 런타임 RAG 흐름

채팅 요청은 다음 순서로 처리됩니다.

1. Query Planner가 사용자 질문을 검색용 query variant로 변환합니다.
2. Retriever가 query를 `bge-m3` embedding으로 바꾼 뒤 Chroma에서 관련 chunk를 검색합니다.
3. Reranker가 검색 후보를 질문과의 관련도 기준으로 재정렬합니다.
4. Evidence Selector가 최종 답변에 사용할 chunk만 선택합니다.
5. Evidence Compressor가 선택된 chunk를 핵심 근거와 snippet으로 압축합니다.
6. Main Answer Agent가 evidence pack만 사용해 한국어 답변과 출처 표시를 생성합니다.

기본 모델 역할은 다음과 같습니다.

- `gemma4:e4b`: Query Planner, Evidence Selector, Evidence Compressor
- `gemma4:26b`: Main Answer Agent
- `bge-m3`: embedding 생성
- `BAAI/bge-reranker-v2-m3`: Cross-Encoder reranking

## 문제 해결

Ollama가 실행 중이 아닐 때:

- `ollama serve`로 실행합니다.
- Backend 환경 변수에 `OLLAMA_BASE_URL=http://localhost:11434`가 설정되어 있는지 확인합니다.
- `GET /api/health`를 다시 호출합니다.

필수 모델이 설치되어 있지 않을 때:

- `ollama list`를 실행합니다.
- 누락된 모델을 위의 `ollama pull` 명령으로 내려받습니다.
- 머신에서 다른 모델 태그를 쓴다면 `backend/.env`를 수정합니다.

인덱싱을 아직 실행하지 않았을 때:

- `POST /api/ingest`가 성공하기 전에는 `POST /api/chat`이 PDF 기반 답변을 할 수 없습니다.
- `GET /api/documents`로 인덱싱된 문서가 있는지 확인합니다.

PDF 텍스트 추출 결과가 거의 없을 때:

- MVP는 PyMuPDF 텍스트 추출을 사용하며 OCR을 포함하지 않습니다.
- 스캔 PDF는 비어 있거나 품질 낮은 텍스트를 만들 수 있습니다.
- Tesseract 또는 별도 OCR pipeline은 `Plan.md` 기준 이후 단계입니다.

Chroma에 오래된 데이터가 남아 있을 때:

- `{ "force": true }`로 인덱싱을 다시 실행합니다.
- 수동 reset이 필요하면 backend를 중지한 뒤 `CHROMA_DIR`, `PROCESSED_DIR`에 설정된 로컬 Chroma 및 처리 메타데이터 디렉터리를 제거합니다.
- Backend를 다시 실행하고 인덱싱을 다시 수행합니다.

Frontend가 backend에 연결하지 못할 때:

- Backend가 `http://localhost:8888`에서 실행 중인지 확인합니다.
- `frontend/.env`에 `VITE_API_BASE_URL=http://localhost:8888`이 있는지 확인합니다.
- Frontend 환경 변수를 바꾼 뒤에는 Vite 개발 서버를 재시작합니다.
- 브라우저에 CORS 오류가 표시되면 FastAPI CORS 설정이 Vite 개발 서버 origin을 허용하는지 확인합니다.

Vite가 file watcher 에러로 실행되지 않을 때:

- `CHOKIDAR_USEPOLLING=true npm run dev -- --host 0.0.0.0`로 실행합니다.
- 가능하다면 시스템의 inotify watcher 한도를 늘립니다.
