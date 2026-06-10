# PDF 기반 Multi-Agent RAG Agent 구축 계획

## 1. 목표

현재 저장소의 `pdf/` 폴더에 있는 PDF 20개를 지식베이스로 사용하여, 로컬 환경에서 동작하는 RAG Agent를 구축한다.

프론트엔드는 `React + Vite + TailwindCSS + TypeScript`로 만들고, 백엔드는 `uv` 기반 `FastAPI` 프로젝트로 구성한다.

LLM과 임베딩은 Ollama 기반 로컬 모델을 사용한다. Agent orchestration은 CrewAI를 사용하되, PDF 처리와 벡터 검색 같은 deterministic pipeline은 일반 Python 서비스 계층으로 분리한다.

## 2. 전체 아키텍처

```text
frontend/
  React + Vite + TailwindCSS + TypeScript
  - 채팅 UI
  - 답변 표시
  - 출처 카드 표시
  - 문서 인덱싱 상태 표시

backend/
  FastAPI + uv
  - PDF ingestion API
  - Chroma vector search
  - CrewAI multi-agent flow
  - Ollama model client

pdf/
  기존 PDF 20개

backend/data/
  chroma/
    Chroma persistent vector DB
  processed/
    PDF 추출 결과, 파일 해시, 인덱싱 메타데이터
```

## 3. 기술 스택

### Frontend

- React
- Vite
- TypeScript
- TailwindCSS
- Fetch API 또는 TanStack Query
- 기본 UI는 채팅 중심으로 구성

### Backend

- Python
- uv
- FastAPI
- Uvicorn
- Pydantic
- PyMuPDF
- ChromaDB
- CrewAI
- Ollama Python client 또는 HTTP client
- sentence-transformers 기반 Cross-Encoder Reranker

### Local AI Runtime

- Ollama
- Main agent model: `gemma4:26b`
- Sub agent model: `gemma4:e4b`
- Embedding model: `bge-m3`
- Reranker model: `BAAI/bge-reranker-v2-m3`

모델 태그는 Ollama에서 실제 사용 가능한 이름과 일치해야 한다. 프로젝트 설정값은 `.env`로 분리한다.

Reranker는 Ollama가 아니라 로컬 Cross-Encoder 모델로 실행한다. 기본 모델은 한국어/다국어 질의-문서 relevance scoring에 적합한 `BAAI/bge-reranker-v2-m3`로 두고, 환경 변수로 교체 가능하게 한다.

## 4. Agent 구조

CrewAI는 agent 간 역할 분리와 실행 흐름 제어에 사용한다. 단, 검색과 인덱싱 자체는 CrewAI에 넣지 않고 FastAPI 내부 서비스로 유지한다.

### Main Answer Agent

- 모델: `gemma4:26b`
- 역할: 최종 답변 생성
- 입력:
  - 사용자 원 질문
  - 정제된 evidence pack
  - 출처 메타데이터
- 출력:
  - 한국어 답변
  - 인용 출처 목록
- 규칙:
  - 검색된 근거 안에서만 답변한다.
  - 문서에서 근거를 찾지 못하면 추측하지 않는다.
  - 답변에는 최소 1개 이상의 출처를 포함한다.

### Sub Agent 1: Query Planner

- 모델: `gemma4:e4b`
- 역할: 사용자 질문을 검색에 적합한 query로 재작성한다.
- 입력:
  - 사용자 원 질문
- 출력:
  - 검색용 query variants 1~3개
  - 핵심 키워드
  - 검색 의도

예시:

```text
원 질문: 치료 지침 알려줘
검색 query:
- PDF 문서에서 질환별 치료 권고안 검색
- 약물 치료 기준과 추적 관리 기준 검색
- 최신 진료 지침의 권고 등급 검색
```

### Sub Agent 2: Evidence Selector

- 모델: `gemma4:e4b`
- 역할: Chroma 검색 결과 중 실제 답변에 필요한 청크만 선별한다.
- 입력:
  - 검색된 top-k 청크 목록
  - 사용자 질문
- 출력:
  - 사용할 청크 ID 목록
  - 제외할 청크 ID 목록
  - 선택 이유

목적은 메인 agent에 불필요한 문맥을 넣지 않아 답변 품질과 속도를 개선하는 것이다.

### Sub Agent 3: Evidence Compressor

- 모델: `gemma4:e4b`
- 역할: 선택된 청크를 짧은 근거 묶음으로 압축한다.
- 입력:
  - Evidence Selector가 선택한 청크
- 출력:
  - source별 핵심 근거 bullet
  - PDF 파일명
  - 페이지 번호
  - 원문 스니펫

메인 agent는 긴 검색 결과 전체가 아니라 Evidence Compressor가 만든 evidence pack을 보고 최종 답변을 생성한다.

## 5. RAG 처리 흐름

`POST /api/chat` 요청은 다음 순서로 처리한다.

1. FastAPI가 사용자 질문을 수신한다.
2. CrewAI Flow에서 Query Planner가 검색 query variants를 생성한다.
3. Retrieval Service가 query variants별로 Chroma 검색을 넓은 후보 풀로 수행한다.
4. 검색 결과를 병합하고 중복 청크를 제거한다.
5. Cross-Encoder Reranker가 사용자 원 질문과 후보 청크 쌍을 점수화해 재정렬한다.
6. Evidence Selector가 rerank된 청크 중 실제 답변에 필요한 청크만 선별한다.
7. Evidence Compressor가 evidence pack을 생성한다.
8. Main Answer Agent가 evidence pack을 기반으로 최종 답변을 생성한다.
9. FastAPI가 답변과 source metadata를 프론트엔드에 반환한다.

CrewAI는 `Flow` 중심으로 시작한다. MVP에서는 agent delegation을 과하게 열지 않고, 명시적인 순차 실행으로 디버깅 가능성을 우선한다.

```text
User Question
  -> Query Planner Agent
  -> Chroma Retrieval Service
  -> Cross-Encoder Reranker
  -> Evidence Selector Agent
  -> Evidence Compressor Agent
  -> Main Answer Agent
  -> Answer + Sources
```

Reranker는 deterministic service 계층으로 취급한다. CrewAI agent가 직접 rerank를 수행하지 않고, 검색 결과를 Evidence Selector에 넘기기 전에 Retrieval/Rerank Service가 후보 순서를 정리한다. Reranker가 비활성화되어 있거나 모델 로딩에 실패하면 기존 Chroma similarity score 정렬로 fallback하고 `agent_trace.warnings`에 사유를 기록한다.

## 6. PDF 인덱싱 계획

### PDF 추출

- `PyMuPDF`를 사용한다.
- PDF를 페이지 단위로 읽는다.
- 페이지별 텍스트와 메타데이터를 추출한다.
- 파일명은 원본 그대로 보존한다.
- PDF metadata title이 있으면 표시명으로 사용할 수 있다.

### 청킹

- 청크 크기: 약 800~1,000 tokens
- overlap: 약 120~180 tokens
- 페이지 번호를 반드시 유지한다.
- 한 청크가 여러 페이지를 포함할 경우 시작 페이지와 끝 페이지를 저장한다.

### 저장 메타데이터

각 청크는 다음 정보를 가진다.

```json
{
  "document_id": "string",
  "file_name": "string",
  "page_start": 1,
  "page_end": 1,
  "chunk_id": "string",
  "text": "string",
  "file_hash": "string"
}
```

### 재인덱싱

- PDF 파일 해시를 계산한다.
- 기존 해시와 동일하면 재처리하지 않는다.
- 변경된 PDF만 다시 추출하고 인덱싱한다.
- 사용자가 강제 재인덱싱을 요청할 수 있도록 `force=true` 옵션을 둔다.

### OCR

MVP에서는 OCR을 제외한다.

스캔 PDF가 많아 텍스트 추출률이 낮은 경우, 2차 단계에서 `Tesseract` 또는 별도 OCR 파이프라인을 추가한다.

## 7. Backend API

### `GET /api/health`

Ollama, Chroma, 인덱스 상태를 확인한다.

반환 예시:

```json
{
  "status": "ok",
  "ollama": true,
  "chroma": true,
  "main_model": "gemma4:26b",
  "sub_model": "gemma4:e4b",
  "embedding_model": "bge-m3",
  "indexed_documents": 20
}
```

### `POST /api/ingest`

`pdf/` 폴더 전체를 인덱싱한다.

요청 예시:

```json
{
  "force": false
}
```

반환 예시:

```json
{
  "status": "completed",
  "documents_total": 20,
  "documents_indexed": 20,
  "chunks_total": 1240
}
```

### `GET /api/documents`

인덱싱된 PDF 목록과 상태를 반환한다.

반환 예시:

```json
[
  {
    "document_id": "string",
    "file_name": "example.pdf",
    "pages": 42,
    "chunks": 88,
    "indexed_at": "2026-06-08T00:00:00Z"
  }
]
```

### `POST /api/chat`

질문을 받아 multi-agent RAG 답변을 반환한다.

요청 예시:

```json
{
  "message": "문서에서 권고하는 치료 기준을 요약해줘",
  "top_k": 8
}
```

반환 예시:

```json
{
  "answer": "문서에 따르면 ...",
  "sources": [
    {
      "file_name": "example.pdf",
      "page_start": 12,
      "page_end": 13,
      "snippet": "관련 원문 일부..."
    }
  ],
  "agent_trace": {
    "query_variants": ["..."],
    "selected_chunk_count": 5
  }
}
```

`agent_trace`는 개발 환경에서는 표시하고, 운영 모드에서는 숨길 수 있게 설정한다.

## 8. Frontend 화면 계획

### 메인 화면

- 채팅 메시지 목록
- 질문 입력창
- 전송 버튼
- 답변 로딩 상태
- 오류 상태
- 출처 카드 목록

### 문서 상태 영역

- 인덱싱된 문서 수
- 전체 청크 수
- 마지막 인덱싱 시간
- 재인덱싱 버튼

### 출처 카드

각 답변 하단에 다음 정보를 표시한다.

- PDF 파일명
- 페이지 번호
- 관련 스니펫
- 접기/펼치기 상태

MVP에서는 PDF 뷰어를 직접 구현하지 않는다. 필요하면 후속 단계에서 PDF 페이지 열기 기능을 추가한다.

## 9. Backend 프로젝트 구조

```text
backend/
  pyproject.toml
  .env.example
  app/
    main.py
    core/
      config.py
    api/
      routes_health.py
      routes_ingest.py
      routes_chat.py
      routes_documents.py
    services/
      pdf_loader.py
      chunker.py
      embeddings.py
      vector_store.py
      retrieval.py
      ollama_client.py
    agents/
      crew.py
      prompts.py
      schemas.py
    models/
      chat.py
      documents.py
  data/
    chroma/
    processed/
  tests/
```

## 10. Frontend 프로젝트 구조

```text
frontend/
  package.json
  index.html
  src/
    main.tsx
    App.tsx
    api/
      client.ts
      chat.ts
      documents.ts
    components/
      ChatInput.tsx
      ChatMessage.tsx
      SourceCard.tsx
      DocumentStatus.tsx
    types/
      api.ts
    styles/
      index.css
```

## 11. 환경 변수

`backend/.env.example`

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

## 12. 실행 계획

### 1단계: Backend 기반 구축

- `backend/` uv 프로젝트 생성
- FastAPI 앱 구성
- `.env` 기반 설정 로딩
- health API 작성
- Ollama 연결 확인

### 2단계: PDF 인덱싱

- PyMuPDF 기반 PDF 추출
- 청킹 로직 작성
- Chroma 저장 로직 작성
- `POST /api/ingest` 구현
- `GET /api/documents` 구현

### 3단계: Retrieval 구현

- query embedding 생성
- Chroma top-k 검색
- 중복 청크 제거
- source metadata 정규화

### 3.5단계: Cross-Encoder Reranker 통합

- `sentence-transformers` 기반 reranker service 추가
- 기본 모델은 `BAAI/bge-reranker-v2-m3` 사용
- Chroma 검색 후보 수를 `min(top_k * RERANKER_CANDIDATE_MULTIPLIER, RERANKER_MAX_CANDIDATES)`로 확장
- query variants별 결과를 병합하고 중복 제거한 뒤 사용자 원 질문 기준으로 rerank
- `SourceChunk.score`는 rerank 이후 최종 ranking score로 갱신
- 원 vector score와 rerank score는 필요 시 내부 trace/debug 값으로 보존
- reranker disabled/import 실패/model load 실패/inference 실패 시 기존 vector score 정렬로 fallback

### 4단계: CrewAI Agent Flow 구현

- Query Planner agent 작성
- Evidence Selector agent 작성
- Evidence Compressor agent 작성
- Main Answer agent 작성
- FastAPI `POST /api/chat`에 연결

### 5단계: Frontend 구현

- Vite React 프로젝트 생성
- TailwindCSS 설정
- 채팅 UI 구현
- source card 구현
- 문서 상태 영역 구현
- API client 연결

### 6단계: 검증 및 튜닝

- PDF 20개 전체 인덱싱
- 검색 품질 확인
- 답변 품질 확인
- 모델 지연시간 확인
- top-k, chunk size, overlap 튜닝

## 13. 테스트 계획

### Backend 테스트

- PDF 20개 전체 인덱싱 성공 여부
- Chroma에 청크와 메타데이터가 저장되는지 확인
- 동일 파일 재인덱싱 시 중복 저장이 발생하지 않는지 확인
- `GET /api/documents`가 문서 목록을 반환하는지 확인
- Query Planner가 query variants를 생성하는지 확인
- Evidence Selector가 관련 청크만 선택하는지 확인
- Evidence Compressor가 source별 evidence pack을 생성하는지 확인
- Main Answer Agent 답변에 source가 포함되는지 확인
- Cross-Encoder Reranker가 vector score와 다른 순서로 후보를 재정렬할 수 있는지 확인
- Reranker 비활성화 시 기존 vector score 기반 순서를 유지하는지 확인
- Reranker 모델 로딩 또는 inference 실패 시 fallback warning을 남기고 chat flow가 계속되는지 확인
- Reranker candidate 수가 `top_k`, multiplier, max candidates 설정을 지키는지 확인
- Ollama 미실행 시 명확한 오류를 반환하는지 확인
- 모델이 설치되지 않은 경우 health check에서 실패하는지 확인
- 빈 인덱스 상태에서 chat 요청 시 안내 메시지를 반환하는지 확인

### Frontend 테스트

- 빈 인덱스 상태 표시
- 문서 상태 로딩 표시
- 질문 입력 후 로딩 상태 표시
- 답변 표시
- 출처 카드 표시
- 백엔드 오류 메시지 표시
- 재인덱싱 버튼 동작 확인

### 수동 검증 질문

- PDF에 명확히 존재하는 내용 3개 질문
- 서로 다른 PDF를 비교해야 하는 질문 2개
- 문서에 없는 내용 질문 2개
- 출처 페이지와 실제 PDF 내용 일치 여부 확인

## 14. 성능 및 운영 고려사항

- `gemma4:26b`와 `gemma4:e4b`를 함께 사용하므로 메모리와 VRAM 사용량을 확인해야 한다.
- 처음에는 sequential flow로 구현하고, 병렬 agent 실행은 후속 최적화로 둔다.
- 응답 시간이 길 경우 Evidence Selector와 Evidence Compressor를 하나의 sub agent step으로 합칠 수 있다.
- 검색 품질이 낮으면 chunk size, overlap, top-k, query variants 개수를 조정한다.
- Reranker 도입 후 응답 시간이 길면 `RERANKER_MAX_CANDIDATES`, `RERANKER_BATCH_SIZE`, `RERANKER_DEVICE`를 우선 조정한다.
- CPU 환경에서는 Cross-Encoder가 병목이 될 수 있으므로 candidate pool을 작게 유지하고, GPU 환경에서는 batch size를 늘려 처리량을 확인한다.
- Reranker 품질 평가는 기존 vector-only 결과와 top source 적중률, 출처 페이지 일치율, 평균 응답 지연시간을 비교한다.
- 답변 환각이 보이면 Main Answer Agent prompt에 "근거 없음" 응답 규칙을 더 강하게 둔다.

## 15. MVP 범위

포함한다.

- PDF 폴더 기반 인덱싱
- Chroma 로컬 벡터 저장
- Ollama 기반 로컬 모델 호출
- CrewAI 기반 multi-agent flow
- 채팅 UI
- 답변 출처 표시
- 재인덱싱 API 및 버튼

포함하지 않는다.

- PDF 업로드 UI
- 사용자 인증
- 배포 자동화
- OCR
- PDF 원문 뷰어
- 질문 로그 대시보드
- 운영용 모니터링

## 16. Reranker 도입 상세 계획

### 목적

Chroma similarity search는 embedding 기반 의미 유사도에 강하지만, 질문과 청크의 세부 조건이 정확히 일치하는지 판단하는 데 한계가 있다. Cross-Encoder Reranker는 사용자 질문과 후보 청크를 쌍으로 입력받아 relevance score를 다시 계산하므로, Evidence Selector에 더 정확한 후보 순서를 제공하는 것이 목적이다.

### 기본 동작

- 사용자 요청의 `top_k`는 최종 rerank 후 Evidence Selector에 넘길 후보 수로 유지한다.
- Chroma에는 `top_k`보다 큰 후보 풀을 요청한다.
- query variants별 검색 결과를 병합하고 `chunk_id` 기준으로 중복 제거한다.
- Reranker는 사용자 원 질문을 기준으로 각 후보 청크를 점수화한다.
- 최종 후보는 rerank score 내림차순으로 정렬한다.
- `RERANKER_ENABLED=false`이면 rerank 단계를 건너뛴다.

### 기본 설정값

```env
RERANKER_ENABLED=true
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=auto
RERANKER_BATCH_SIZE=8
RERANKER_MAX_LENGTH=512
RERANKER_CANDIDATE_MULTIPLIER=3
RERANKER_MAX_CANDIDATES=40
```

### 실패 처리

- `sentence-transformers` import 실패: reranker unavailable warning 기록 후 vector-only 검색으로 진행
- 모델 다운로드 또는 로딩 실패: warning 기록 후 vector-only 검색으로 진행
- inference timeout 또는 예외: warning 기록 후 vector-only 검색으로 진행
- rerank 결과가 비어 있음: 기존 병합 후보를 vector score 순서로 사용

### 구현 대상 파일

- `backend/pyproject.toml`: `sentence-transformers` 의존성 추가
- `backend/app/core/config.py`: reranker 환경 변수 추가
- `backend/app/services/reranker.py`: Cross-Encoder rerank service 추가
- `backend/app/agents/schemas.py`: 필요 시 `SourceChunk`에 `retrieval_score`, `rerank_score` optional debug field 추가
- `backend/app/agents/crew.py`: retrieval 이후, evidence selection 이전 rerank 단계 연결
- `backend/.env.example`, `README.md`: 실행 환경과 모델 준비 방법 문서화
- `backend/tests/test_rag_flow.py`, `backend/tests/test_api_routes_and_config.py`: reranker flow와 설정 테스트 추가

## 17. 최종 기본값

- Backend: `uv + FastAPI`
- Frontend: `React + Vite + TailwindCSS + TypeScript`
- Orchestration: `CrewAI Flow`
- Vector DB: `Chroma`
- PDF Parser: `PyMuPDF`
- Main Agent: `gemma4:26b`
- Sub Agent: `gemma4:e4b`
- Embedding: `bge-m3`
- Reranker: `BAAI/bge-reranker-v2-m3`
- 초기 top-k: `8`
- Chunk size: `800~1,000 tokens`
- Chunk overlap: `120~180 tokens`

## 18. 구현 작업용 서브 에이전트 구성

이 섹션의 에이전트는 애플리케이션 런타임에서 동작하는 CrewAI agent가 아니라, 실제 개발 작업을 병렬로 진행할 때 사용할 작업용 서브 에이전트 구성이다.

메인 작업자는 전체 설계, 통합, 최종 코드 리뷰, 충돌 해결을 담당한다. 서브 에이전트는 서로 겹치지 않는 파일 영역을 맡아 병렬로 작업한다.

영속적인 에이전트 정의는 `.agents/` 폴더에 둔다. 새 세션을 시작할 때 이 폴더의 정의 파일을 읽어 역할별 서브 에이전트를 구성한다.

```text
.agents/
  README.md
  frontend-agent.md
  backend-api-agent.md
  vector-ingestion-agent.md
  crewai-rag-agent.md
  qa-evaluation-agent.md
  devops-docs-agent.md
```

### Main Coordinator

- 담당: 전체 구현 순서 관리, 아키텍처 결정, PR 수준 최종 통합
- 직접 처리:
  - 프로젝트 구조 생성
  - 공통 타입과 API 계약 확정
  - 서브 에이전트 결과 병합
  - 통합 테스트
  - 최종 실행 검증
- 주요 산출물:
  - 일관된 디렉터리 구조
  - frontend/backend 연결
  - 최종 동작 가능한 MVP

### Frontend Agent

- 담당 영역: `frontend/`
- 기술 스택:
  - React
  - Vite
  - TypeScript
  - TailwindCSS
- 작업 범위:
  - 채팅 UI 구현
  - 질문 입력창과 전송 동작 구현
  - 답변 메시지 렌더링
  - source card 렌더링
  - 문서 인덱싱 상태 UI 구현
  - 재인덱싱 버튼 구현
  - 로딩, 빈 상태, 오류 상태 구현
- 금지:
  - 백엔드 API 스펙 임의 변경
  - RAG 로직 직접 구현
  - PDF 처리 로직 작성
- 완료 기준:
  - `/api/chat`, `/api/documents`, `/api/ingest`, `/api/health`와 연동 가능
  - 답변과 출처가 분리되어 표시됨
  - 모바일과 데스크톱에서 UI가 깨지지 않음

### Backend API Agent

- 담당 영역: `backend/app/main.py`, `backend/app/api/`, `backend/app/models/`, `backend/app/core/`
- 기술 스택:
  - FastAPI
  - Pydantic
  - uv
- 작업 범위:
  - FastAPI 앱 초기화
  - CORS 설정
  - 환경 변수 설정 로딩
  - API route 작성
  - request/response schema 작성
  - 공통 error response 정리
- 금지:
  - PDF parsing 세부 로직 구현
  - Chroma 저장 로직 직접 구현
  - CrewAI prompt 세부 내용 임의 변경
- 완료 기준:
  - `GET /api/health` 동작
  - `POST /api/ingest` route 연결
  - `GET /api/documents` route 연결
  - `POST /api/chat` route 연결
  - OpenAPI 문서에서 schema 확인 가능

### Vector DB / Ingestion Agent

- 담당 영역: `backend/app/services/pdf_loader.py`, `chunker.py`, `embeddings.py`, `vector_store.py`, `retrieval.py`
- 기술 스택:
  - PyMuPDF
  - ChromaDB
  - Ollama embedding
- 작업 범위:
  - PDF 파일 탐색
  - 파일 해시 계산
  - 페이지 단위 텍스트 추출
  - 청킹 로직 구현
  - Chroma collection 생성
  - embedding 생성
  - 청크 upsert
  - top-k retrieval
  - source metadata 정규화
- 금지:
  - FastAPI route 직접 수정
  - CrewAI agent prompt 작성
  - frontend 수정
- 완료 기준:
  - `pdf/` 폴더 PDF 20개 인덱싱 가능
  - 동일 파일 재인덱싱 시 중복 청크 방지
  - 검색 결과에 파일명, 페이지, 청크 ID 포함

### CrewAI / RAG Agent

- 담당 영역: `backend/app/agents/`
- 기술 스택:
  - CrewAI Flow
  - Ollama LLM
- 작업 범위:
  - Query Planner agent 구현
  - Evidence Selector agent 구현
  - Evidence Compressor agent 구현
  - Main Answer agent 구현
  - agent별 prompt 작성
  - multi-agent flow 작성
  - retrieval service와 flow 연결 인터페이스 정의
- 금지:
  - Chroma 내부 저장 구조 변경
  - FastAPI route schema 임의 변경
  - frontend 수정
- 완료 기준:
  - `gemma4:e4b`가 query planning, evidence selection, compression 수행
  - `gemma4:26b`가 최종 답변 생성
  - 근거 부족 시 추측하지 않는 응답 생성
  - 최종 출력에 answer와 sources 포함

### QA / Evaluation Agent

- 담당 영역: `backend/tests/`, `frontend` 테스트 또는 수동 테스트 체크리스트
- 작업 범위:
  - backend unit test 작성
  - ingestion smoke test 작성
  - retrieval smoke test 작성
  - chat API smoke test 작성
  - frontend 주요 상태 수동 검증 항목 작성
  - 문서에 있는 질문과 없는 질문 샘플 작성
- 금지:
  - 기능 구현 파일 대규모 수정
  - API 스펙 변경
- 완료 기준:
  - 핵심 API 테스트 통과
  - 인덱싱, 검색, 답변 생성 경로 검증 가능
  - 실패 케이스가 최소 3개 이상 검증됨

### DevOps / Documentation Agent

- 담당 영역: `README.md`, `.env.example`, 실행 스크립트, 개발 문서
- 작업 범위:
  - backend 실행 방법 작성
  - frontend 실행 방법 작성
  - Ollama 모델 준비 명령 작성
  - 환경 변수 예시 작성
  - 인덱싱 및 질의 테스트 절차 작성
- 금지:
  - 애플리케이션 로직 수정
  - API 스펙 변경
- 완료 기준:
  - 새 개발자가 README만 보고 로컬 실행 가능
  - 필요한 Ollama 모델 목록이 명확함
  - 문제 해결 섹션에 모델 미설치, Ollama 미실행, 빈 인덱스 상황 포함

## 19. 서브 에이전트 병렬 작업 순서

### Phase A: 프로젝트 기반 구성

1. Main Coordinator가 최상위 디렉터리 구조와 API 계약을 확정한다.
2. Backend API Agent가 FastAPI skeleton을 만든다.
3. Frontend Agent가 Vite React skeleton과 기본 화면을 만든다.
4. DevOps / Documentation Agent가 초기 실행 문서를 준비한다.

### Phase B: RAG 핵심 기능 구현

1. Vector DB / Ingestion Agent가 PDF 추출, 청킹, Chroma 저장을 구현한다.
2. CrewAI / RAG Agent가 agent flow와 prompt를 구현한다.
3. Backend API Agent가 ingestion, documents, chat route에 service를 연결한다.

### Phase C: 통합

1. Main Coordinator가 backend service와 CrewAI flow를 연결한다.
2. Frontend Agent가 실제 API 응답 구조에 맞춰 UI를 연결한다.
3. QA / Evaluation Agent가 smoke test와 실패 케이스를 검증한다.

### Phase D: 안정화

1. 검색 품질을 기준으로 chunk size, overlap, top-k를 조정한다.
2. 응답 속도를 기준으로 Evidence Selector와 Evidence Compressor 분리 여부를 조정한다.
3. README와 `.env.example`을 최종 실행 상태에 맞게 갱신한다.

## 20. 서브 에이전트 간 인터페이스 규칙

- API schema는 Backend API Agent가 정의하고 Main Coordinator가 승인한다.
- Frontend Agent는 확정된 API schema만 사용한다.
- Vector DB / Ingestion Agent는 retrieval 결과를 표준 `SourceChunk` 형태로 반환한다.
- Reranker는 `SourceChunk[]`를 입력받아 같은 형태로 재정렬하고, `score`는 Evidence Selector가 보는 최종 ranking score로 유지한다.
- CrewAI / RAG Agent는 retrieval 내부 구현을 알 필요 없이 `SourceChunk[]`만 입력받는다.
- QA / Evaluation Agent는 구현 세부사항보다 외부 동작 기준으로 검증한다.

표준 `SourceChunk` 형태:

```json
{
  "chunk_id": "string",
  "file_name": "string",
  "page_start": 1,
  "page_end": 1,
  "text": "string",
  "score": 0.82
}
```

표준 `ChatResponse` 형태:

```json
{
  "answer": "string",
  "sources": [
    {
      "file_name": "string",
      "page_start": 1,
      "page_end": 1,
      "snippet": "string"
    }
  ],
  "agent_trace": {
    "query_variants": ["string"],
    "selected_chunk_count": 3
  }
}
```
