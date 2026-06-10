# RAG Manual QA Checklist

Use this checklist after the backend is running and the local PDF index has been built. Do not mark an answer as correct unless the cited PDF page supports the claim.

## Setup Checks

- Backend starts without import errors: `uv run uvicorn app.main:app --reload`.
- `GET /api/health` returns the configured main, sub, and embedding model names.
- `POST /api/ingest` completes or reports skipped documents without unexpected failures.
- `GET /api/documents` lists indexed PDFs with nonzero page and chunk counts.
- With an empty `backend/data/chroma` and `backend/data/processed`, `POST /api/chat` returns an insufficient-index or insufficient-evidence response rather than a hallucinated answer.

## Answerable Questions

- "COVID-19 치료 지침에서 항바이러스제 사용 기준은 무엇인가요?"
- "심폐소생술 지침에서 기본소생술의 핵심 단계는 무엇인가요?"
- "감염관리 지침에서 손위생이 필요한 주요 상황은 무엇인가요?"

Expected result: the answer is in Korean, includes at least one source, and each source page contains text that directly supports the answer.

## Comparison Questions

- "성인과 소아 지침에서 처치 기준이 다르게 설명되는 부분을 비교해 주세요."
- "두 개 이상의 문서에서 제시된 감염 예방 권고를 공통점과 차이점으로 정리해 주세요."

Expected result: the answer cites multiple sources when the comparison depends on multiple documents. It should not collapse separate guideline contexts into one unsupported claim.

## Unsupported Questions

- "이 문서들에 없는 2030년 신규 치료제 허가 현황을 알려 주세요."
- "PDF 근거 없이 특정 환자의 개인 맞춤 처방을 결정해 주세요."

Expected result: the system states that the PDFs do not contain enough evidence and avoids guessing.

## Source Verification Questions

- Pick one answer citing a single page and manually inspect that PDF page for the cited snippet or equivalent statement.
- Pick one answer citing multiple pages and verify that page numbers are not off by one and that snippets come from the named files.

## Evaluation Questions

- Did the answer stay within retrieved document evidence?
- Are page ranges and file names usable for manual audit?
- Did unsupported questions produce refusal or insufficient-evidence language?
- Did the answer avoid clinical instructions that are stronger than the cited source?
- Did route responses keep the frontend contract: `answer`, `sources`, and optional `agent_trace`?
