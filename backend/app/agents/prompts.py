from __future__ import annotations

from .schemas import EvidencePack, SourceChunk


QUERY_PLANNER_SYSTEM = """당신은 PDF 기반 RAG 검색 질의 계획자입니다.
사용자 질문을 검색에 적합한 한국어 query variants 1~3개로 바꾸세요.
반드시 JSON만 출력하세요.
형식:
{
  "query_variants": ["검색 질의"],
  "keywords": ["핵심어"],
  "search_intent": "검색 의도"
}
"""

EVIDENCE_SELECTOR_SYSTEM = """당신은 RAG 근거 선별자입니다.
질문에 실제로 답하는 데 필요한 chunk_id만 고르세요.
근거가 약하거나 질문과 무관한 청크는 제외하세요.
반드시 JSON만 출력하세요.
형식:
{
  "selected_chunk_ids": ["chunk_id"],
  "rejected_chunk_ids": ["chunk_id"],
  "rationale": "간단한 이유"
}
"""

EVIDENCE_COMPRESSOR_SYSTEM = """당신은 RAG 근거 압축자입니다.
선택된 청크를 source별 핵심 근거 bullet로 압축하세요.
원문에 없는 내용을 추가하지 마세요.
반드시 JSON만 출력하세요.
형식:
{
  "items": [
    {
      "chunk_id": "chunk_id",
      "bullet": "핵심 근거",
      "snippet": "원문 일부"
    }
  ]
}
"""

ANSWER_SYSTEM = """당신은 PDF 문서 기반 RAG 답변 에이전트입니다.
한국어로 답변하세요.
제공된 evidence pack 안의 근거만 사용하세요.
근거가 충분하지 않으면 문서에서 충분한 근거를 찾지 못했다고 말하세요.
답변에는 사용한 출처를 [1], [2] 형식으로 표시하세요.
내부 추론 과정은 노출하지 마세요.
"""


def build_query_planner_prompt(question: str) -> str:
    return f"{QUERY_PLANNER_SYSTEM}\n[사용자 질문]\n{question.strip()}\n"


def build_selector_prompt(question: str, chunks: list[SourceChunk]) -> str:
    chunk_text = "\n\n".join(_format_chunk_for_prompt(chunk) for chunk in chunks)
    return f"{EVIDENCE_SELECTOR_SYSTEM}\n[질문]\n{question.strip()}\n\n[검색 청크]\n{chunk_text}\n"


def build_compressor_prompt(chunks: list[SourceChunk]) -> str:
    chunk_text = "\n\n".join(_format_chunk_for_prompt(chunk) for chunk in chunks)
    return f"{EVIDENCE_COMPRESSOR_SYSTEM}\n[선택 청크]\n{chunk_text}\n"


def build_answer_prompt(question: str, evidence_pack: EvidencePack) -> str:
    evidence_lines = []
    for idx, item in enumerate(evidence_pack.items, start=1):
        citation = item.citation
        page_label = _page_label(citation.page_start, citation.page_end)
        evidence_lines.append(
            f"[{idx}] {citation.file_name}{page_label}\n"
            f"- 근거: {item.bullet}\n"
            f"- 원문: {citation.snippet}"
        )
    evidence_text = "\n\n".join(evidence_lines) if evidence_lines else "(사용 가능한 근거 없음)"
    return f"{ANSWER_SYSTEM}\n[질문]\n{question.strip()}\n\n[evidence pack]\n{evidence_text}\n"


def _format_chunk_for_prompt(chunk: SourceChunk) -> str:
    page_label = _page_label(chunk.page_start, chunk.page_end)
    text = chunk.text[:1600]
    return (
        f"chunk_id: {chunk.chunk_id}\n"
        f"source: {chunk.file_name}{page_label}\n"
        f"score: {chunk.score:.4f}\n"
        f"text: {text}"
    )


def _page_label(page_start: int, page_end: int) -> str:
    if page_start == page_end:
        return f" p.{page_start}"
    return f" pp.{page_start}-{page_end}"
