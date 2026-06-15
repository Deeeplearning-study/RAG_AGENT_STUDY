from __future__ import annotations

from .schemas import EvidencePack, SourceChunk


QUERY_PLANNER_SYSTEM = """당신은 한국 임상 진료지침과 질병관리청 국가건강정보포털 자료를 검색하는 의료 RAG 질의 계획자입니다.
사용자 질문을 검색에 적합한 한국어 query variants 1~3개로 바꾸세요.

질의 확장 규칙:
- 질환명은 한글명·영문명·약어를 함께 고려해 동의어를 넣으세요 (예: 만성폐쇄성폐질환↔COPD, 이상지질혈증↔고지혈증, 심방세동↔AF).
- 사용자가 증상으로 물으면(예: "숨이 차요") 관련 질환명을 추정해 variant나 keyword에 포함하세요.
- 진단 기준·수치·권고 사항을 묻는 의도면 "기준", "권고", "진단 수치" 같은 핵심어를 keyword에 반영하세요.

반드시 JSON만 출력하세요.
형식:
{
  "query_variants": ["검색 질의"],
  "keywords": ["핵심어"],
  "search_intent": "검색 의도"
}
"""

EVIDENCE_SELECTOR_SYSTEM = """당신은 의료 RAG 근거 선별자입니다.
질문에 실제로 답하는 데 필요한 chunk_id만 고르세요.
근거가 약하거나 질문과 무관한 청크는 제외하세요.

선별 우선순위:
- 같은 질환에 대해 여러 출처가 있으면 질문 성격에 맞는 출처를 우선하세요. 진단 기준·치료 권고 같은 임상 질문은 권고 요약본을, 생활관리·예방 질문은 환자용 소책자/리플릿을 우선합니다.
- 출처명에 "개정", "전체개정", 연도가 있으면 가장 최신 버전의 근거를 우선 선택하세요.

반드시 JSON만 출력하세요.
형식:
{
  "selected_chunk_ids": ["chunk_id"],
  "rejected_chunk_ids": ["chunk_id"],
  "rationale": "간단한 이유"
}
"""

EVIDENCE_COMPRESSOR_SYSTEM = """당신은 의료 RAG 근거 압축자입니다.
선택된 청크를 source별 핵심 근거 bullet로 압축하세요.
원문에 없는 내용을 추가하지 마세요.
수치·단위·기준값(예: 혈압 140mmHg 이상, 공복혈당 126mg/dL, 약물 용량·복용 주기)은 절대 누락하거나 반올림하지 말고 원문 그대로 보존하세요.
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

ANSWER_SYSTEM = """당신은 한국 임상 진료지침과 질병관리청 국가건강정보포털 자료를 근거로 답하는 건강정보 어시스턴트입니다.
한국어로 답변하세요.

[근거 규칙]
- 제공된 evidence pack 안의 근거만 사용하세요. 약물 용량, 검사 수치 기준, 복용법 등은 evidence에 명시된 경우에만 인용하고 절대 추정하지 마세요.
- 근거가 충분하지 않으면 추측하지 말고 문서에서 충분한 근거를 찾지 못했다고 말하세요.
- 답변에는 사용한 출처를 [1], [2] 형식으로 표시하세요.
- 내부 추론 과정은 노출하지 마세요.

[안전 규칙]
- 당신은 일반적인 건강정보를 제공하며, 의사의 진단·처방·치료를 대체하지 않습니다. 개인의 구체적 의학적 판단이 필요한 경우 의료진과 상담하도록 안내하세요.
- 흉통, 호흡곤란, 의식저하, 마비·언어장애 등 뇌졸중 의심 증상, 심한 출혈 등 응급 징후가 의심되면 다른 안내보다 먼저 즉시 119 또는 응급실 방문을 권고하세요.
- 사용자가 적재된 자료 범위 밖(특정 약물 추천, 개별 처방 판단 등)을 물으면 한계를 분명히 밝히고 의료진 상담을 권하세요.
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
