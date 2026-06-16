"""CrewAI-based agentic RAG orchestrator.

This layer turns the previously hand-written sequential pipeline into a real
CrewAI multi-agent flow while preserving the deterministic safety net required
by the medical domain.

What it adds (on top of the deterministic ``RAGFlowFacade``):
  1. Tools      - retrieval/rerank is exposed as a CrewAI tool the research
                  agent actually decides to call (``search_clinical_docs``).
  2. Re-search  - a code-controlled conditional retry: if the selector finds no
                  usable evidence, the query is broadened and retrieval runs
                  again once (controlled "2B" loop, safe for small local models).
  3. Guardrail  - the answer task is validated (citation markers, emergency
                  triage) and regenerated on failure.
  4. Structured - planner/selector/compressor outputs use ``output_pydantic``.

If CrewAI or the Ollama backend is unavailable (or any stage raises), the whole
run degrades gracefully to the deterministic ``RAGFlowFacade`` pipeline, so the
chat endpoint never hard-fails and never fabricates ungrounded medical claims.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .crew import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    AgentSettings,
    RAGFlowFacade,
    _crewai_available,
    _evidence_item,
    _fallback_evidence_item,
    _snippet,
    _summarize_text,
)
from .prompts import (
    ANSWER_SYSTEM,
    EVIDENCE_COMPRESSOR_SYSTEM,
    EVIDENCE_SELECTOR_SYSTEM,
    QUERY_PLANNER_SYSTEM,
    _format_chunk_for_prompt,
    _page_label,
)
from .schemas import (
    EvidenceItem,
    EvidencePack,
    EvidenceSelection,
    RAGFlowResult,
    SourceChunk,
    SourceCitation,
)


# Emergency triage keywords; if present in the question, the answer guardrail
# requires an explicit 119 / emergency-room instruction (ANSWER_SYSTEM safety rule).
EMERGENCY_KEYWORDS = (
    "흉통",
    "가슴 통증",
    "호흡곤란",
    "숨이 차",
    "숨을 못",
    "의식",
    "마비",
    "언어장애",
    "뇌졸중",
    "심한 출혈",
    "쓰러",
    "경련",
)


def _none_to_str(value: Any) -> Any:
    """Small local models often emit JSON ``null`` for optional fields."""
    return "" if value is None else value


def _clean_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(item).strip() for item in value if item is not None and str(item).strip()]


class EvidenceSelectionOut(BaseModel):
    selected_chunk_ids: list[str] = Field(default_factory=list)
    rejected_chunk_ids: list[str] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("selected_chunk_ids", "rejected_chunk_ids", mode="before")
    @classmethod
    def _coerce_lists(cls, value: Any) -> list[str]:
        return _clean_str_list(value)

    @field_validator("rationale", mode="before")
    @classmethod
    def _coerce_rationale(cls, value: Any) -> Any:
        return _none_to_str(value)


class EvidenceItemOut(BaseModel):
    chunk_id: str = ""
    bullet: str = ""
    snippet: str = ""

    @field_validator("chunk_id", "bullet", "snippet", mode="before")
    @classmethod
    def _coerce_str(cls, value: Any) -> Any:
        return _none_to_str(value)


class EvidencePackOut(BaseModel):
    items: list[EvidenceItemOut] = Field(default_factory=list)

    @field_validator("items", mode="before")
    @classmethod
    def _coerce_items(cls, value: Any) -> Any:
        return value or []


def _make_answer_guardrail(question: str):
    """Validate the final answer against the medical ANSWER_SYSTEM rules.

    CrewAI calls this with the task output and re-prompts the agent (up to
    ``max_retries``) when it returns ``(False, message)``.
    """

    needs_emergency = any(keyword in question for keyword in EMERGENCY_KEYWORDS)

    # NOTE: no return annotation on purpose. CrewAI inspects the guardrail's
    # return annotation and only accepts ``Tuple[bool, Any]``; leaving it
    # unannotated is the supported way to return ``(ok, value_or_message)``.
    def guardrail(task_output: Any):
        text = (getattr(task_output, "raw", None) or str(task_output)).strip()
        if not text:
            return (False, "빈 답변입니다. evidence pack의 근거만으로 답변을 작성하세요.")
        is_insufficient = "충분한 근거를 찾지 못" in text
        if not is_insufficient and not re.search(r"\[\d+\]", text):
            return (
                False,
                "근거로 사용한 출처를 [1], [2] 형식으로 본문에 표기해 다시 작성하세요.",
            )
        if needs_emergency and ("119" not in text and "응급" not in text):
            return (
                False,
                "응급 의심 증상이 포함된 질문입니다. 다른 안내보다 먼저 즉시 119 또는 "
                "응급실 방문을 권고하는 문장을 포함해 다시 작성하세요.",
            )
        return (True, text)

    return guardrail


class CrewRAGOrchestrator:
    """Agentic RAG flow built on CrewAI with a deterministic fallback."""

    def __init__(
        self,
        retrieval_service: Any,
        settings: AgentSettings | None = None,
        reranker: Any | None = None,
    ) -> None:
        self.settings = settings or AgentSettings.from_env()
        self.retrieval_service = retrieval_service
        self.reranker = reranker
        # Deterministic engine reused for retrieval/rerank glue and as fallback.
        self._engine = RAGFlowFacade(
            retrieval_service,
            settings=self.settings,
            reranker=reranker,
            enable_llm=False,
        )

    # ------------------------------------------------------------------ public

    def run(self, question: str, top_k: int | None = None) -> RAGFlowResult:
        question = (question or "").strip()
        if not question:
            return RAGFlowResult(
                answer="질문을 입력해 주세요.",
                sources=[],
                agent_trace={"query_variants": [], "selected_chunk_count": 0, "warnings": []},
            )

        if not _crewai_available():
            return self._fallback(question, top_k, ["crewai not installed; using deterministic pipeline"])

        try:
            return self._run_crew(question, top_k)
        except Exception as exc:  # noqa: BLE001 - any crew/runtime failure degrades safely
            return self._fallback(question, top_k, [f"crew execution failed: {exc}"])

    # ------------------------------------------------------------------ crew flow

    def _run_crew(self, question: str, top_k: int | None) -> RAGFlowResult:
        requested_top_k = top_k or self.settings.default_top_k
        retrieval_top_k = self._engine._retrieval_candidate_count(requested_top_k)
        warnings: list[str] = []
        store: dict[str, SourceChunk] = {}
        queries: list[str] = []

        # Stage 1 (Tools): the research agent decides how to query and calls the tool.
        self._agentic_search(question, store, queries, retrieval_top_k, warnings)
        if not store:
            self._deterministic_search(
                [question], store, queries, retrieval_top_k, warnings,
                note="agentic search returned no chunks; retrieving deterministically",
            )

        candidates: list[SourceChunk] = []
        selected: list[SourceChunk] = []
        selection = EvidenceSelection(selected_chunk_ids=[], rejected_chunk_ids=[], rationale="")

        # Stage 2 (controlled re-search loop): rerank -> select, broaden once if empty.
        for attempt in range(2):
            candidates = self._engine._rerank(question, list(store.values()), requested_top_k, warnings)
            if not candidates:
                break
            selection = self._select(question, candidates, warnings)
            selected = [c for c in candidates if c.chunk_id in set(selection.selected_chunk_ids)]
            if selected or attempt == 1:
                break
            warnings.append("evidence insufficient; broadening query and re-retrieving (agentic retry)")
            broadened = self._engine._fallback_query_plan(question).query_variants
            self._deterministic_search(broadened, store, queries, retrieval_top_k, warnings)

        if not candidates:
            return self._insufficient(question, queries, store, warnings)

        if not selected:
            selection = self._engine._fallback_select(question, candidates)
            selected = [c for c in candidates if c.chunk_id in set(selection.selected_chunk_ids)]
            warnings.append("selector found nothing after retry; used deterministic keyword/score selection")

        # Stage 3: compress -> answer (with guardrail).
        evidence_pack = self._compress(selected, warnings)
        answer = self._answer(question, evidence_pack, warnings)

        return RAGFlowResult(
            answer=answer,
            sources=evidence_pack.citations,
            agent_trace=self._trace(queries, store, candidates, selected, selection, warnings),
        )

    def _agentic_search(
        self,
        question: str,
        store: dict[str, SourceChunk],
        queries: list[str],
        retrieval_top_k: int,
        warnings: list[str],
    ) -> None:
        try:
            from crewai import Agent, Task

            tool = self._build_search_tool(store, queries, retrieval_top_k, warnings)
            researcher = Agent(
                role="의료 RAG 질의 계획·검색 담당",
                goal="사용자 질문을 적절한 한국어 검색 쿼리로 바꾸고 도구로 근거 문서를 찾는다",
                backstory=QUERY_PLANNER_SYSTEM,
                tools=[tool],
                llm=self._make_llm(self.settings.sub_model),
                max_iter=3,
                allow_delegation=False,
                verbose=False,
            )
            task = Task(
                description=(
                    f"사용자 질문: {question}\n\n"
                    "이 질문을 검색에 적합한 한국어 쿼리 1~3개로 바꾸고, 각 쿼리로 "
                    "search_clinical_docs 도구를 호출해 근거 문서를 검색하세요. "
                    "질환명은 한글명·영문명·약어 동의어를 함께 고려하세요. "
                    "결과가 부족하면 다른 쿼리로 다시 검색하세요."
                ),
                expected_output="검색에 사용한 쿼리 목록과 찾은 근거 요약",
                agent=researcher,
            )
            self._kickoff(researcher, task)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"agentic research stage failed: {exc}")

    def _build_search_tool(
        self,
        store: dict[str, SourceChunk],
        queries: list[str],
        retrieval_top_k: int,
        warnings: list[str],
    ):
        from crewai.tools import tool

        engine = self._engine

        @tool("search_clinical_docs")
        def search_clinical_docs(query: str) -> str:
            """한국 임상 진료지침·질병관리청 자료에서 query로 근거 청크를 검색한다.
            근거가 부족하면 다른 query로 다시 호출할 수 있다."""
            normalized = (query or "").strip()
            if not normalized:
                return "빈 쿼리입니다. 검색할 한국어 쿼리를 입력하세요."
            queries.append(normalized)
            chunks = engine._retrieve([normalized], retrieval_top_k, warnings)
            for chunk in chunks:
                store[chunk.chunk_id] = chunk
            if not chunks:
                return f"'{normalized}'에 대한 검색 결과가 없습니다. 다른 쿼리를 시도하세요."
            return "\n\n".join(_format_chunk_for_prompt(chunk) for chunk in chunks[:retrieval_top_k])

        return search_clinical_docs

    def _deterministic_search(
        self,
        variants: list[str],
        store: dict[str, SourceChunk],
        queries: list[str],
        retrieval_top_k: int,
        warnings: list[str],
        note: str | None = None,
    ) -> None:
        if note:
            warnings.append(note)
        chunks = self._engine._retrieve(variants, retrieval_top_k, warnings)
        for chunk in chunks:
            store[chunk.chunk_id] = chunk
        queries.extend(variants)

    def _select(
        self,
        question: str,
        candidates: list[SourceChunk],
        warnings: list[str],
    ) -> EvidenceSelection:
        try:
            from crewai import Agent, Task

            chunk_text, id_map = self._format_candidates(candidates)
            selector = Agent(
                role="의료 RAG 근거 선별자",
                goal="질문에 실제로 답하는 근거 chunk_id만 선별한다",
                backstory=EVIDENCE_SELECTOR_SYSTEM,
                llm=self._make_llm(self.settings.sub_model),
                allow_delegation=False,
                verbose=False,
            )
            task = Task(
                description=(
                    f"[질문]\n{question}\n\n[검색 청크]\n{chunk_text}\n\n"
                    "위 청크 중 질문에 답하는 것만 골라 selected_chunk_ids에 chunk_id"
                    "(C1, C2 형태)를 그대로 넣으세요."
                ),
                expected_output="selected_chunk_ids / rejected_chunk_ids / rationale 구조",
                agent=selector,
                output_pydantic=EvidenceSelectionOut,
            )
            parsed = getattr(self._kickoff(selector, task), "pydantic", None)
            if isinstance(parsed, EvidenceSelectionOut):
                selected = [
                    id_map[ref].chunk_id for ref in parsed.selected_chunk_ids if ref in id_map
                ][: self.settings.max_selected_chunks]
                rejected = [id_map[ref].chunk_id for ref in parsed.rejected_chunk_ids if ref in id_map]
                if selected:
                    return EvidenceSelection(
                        selected_chunk_ids=selected,
                        rejected_chunk_ids=rejected,
                        rationale=parsed.rationale.strip(),
                    )
            warnings.append("selector returned no usable chunk_ids")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"selector stage failed: {exc}")
        return EvidenceSelection(selected_chunk_ids=[], rejected_chunk_ids=[], rationale="")

    def _compress(self, chunks: list[SourceChunk], warnings: list[str]) -> EvidencePack:
        if not chunks:
            return EvidencePack(items=[])
        try:
            from crewai import Agent, Task

            chunk_text, id_map = self._format_candidates(chunks)
            compressor = Agent(
                role="의료 RAG 근거 압축자",
                goal="선택된 청크를 출처별 핵심 근거 bullet로 압축한다",
                backstory=EVIDENCE_COMPRESSOR_SYSTEM,
                llm=self._make_llm(self.settings.sub_model),
                allow_delegation=False,
                verbose=False,
            )
            task = Task(
                description=(
                    f"[선택 청크]\n{chunk_text}\n\n"
                    "각 항목의 chunk_id에는 위 청크의 chunk_id(C1, C2 형태)를 그대로 넣으세요."
                ),
                expected_output="items[] (chunk_id / bullet / snippet) 구조",
                agent=compressor,
                output_pydantic=EvidencePackOut,
            )
            parsed = getattr(self._kickoff(compressor, task), "pydantic", None)
            if isinstance(parsed, EvidencePackOut) and parsed.items:
                items: list[EvidenceItem] = []
                for value in parsed.items:
                    chunk = id_map.get(value.chunk_id)
                    if chunk is None:
                        continue
                    bullet = value.bullet.strip() or _summarize_text(chunk.text, max_chars=220)
                    snippet = value.snippet.strip() or _snippet(chunk.text)
                    items.append(_evidence_item(chunk, bullet, snippet))
                if items:
                    return EvidencePack(items=items[: self.settings.max_selected_chunks])
            warnings.append("compressor returned no usable items; using deterministic compression")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"compressor stage failed: {exc}")
        return EvidencePack(
            items=[_fallback_evidence_item(chunk) for chunk in chunks[: self.settings.max_selected_chunks]]
        )

    def _answer(self, question: str, evidence_pack: EvidencePack, warnings: list[str]) -> str:
        if not evidence_pack.items:
            return INSUFFICIENT_EVIDENCE_ANSWER

        evidence_text = self._evidence_text(evidence_pack)
        try:
            from crewai import Agent, Task

            answerer = Agent(
                role="건강정보 어시스턴트",
                goal="evidence pack의 근거만으로 출처를 표기한 한국어 답변을 작성한다",
                backstory=ANSWER_SYSTEM,
                llm=self._make_llm(self.settings.main_model),
                allow_delegation=False,
                verbose=False,
            )
            task = Task(
                description=f"[질문]\n{question}\n\n[evidence pack]\n{evidence_text}",
                expected_output="출처 [n] 표기를 포함한 한국어 답변",
                agent=answerer,
                guardrail=_make_answer_guardrail(question),
                max_retries=2,
            )
            text = (getattr(self._kickoff(answerer, task), "raw", None) or "").strip()
            if text:
                return text
            warnings.append("answer agent returned empty output; using deterministic summary")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"answer stage failed: {exc}; using deterministic summary")

        lines = ["검색된 문서 근거만 바탕으로 요약합니다.", ""]
        for idx, item in enumerate(evidence_pack.items, start=1):
            citation = item.citation
            lines.append(f"- {item.bullet} [{idx}]")
            lines.append(f"  출처: {citation.file_name} {_page_label(citation.page_start, citation.page_end)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ helpers

    def _format_candidates(self, chunks: list[SourceChunk]) -> tuple[str, dict[str, SourceChunk]]:
        """Render chunks with short ordinal ids (C1, C2, ...).

        Long real chunk_ids are hard for small local models to copy verbatim,
        which made selection/compression fail and fall back. Short ordinals are
        copied reliably and mapped back to real chunks here.
        """
        lines: list[str] = []
        id_map: dict[str, SourceChunk] = {}
        for index, chunk in enumerate(chunks, start=1):
            ref = f"C{index}"
            id_map[ref] = chunk
            page_label = _page_label(chunk.page_start, chunk.page_end)
            lines.append(
                f"chunk_id: {ref}\n"
                f"source: {chunk.file_name}{page_label}\n"
                f"score: {chunk.score:.4f}\n"
                f"text: {chunk.text[:1600]}"
            )
        return "\n\n".join(lines), id_map

    def _kickoff(self, agent: Any, task: Any) -> Any:
        from crewai import Crew, Process

        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        return crew.kickoff()

    def _make_llm(self, model: str) -> Any:
        from crewai import LLM

        return LLM(
            model=f"ollama/{model}",
            base_url=self.settings.ollama_base_url,
            temperature=0.2,
        )

    def _evidence_text(self, evidence_pack: EvidencePack) -> str:
        lines: list[str] = []
        for idx, item in enumerate(evidence_pack.items, start=1):
            citation = item.citation
            page_label = _page_label(citation.page_start, citation.page_end)
            lines.append(
                f"[{idx}] {citation.file_name} {page_label}\n"
                f"- 근거: {item.bullet}\n"
                f"- 원문: {citation.snippet}"
            )
        return "\n\n".join(lines) if lines else "(사용 가능한 근거 없음)"

    def _insufficient(
        self,
        question: str,
        queries: list[str],
        store: dict[str, SourceChunk],
        warnings: list[str],
    ) -> RAGFlowResult:
        return RAGFlowResult(
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            sources=[],
            agent_trace=self._trace(queries, store, [], [], None, warnings),
        )

    def _trace(
        self,
        queries: list[str],
        store: dict[str, SourceChunk],
        candidates: list[SourceChunk],
        selected: list[SourceChunk],
        selection: EvidenceSelection | None,
        warnings: list[str],
    ) -> dict[str, Any]:
        deduped_queries: list[str] = []
        for query in queries:
            if query not in deduped_queries:
                deduped_queries.append(query)
        rejected_count = len(selection.rejected_chunk_ids) if selection else 0
        return {
            "orchestrator": "crewai",
            "query_variants": deduped_queries,
            "retrieved_chunk_count": len(store),
            "reranked_chunk_count": len(candidates),
            "selected_chunk_count": len(selected),
            "rejected_chunk_count": rejected_count,
            "models": {
                "main": self.settings.main_model,
                "sub": self.settings.sub_model,
            },
            "crewai_available": True,
            "reranker_enabled": bool(self.reranker and getattr(self.reranker, "enabled", True)),
            "warnings": warnings,
        }

    # ------------------------------------------------------------------ fallback

    def _fallback(self, question: str, top_k: int | None, warnings: list[str]) -> RAGFlowResult:
        result = RAGFlowFacade(
            self.retrieval_service,
            settings=self.settings,
            reranker=self.reranker,
            enable_llm=True,
        ).run(question, top_k=top_k)
        trace = result.agent_trace
        trace["orchestrator"] = "deterministic_fallback"
        existing = trace.get("warnings") or []
        trace["warnings"] = warnings + list(existing)
        return result


def create_crew_orchestrator(
    retrieval_service: Any,
    settings: AgentSettings | None = None,
    reranker: Any | None = None,
) -> CrewRAGOrchestrator:
    return CrewRAGOrchestrator(retrieval_service, settings=settings, reranker=reranker)
