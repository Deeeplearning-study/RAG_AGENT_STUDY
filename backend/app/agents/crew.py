from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .prompts import (
    build_answer_prompt,
    build_compressor_prompt,
    build_query_planner_prompt,
    build_selector_prompt,
)
from .schemas import (
    EvidenceItem,
    EvidencePack,
    EvidenceSelection,
    QueryPlan,
    RAGFlowResult,
    SourceChunk,
    SourceCitation,
)


INSUFFICIENT_EVIDENCE_ANSWER = (
    "문서에서 질문에 답할 충분한 근거를 찾지 못했습니다. "
    "PDF 인덱싱 상태를 확인하거나 질문에 포함된 용어를 더 구체화해 주세요."
)


class RetrievalService(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[Any]:
        ...


class RerankerService(Protocol):
    enabled: bool

    def rerank(self, question: str, chunks: list[Any], limit: int | None = None) -> list[Any]:
        ...


class TextLLM(Protocol):
    def generate(self, prompt: str, model: str) -> str:
        ...


@dataclass(frozen=True)
class AgentSettings:
    ollama_base_url: str = "http://localhost:11434"
    main_model: str = "gemma4:26b"
    sub_model: str = "gemma4:e4b"
    default_top_k: int = 8
    max_query_variants: int = 3
    max_selected_chunks: int = 5
    min_score: float = 0.0
    min_rerank_score: float = 0.0
    request_timeout_seconds: float = 45.0
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "auto"
    reranker_batch_size: int = 8
    reranker_max_length: int = 512
    reranker_candidate_multiplier: int = 3
    reranker_max_candidates: int = 40

    @classmethod
    def from_env(cls) -> "AgentSettings":
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url),
            main_model=os.getenv("MAIN_AGENT_MODEL", os.getenv("OLLAMA_MODEL", cls.main_model)),
            sub_model=os.getenv("SUB_AGENT_MODEL", cls.sub_model),
            default_top_k=_int_env("DEFAULT_TOP_K", cls.default_top_k),
            max_query_variants=_int_env("MAX_QUERY_VARIANTS", cls.max_query_variants),
            max_selected_chunks=_int_env("MAX_SELECTED_CHUNKS", cls.max_selected_chunks),
            min_score=_float_env("RAG_MIN_SCORE", cls.min_score),
            min_rerank_score=_float_env("RAG_MIN_RERANK_SCORE", cls.min_rerank_score),
            request_timeout_seconds=_float_env("OLLAMA_TIMEOUT_SECONDS", cls.request_timeout_seconds),
            reranker_enabled=_bool_env("RERANKER_ENABLED", cls.reranker_enabled),
            reranker_model=os.getenv("RERANKER_MODEL", cls.reranker_model),
            reranker_device=os.getenv("RERANKER_DEVICE", cls.reranker_device),
            reranker_batch_size=_int_env("RERANKER_BATCH_SIZE", cls.reranker_batch_size),
            reranker_max_length=_int_env("RERANKER_MAX_LENGTH", cls.reranker_max_length),
            reranker_candidate_multiplier=_int_env("RERANKER_CANDIDATE_MULTIPLIER", cls.reranker_candidate_multiplier),
            reranker_max_candidates=_int_env("RERANKER_MAX_CANDIDATES", cls.reranker_max_candidates),
        )


class OllamaTextClient:
    def __init__(self, base_url: str, timeout_seconds: float = 45.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, model: str) -> str:
        body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama generation failed for model '{model}': {exc}") from exc

        text = str(payload.get("response") or "").strip()
        if not text:
            raise RuntimeError(f"Ollama returned an empty response for model '{model}'.")
        return text


class RAGFlowFacade:
    """Sequential MVP facade for CrewAI-style RAG orchestration.

    The deterministic retrieval service is injected by the API/service layer.
    LLM calls are optional; if CrewAI or Ollama is unavailable, the flow still
    returns grounded fallback answers and trace warnings.
    """

    def __init__(
        self,
        retrieval_service: Any,
        settings: AgentSettings | None = None,
        llm: TextLLM | None = None,
        enable_llm: bool = True,
        reranker: RerankerService | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.settings = settings or AgentSettings.from_env()
        self.llm = llm or OllamaTextClient(
            self.settings.ollama_base_url,
            timeout_seconds=self.settings.request_timeout_seconds,
        )
        self.enable_llm = enable_llm
        self.reranker = reranker
        self.crewai_available = _crewai_available()

    def run(self, question: str, top_k: int | None = None) -> RAGFlowResult:
        question = question.strip()
        requested_top_k = top_k or self.settings.default_top_k
        warnings: list[str] = []

        if not question:
            return RAGFlowResult(
                answer="질문을 입력해 주세요.",
                sources=[],
                agent_trace={"query_variants": [], "selected_chunk_count": 0, "warnings": []},
            )

        query_plan = self._plan_query(question, warnings)
        retrieval_top_k = self._retrieval_candidate_count(requested_top_k)
        retrieved_chunks = self._retrieve(query_plan.query_variants, retrieval_top_k, warnings)
        reranked_chunks = self._rerank(question, retrieved_chunks, requested_top_k, warnings)
        selected = self._select_evidence(question, reranked_chunks, warnings)
        selected_chunks = [chunk for chunk in reranked_chunks if chunk.chunk_id in set(selected.selected_chunk_ids)]
        evidence_pack = self._compress_evidence(selected_chunks, warnings)
        answer = self._answer(question, evidence_pack, warnings)

        return RAGFlowResult(
            answer=answer,
            sources=evidence_pack.citations,
            agent_trace={
                "query_variants": query_plan.query_variants,
                "keywords": query_plan.keywords,
                "search_intent": query_plan.search_intent,
                "retrieved_chunk_count": len(retrieved_chunks),
                "reranked_chunk_count": len(reranked_chunks),
                "selected_chunk_count": len(selected_chunks),
                "rejected_chunk_count": len(selected.rejected_chunk_ids),
                "models": {
                    "main": self.settings.main_model,
                    "sub": self.settings.sub_model,
                },
                "crewai_available": self.crewai_available,
                "reranker_enabled": bool(self.reranker and getattr(self.reranker, "enabled", True)),
                "warnings": warnings,
            },
        )

    def _plan_query(self, question: str, warnings: list[str]) -> QueryPlan:
        prompt = build_query_planner_prompt(question)
        payload = self._generate_json(prompt, self.settings.sub_model, warnings, "query_planner")
        variants = _clean_string_list(payload.get("query_variants"), limit=self.settings.max_query_variants)
        keywords = _clean_string_list(payload.get("keywords"), limit=12)
        intent = str(payload.get("search_intent") or "").strip()
        if variants:
            return QueryPlan(query_variants=variants, keywords=keywords, search_intent=intent)

        return self._fallback_query_plan(question)

    def _retrieve(self, query_variants: list[str], top_k: int, warnings: list[str]) -> list[SourceChunk]:
        merged: list[SourceChunk] = []
        seen: set[str] = set()

        for query in query_variants:
            try:
                raw_chunks = _call_retrieval_service(self.retrieval_service, query, top_k)
            except Exception as exc:
                warnings.append(f"retrieval failed for query '{query}': {exc}")
                continue

            for raw_chunk in raw_chunks:
                chunk = SourceChunk.from_any(raw_chunk)
                if not chunk.text:
                    continue
                dedupe_key = chunk.chunk_id or f"{chunk.file_name}:{chunk.page_start}:{chunk.text[:80]}"
                if dedupe_key in seen:
                    continue
                if chunk.score < self.settings.min_score:
                    continue
                seen.add(dedupe_key)
                merged.append(chunk)

        merged.sort(key=lambda chunk: chunk.score, reverse=True)
        return merged[: max(top_k * max(1, len(query_variants)), top_k)]

    def _retrieval_candidate_count(self, requested_top_k: int) -> int:
        if not self.reranker or not getattr(self.reranker, "enabled", True):
            return requested_top_k
        multiplier = max(1, int(self.settings.reranker_candidate_multiplier))
        max_candidates = max(1, int(self.settings.reranker_max_candidates))
        return min(max(requested_top_k, requested_top_k * multiplier), max_candidates)

    def _rerank(
        self,
        question: str,
        chunks: list[SourceChunk],
        limit: int,
        warnings: list[str],
    ) -> list[SourceChunk]:
        if not chunks:
            return []
        if not self.reranker or not getattr(self.reranker, "enabled", True):
            return chunks
        try:
            reranked = self.reranker.rerank(question, chunks, limit=limit)
        except Exception as exc:
            warnings.append(f"reranker unavailable; using vector score order: {exc}")
            return chunks[:limit]
        normalized: list[SourceChunk] = []
        for raw_chunk in reranked:
            chunk = SourceChunk.from_any(raw_chunk)
            if chunk.text:
                normalized.append(chunk)
        if not normalized:
            warnings.append("reranker returned no usable chunks; using vector score order")
            return chunks[:limit]
        gated = [
            chunk
            for chunk in normalized
            if chunk.rerank_score is None or chunk.rerank_score >= self.settings.min_rerank_score
        ]
        if not gated:
            warnings.append(
                f"all reranked chunks scored below min_rerank_score={self.settings.min_rerank_score}; "
                "treating as insufficient evidence"
            )
            return []
        return gated[:limit]

    def _select_evidence(
        self,
        question: str,
        chunks: list[SourceChunk],
        warnings: list[str],
    ) -> EvidenceSelection:
        if not chunks:
            return EvidenceSelection(selected_chunk_ids=[], rejected_chunk_ids=[], rationale="검색 결과 없음")

        prompt = build_selector_prompt(question, chunks)
        payload = self._generate_json(prompt, self.settings.sub_model, warnings, "evidence_selector")
        available_ids = {chunk.chunk_id for chunk in chunks}
        selected_ids = [
            chunk_id
            for chunk_id in _clean_string_list(payload.get("selected_chunk_ids"), self.settings.max_selected_chunks)
            if chunk_id in available_ids
        ]
        rejected_ids = [chunk_id for chunk_id in _clean_string_list(payload.get("rejected_chunk_ids"), len(chunks)) if chunk_id in available_ids]

        if selected_ids:
            return EvidenceSelection(
                selected_chunk_ids=selected_ids,
                rejected_chunk_ids=rejected_ids,
                rationale=str(payload.get("rationale") or "").strip(),
            )

        return self._fallback_select(question, chunks)

    def _compress_evidence(self, chunks: list[SourceChunk], warnings: list[str]) -> EvidencePack:
        if not chunks:
            return EvidencePack(items=[])

        prompt = build_compressor_prompt(chunks)
        payload = self._generate_json(prompt, self.settings.sub_model, warnings, "evidence_compressor")
        payload_items = payload.get("items") if isinstance(payload.get("items"), list) else []
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        items: list[EvidenceItem] = []

        for value in payload_items:
            if not isinstance(value, dict):
                continue
            chunk_id = str(value.get("chunk_id") or "").strip()
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            bullet = str(value.get("bullet") or "").strip()
            snippet = str(value.get("snippet") or "").strip()
            if not bullet:
                bullet = _summarize_text(chunk.text, max_chars=220)
            if not snippet:
                snippet = _snippet(chunk.text)
            items.append(_evidence_item(chunk, bullet, snippet))

        if items:
            return EvidencePack(items=items[: self.settings.max_selected_chunks])

        return EvidencePack(items=[_fallback_evidence_item(chunk) for chunk in chunks[: self.settings.max_selected_chunks]])

    def _answer(self, question: str, evidence_pack: EvidencePack, warnings: list[str]) -> str:
        if not evidence_pack.items:
            return INSUFFICIENT_EVIDENCE_ANSWER

        prompt = build_answer_prompt(question, evidence_pack)
        answer = self._generate_text(prompt, self.settings.main_model, warnings, "main_answer")
        if answer:
            return answer

        warnings.append("using deterministic evidence-summary answer because the main answer model is unavailable")
        lines = [
            "Ollama 또는 CrewAI 답변 생성을 사용할 수 없어, 검색된 문서 근거만 바탕으로 요약합니다.",
            "",
        ]
        for idx, item in enumerate(evidence_pack.items, start=1):
            citation = item.citation
            lines.append(f"- {item.bullet} [{idx}]")
            lines.append(f"  출처: {citation.file_name} {_page_label(citation.page_start, citation.page_end)}")
        return "\n".join(lines)

    def _generate_json(self, prompt: str, model: str, warnings: list[str], step: str) -> dict[str, Any]:
        text = self._generate_text(prompt, model, warnings, step)
        if not text:
            return {}
        try:
            return _extract_json_object(text)
        except ValueError as exc:
            warnings.append(f"{step} returned non-json output: {exc}")
            return {}

    def _generate_text(self, prompt: str, model: str, warnings: list[str], step: str) -> str:
        if not self.enable_llm:
            return ""
        try:
            return self.llm.generate(prompt, model).strip()
        except Exception as exc:
            warnings.append(f"{step} unavailable for model '{model}': {exc}")
            return ""

    def _fallback_query_plan(self, question: str) -> QueryPlan:
        keywords = _keyword_candidates(question)
        variants = [question]
        if keywords:
            variants.append(" ".join(keywords[:6]))
        variants.append(f"{question} 근거 출처 페이지")
        return QueryPlan(
            query_variants=_dedupe_keep_order(variants)[: self.settings.max_query_variants],
            keywords=keywords,
            search_intent="deterministic fallback query planning",
        )

    def _fallback_select(self, question: str, chunks: list[SourceChunk]) -> EvidenceSelection:
        question_terms = set(_keyword_candidates(question))
        ranked: list[tuple[float, SourceChunk]] = []
        for chunk in chunks:
            text_terms = set(_keyword_candidates(chunk.text))
            overlap = len(question_terms & text_terms)
            relevance = overlap + max(chunk.score, 0.0)
            ranked.append((relevance, chunk))

        ranked.sort(key=lambda pair: pair[0], reverse=True)
        selected = [
            chunk.chunk_id
            for relevance, chunk in ranked
            if relevance > 0 or chunk.score > 0
        ][: self.settings.max_selected_chunks]
        if not selected and chunks and not question_terms:
            selected = [chunks[0].chunk_id]
        rejected = [chunk.chunk_id for chunk in chunks if chunk.chunk_id not in set(selected)]
        return EvidenceSelection(
            selected_chunk_ids=selected,
            rejected_chunk_ids=rejected,
            rationale="deterministic keyword/score fallback selection",
        )


def create_rag_flow(
    retrieval_service: Any,
    settings: AgentSettings | None = None,
    llm: TextLLM | None = None,
    enable_llm: bool = True,
    reranker: RerankerService | None = None,
) -> RAGFlowFacade:
    return RAGFlowFacade(
        retrieval_service=retrieval_service,
        settings=settings,
        llm=llm,
        enable_llm=enable_llm,
        reranker=reranker,
    )


def answer_question(
    message: str,
    top_k: int | None = None,
    settings: Any | None = None,
    include_agent_trace: bool = True,
) -> dict[str, Any]:
    agent_settings = _agent_settings_from_app_settings(settings)
    retrieval_service = _default_retrieval_service(settings, agent_settings)
    reranker_service = _default_reranker_service(agent_settings)
    result = create_rag_flow(retrieval_service, settings=agent_settings, reranker=reranker_service).run(message, top_k=top_k)
    payload = result.to_chat_response()
    if not include_agent_trace:
        payload["agent_trace"] = None
    return payload


def run_chat(
    message: str,
    top_k: int | None = None,
    settings: Any | None = None,
    include_agent_trace: bool = True,
) -> dict[str, Any]:
    return answer_question(
        message=message,
        top_k=top_k,
        settings=settings,
        include_agent_trace=include_agent_trace,
    )


def _call_retrieval_service(service: Any, query: str, top_k: int) -> list[Any]:
    if hasattr(service, "retrieve"):
        return list(service.retrieve(query, top_k))
    if hasattr(service, "search"):
        try:
            return list(service.search(query, top_k=top_k))
        except TypeError:
            return list(service.search(query, top_k))
    if hasattr(service, "retrieve_sources"):
        return list(service.retrieve_sources(query, top_k))
    if callable(service):
        return list(service(query, top_k))
    raise TypeError("retrieval_service must expose retrieve/search/retrieve_sources or be callable")


def _crewai_available() -> bool:
    try:
        __import__("crewai")
    except Exception:
        return False
    return True


def _agent_settings_from_app_settings(settings: Any | None) -> AgentSettings:
    if settings is None:
        return AgentSettings.from_env()
    return AgentSettings(
        ollama_base_url=str(getattr(settings, "ollama_base_url", AgentSettings.ollama_base_url)),
        main_model=str(getattr(settings, "main_agent_model", AgentSettings.main_model)),
        sub_model=str(getattr(settings, "sub_agent_model", AgentSettings.sub_model)),
        default_top_k=int(getattr(settings, "default_top_k", AgentSettings.default_top_k)),
        min_rerank_score=float(getattr(settings, "min_rerank_score", AgentSettings.min_rerank_score)),
        reranker_enabled=bool(getattr(settings, "reranker_enabled", AgentSettings.reranker_enabled)),
        reranker_model=str(getattr(settings, "reranker_model", AgentSettings.reranker_model)),
        reranker_device=str(getattr(settings, "reranker_device", AgentSettings.reranker_device)),
        reranker_batch_size=int(getattr(settings, "reranker_batch_size", AgentSettings.reranker_batch_size)),
        reranker_max_length=int(getattr(settings, "reranker_max_length", AgentSettings.reranker_max_length)),
        reranker_candidate_multiplier=int(getattr(settings, "reranker_candidate_multiplier", AgentSettings.reranker_candidate_multiplier)),
        reranker_max_candidates=int(getattr(settings, "reranker_max_candidates", AgentSettings.reranker_max_candidates)),
    )


def _default_retrieval_service(settings: Any | None, agent_settings: AgentSettings) -> Any:
    try:
        from app.services.embeddings import OllamaEmbeddingsClient
        from app.services.retrieval import RetrievalService as AppRetrievalService
        from app.services.vector_store import ChromaVectorStore

        return AppRetrievalService(
            vector_store=ChromaVectorStore(
                persist_dir=getattr(settings, "chroma_dir", None),
            ),
            embeddings_client=OllamaEmbeddingsClient(
                model=getattr(settings, "embedding_model", None),
                base_url=agent_settings.ollama_base_url,
            ),
            default_top_k=agent_settings.default_top_k,
        )
    except Exception as exc:
        return _UnavailableRetrievalService(exc)


def _default_reranker_service(agent_settings: AgentSettings) -> RerankerService | None:
    if not agent_settings.reranker_enabled:
        return None
    try:
        from app.services.reranker import CrossEncoderReranker, RerankerConfig

        return CrossEncoderReranker(
            RerankerConfig(
                enabled=agent_settings.reranker_enabled,
                model=agent_settings.reranker_model,
                device=agent_settings.reranker_device,
                batch_size=agent_settings.reranker_batch_size,
                max_length=agent_settings.reranker_max_length,
            )
        )
    except Exception:
        return None


class _UnavailableRetrievalService:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def retrieve(self, query: str, top_k: int) -> list[Any]:
        raise RuntimeError(f"retrieval service is unavailable: {self.error}")


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON object found")
        stripped = stripped[start : end + 1]
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("JSON root is not an object")
    return payload


def _clean_string_list(value: Any, limit: int) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        return []
    cleaned = [str(item).strip() for item in values if str(item).strip()]
    return _dedupe_keep_order(cleaned)[:limit]


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _keyword_candidates(text: str) -> list[str]:
    tokens = re.findall(r"[0-9A-Za-z가-힣]{2,}", text.lower())
    stopwords = {
        "그리고",
        "또는",
        "에서",
        "으로",
        "에게",
        "관련",
        "문서",
        "질문",
        "what",
        "when",
        "where",
        "about",
        "the",
        "and",
        "for",
    }
    return _dedupe_keep_order([token for token in tokens if token not in stopwords])


def _fallback_evidence_item(chunk: SourceChunk) -> EvidenceItem:
    return _evidence_item(
        chunk,
        bullet=_summarize_text(chunk.text, max_chars=220),
        snippet=_snippet(chunk.text),
    )


def _evidence_item(chunk: SourceChunk, bullet: str, snippet: str) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk.chunk_id,
        bullet=bullet,
        citation=SourceCitation(
            file_name=chunk.file_name,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            snippet=snippet,
        ),
    )


def _summarize_text(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _snippet(text: str, max_chars: int = 260) -> str:
    return _summarize_text(text, max_chars=max_chars)


def _page_label(page_start: int, page_end: int) -> str:
    if page_start == page_end:
        return f"p.{page_start}"
    return f"pp.{page_start}-{page_end}"


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
