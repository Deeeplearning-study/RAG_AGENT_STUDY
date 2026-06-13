from __future__ import annotations

import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.agents import AgentSettings, SourceChunk, create_rag_flow  # noqa: E402


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int) -> list[SourceChunk]:
        self.queries.append(query)
        return [
            SourceChunk(
                chunk_id="a",
                file_name="guideline.pdf",
                page_start=3,
                page_end=3,
                text="고열 환자는 수분 공급과 해열 치료를 우선 고려한다.",
                score=0.92,
            ),
            SourceChunk(
                chunk_id="a",
                file_name="guideline.pdf",
                page_start=3,
                page_end=3,
                text="고열 환자는 수분 공급과 해열 치료를 우선 고려한다.",
                score=0.92,
            ),
            SourceChunk(
                chunk_id="b",
                file_name="other.pdf",
                page_start=9,
                page_end=10,
                text="퇴원 후 추적 관찰은 증상 변화에 따라 조정한다.",
                score=0.4,
            ),
        ][:top_k]


class ScoredRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[SourceChunk]:
        self.calls.append((query, top_k))
        return [
            SourceChunk(
                chunk_id="low",
                file_name="low.pdf",
                page_start=1,
                page_end=1,
                text="고열 치료 기준은 휴식과 관찰이다.",
                score=0.9,
            ),
            SourceChunk(
                chunk_id="high",
                file_name="high.pdf",
                page_start=2,
                page_end=2,
                text="고열 치료 기준은 수분 공급과 해열 치료이다.",
                score=0.1,
            ),
        ][:top_k]


class ReverseReranker:
    enabled = True

    def rerank(self, question: str, chunks: list[SourceChunk], limit: int | None = None) -> list[SourceChunk]:
        reranked = [
            SourceChunk(
                chunk_id=chunk.chunk_id,
                file_name=chunk.file_name,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text=chunk.text,
                score=10.0 if chunk.chunk_id == "high" else 0.1,
                retrieval_score=chunk.score,
                rerank_score=10.0 if chunk.chunk_id == "high" else 0.1,
            )
            for chunk in chunks
        ]
        reranked.sort(key=lambda chunk: chunk.score, reverse=True)
        return reranked[:limit] if limit is not None else reranked


class FailingReranker:
    enabled = True

    def rerank(self, question: str, chunks: list[SourceChunk], limit: int | None = None) -> list[SourceChunk]:
        raise RuntimeError("model unavailable")


class FakeLLM:
    def generate(self, prompt: str, model: str) -> str:
        if "질의 계획자" in prompt:
            return json.dumps(
                {
                    "query_variants": ["고열 치료 기준", "해열 치료 수분 공급"],
                    "keywords": ["고열", "치료", "수분"],
                    "search_intent": "치료 기준 확인",
                },
                ensure_ascii=False,
            )
        if "근거 선별자" in prompt:
            return json.dumps(
                {
                    "selected_chunk_ids": ["a"],
                    "rejected_chunk_ids": ["b"],
                    "rationale": "질문과 직접 관련된 치료 기준",
                },
                ensure_ascii=False,
            )
        if "근거 압축자" in prompt:
            return json.dumps(
                {
                    "items": [
                        {
                            "chunk_id": "a",
                            "bullet": "고열 환자는 수분 공급과 해열 치료를 우선 고려한다.",
                            "snippet": "수분 공급과 해열 치료를 우선 고려",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        return "고열 치료는 문서 근거에 따르면 수분 공급과 해열 치료를 우선 고려합니다. [1]"


def test_sequential_flow_uses_retrieval_injection_and_returns_sources() -> None:
    retriever = FakeRetriever()
    flow = create_rag_flow(
        retriever,
        settings=AgentSettings(max_selected_chunks=3),
        llm=FakeLLM(),
    )

    result = flow.run("고열 치료 기준은?", top_k=3)
    payload = result.to_chat_response()

    assert retriever.queries == ["고열 치료 기준", "해열 치료 수분 공급"]
    assert payload["answer"].startswith("고열 치료는")
    assert payload["sources"] == [
        {
            "file_name": "guideline.pdf",
            "page_start": 3,
            "page_end": 3,
            "snippet": "수분 공급과 해열 치료를 우선 고려",
        }
    ]
    assert payload["agent_trace"]["selected_chunk_count"] == 1
    assert payload["agent_trace"]["query_variants"] == ["고열 치료 기준", "해열 치료 수분 공급"]


def test_flow_returns_insufficient_evidence_without_retrieved_chunks() -> None:
    flow = create_rag_flow(
        lambda query, top_k: [],
        settings=AgentSettings(default_top_k=2),
        enable_llm=False,
    )

    result = flow.run("문서에 없는 질문")

    assert "충분한 근거를 찾지 못했습니다" in result.answer
    assert result.sources == []
    assert result.agent_trace["selected_chunk_count"] == 0


def test_flow_falls_back_when_llm_is_unavailable() -> None:
    retriever = FakeRetriever()
    flow = create_rag_flow(
        retriever,
        settings=AgentSettings(max_selected_chunks=2),
        enable_llm=False,
    )

    result = flow.run("고열 치료 기준은?", top_k=2)

    assert "검색된 문서 근거만 바탕으로 요약합니다" in result.answer
    assert len(result.sources) >= 1
    assert result.agent_trace["warnings"]



def test_flow_reranks_retrieved_chunks_before_fallback_selection() -> None:
    retriever = ScoredRetriever()
    flow = create_rag_flow(
        retriever,
        settings=AgentSettings(
            max_selected_chunks=1,
            reranker_candidate_multiplier=3,
            reranker_max_candidates=9,
        ),
        enable_llm=False,
        reranker=ReverseReranker(),
    )

    result = flow.run("고열 치료 기준은?", top_k=2)

    assert result.sources[0].file_name == "high.pdf"
    assert result.agent_trace["reranked_chunk_count"] == 2
    assert result.agent_trace["reranker_enabled"] is True
    assert all(call[1] == 6 for call in retriever.calls)


def test_flow_falls_back_to_vector_order_when_reranker_fails() -> None:
    flow = create_rag_flow(
        ScoredRetriever(),
        settings=AgentSettings(max_selected_chunks=1),
        enable_llm=False,
        reranker=FailingReranker(),
    )

    result = flow.run("고열 치료 기준은?", top_k=2)

    assert result.sources[0].file_name == "low.pdf"
    assert any("reranker unavailable" in warning for warning in result.agent_trace["warnings"])
