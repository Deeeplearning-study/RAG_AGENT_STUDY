from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceChunk:
    chunk_id: str
    file_name: str
    page_start: int
    page_end: int
    text: str
    score: float = 0.0
    retrieval_score: float | None = None
    rerank_score: float | None = None

    @classmethod
    def from_any(cls, value: Any) -> "SourceChunk":
        if isinstance(value, cls):
            return value

        getter = value.get if isinstance(value, dict) else lambda key, default=None: getattr(value, key, default)
        chunk_id = str(
            getter("chunk_id")
            or getter("id")
            or f"{getter('file_name', getter('title', 'document'))}:{getter('page_start', getter('page', 0))}"
        )
        file_name = str(getter("file_name") or getter("title") or "document")
        page_start = _int_or_default(getter("page_start", getter("page", 1)), 1)
        page_end = _int_or_default(getter("page_end", getter("page", page_start)), page_start)
        text = str(getter("text") or getter("snippet") or "")
        score = _float_or_default(getter("score", 0.0), 0.0)
        retrieval_score = _optional_float(getter("retrieval_score", None))
        rerank_score = _optional_float(getter("rerank_score", None))
        return cls(
            chunk_id=chunk_id,
            file_name=file_name,
            page_start=page_start,
            page_end=page_end,
            text=text.strip(),
            score=score,
            retrieval_score=retrieval_score,
            rerank_score=rerank_score,
        )


@dataclass(frozen=True)
class QueryPlan:
    query_variants: list[str]
    keywords: list[str] = field(default_factory=list)
    search_intent: str = ""


@dataclass(frozen=True)
class EvidenceSelection:
    selected_chunk_ids: list[str]
    rejected_chunk_ids: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass(frozen=True)
class SourceCitation:
    file_name: str
    page_start: int
    page_end: int
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class EvidenceItem:
    chunk_id: str
    bullet: str
    citation: SourceCitation


@dataclass(frozen=True)
class EvidencePack:
    items: list[EvidenceItem]

    @property
    def citations(self) -> list[SourceCitation]:
        seen: set[tuple[str, int, int, str]] = set()
        citations: list[SourceCitation] = []
        for item in self.items:
            key = (
                item.citation.file_name,
                item.citation.page_start,
                item.citation.page_end,
                item.citation.snippet,
            )
            if key not in seen:
                seen.add(key)
                citations.append(item.citation)
        return citations


@dataclass(frozen=True)
class RAGFlowResult:
    answer: str
    sources: list[SourceCitation]
    agent_trace: dict[str, Any] = field(default_factory=dict)

    def to_chat_response(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
            "agent_trace": self.agent_trace,
        }


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
