"""Cross-Encoder reranking service for retrieved RAG chunks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool = True
    model: str = DEFAULT_RERANKER_MODEL
    device: str = "auto"
    batch_size: int = 8
    max_length: int = 512


class CrossEncoderReranker:
    """Lazy Cross-Encoder reranker.

    The heavy sentence-transformers import and model load happen on first use,
    so tests and vector-only runs do not pay the startup cost.
    """

    def __init__(self, config: RerankerConfig | None = None) -> None:
        self.config = config or RerankerConfig()
        self._model: Any | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def rerank(self, question: str, chunks: list[Any], limit: int | None = None) -> list[Any]:
        if not self.config.enabled or not chunks:
            return chunks[:limit] if limit is not None else list(chunks)

        normalized_question = question.strip()
        if not normalized_question:
            return chunks[:limit] if limit is not None else list(chunks)

        pairs = [[normalized_question, _chunk_text(chunk)] for chunk in chunks]
        scores = self._predict_scores(pairs)
        ranked = [(_with_rerank_score(chunk, score), score) for chunk, score in zip(chunks, scores, strict=False)]
        ranked.sort(key=lambda item: item[1], reverse=True)
        result = [chunk for chunk, _ in ranked]
        return result[:limit] if limit is not None else result

    def _predict_scores(self, pairs: list[list[str]]) -> list[float]:
        model = self._get_model()
        raw_scores = model.predict(
            pairs,
            batch_size=max(1, int(self.config.batch_size)),
            convert_to_numpy=False,
            show_progress_bar=False,
        )
        return [_coerce_float(score) for score in raw_scores]

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is required for Cross-Encoder reranking.") from exc

            kwargs: dict[str, Any] = {"max_length": max(1, int(self.config.max_length))}
            if self.config.device and self.config.device != "auto":
                kwargs["device"] = self.config.device
            self._model = CrossEncoder(self.config.model, **kwargs)
        return self._model


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("text") or chunk.get("snippet") or "")
    return str(getattr(chunk, "text", getattr(chunk, "snippet", "")) or "")


def _with_rerank_score(chunk: Any, rerank_score: float) -> Any:
    existing_retrieval_score = _get_value(chunk, "retrieval_score", None)
    retrieval_score = _coerce_float(
        _get_value(chunk, "score", 0.0) if existing_retrieval_score is None else existing_retrieval_score
    )
    if hasattr(chunk, "__dataclass_fields__"):
        updates: dict[str, Any] = {"score": rerank_score}
        fields = getattr(chunk, "__dataclass_fields__", {})
        if "retrieval_score" in fields:
            updates["retrieval_score"] = retrieval_score
        if "rerank_score" in fields:
            updates["rerank_score"] = rerank_score
        return replace(chunk, **updates)
    if isinstance(chunk, dict):
        updated = dict(chunk)
        updated.setdefault("retrieval_score", retrieval_score)
        updated["rerank_score"] = rerank_score
        updated["score"] = rerank_score
        return updated
    return chunk


def _get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
