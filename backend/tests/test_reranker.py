from __future__ import annotations

from app.agents import SourceChunk
from app.services.reranker import CrossEncoderReranker, RerankerConfig


class FakeCrossEncoderReranker(CrossEncoderReranker):
    def _predict_scores(self, pairs: list[list[str]]) -> list[float]:
        return [0.2, 2.5]


def test_cross_encoder_reranker_sorts_and_preserves_scores_without_model_load() -> None:
    chunks = [
        SourceChunk("a", "a.pdf", 1, 1, "alpha", score=0.9),
        SourceChunk("b", "b.pdf", 2, 2, "beta", score=0.1),
    ]

    reranked = FakeCrossEncoderReranker(RerankerConfig()).rerank("question", chunks, limit=2)

    assert [chunk.chunk_id for chunk in reranked] == ["b", "a"]
    assert reranked[0].score == 2.5
    assert reranked[0].retrieval_score == 0.1
    assert reranked[0].rerank_score == 2.5


def test_cross_encoder_reranker_returns_original_order_when_disabled() -> None:
    chunks = [
        SourceChunk("a", "a.pdf", 1, 1, "alpha", score=0.9),
        SourceChunk("b", "b.pdf", 2, 2, "beta", score=0.1),
    ]

    reranked = CrossEncoderReranker(RerankerConfig(enabled=False)).rerank("question", chunks, limit=1)

    assert [chunk.chunk_id for chunk in reranked] == ["a"]
