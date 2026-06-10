"""Retrieval interface returning normalized SourceChunk-compatible results."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .embeddings import OllamaEmbeddingsClient
from .vector_store import ChromaVectorStore, SourceChunk


DEFAULT_TOP_K = 8


class RetrievalService:
    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        embeddings_client: OllamaEmbeddingsClient | None = None,
        default_top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.vector_store = vector_store or ChromaVectorStore()
        self.embeddings_client = embeddings_client or OllamaEmbeddingsClient()
        self.default_top_k = default_top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[SourceChunk]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        query_embedding = self.embeddings_client.embed_query(normalized_query)
        return self.vector_store.query(query_embedding=query_embedding, top_k=top_k or self.default_top_k)

    def retrieve_as_dicts(self, query: str, top_k: int | None = None) -> list[dict[str, object]]:
        return [asdict(chunk) for chunk in self.retrieve(query=query, top_k=top_k)]


def retrieve_chunks(
    query: str,
    settings: object | None = None,
    top_k: int | None = None,
    chroma_dir: Path | str | None = None,
) -> list[dict[str, object]]:
    chroma_path = _resolve_backend_relative(chroma_dir or getattr(settings, "chroma_dir", None))
    embedding_model = getattr(settings, "embedding_model", None)
    ollama_base_url = getattr(settings, "ollama_base_url", None)
    default_top_k = int(getattr(settings, "default_top_k", DEFAULT_TOP_K) or DEFAULT_TOP_K)

    service = RetrievalService(
        vector_store=ChromaVectorStore(persist_dir=chroma_path),
        embeddings_client=OllamaEmbeddingsClient(model=embedding_model, base_url=ollama_base_url),
        default_top_k=default_top_k,
    )
    return service.retrieve_as_dicts(query=query, top_k=top_k)


def _resolve_backend_relative(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return Path(__file__).resolve().parents[2] / resolved
