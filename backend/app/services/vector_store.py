"""Chroma persistent vector storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunker import TextChunk


DEFAULT_COLLECTION_NAME = "pdf_chunks"


@dataclass(frozen=True)
class SourceChunk:
    chunk_id: str
    file_name: str
    page_start: int
    page_end: int
    text: str
    score: float


def default_chroma_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "chroma"


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: Path | str | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.persist_dir = Path(persist_dir) if persist_dir is not None else default_chroma_dir()
        self.collection_name = collection_name
        self._client: Any | None = None
        self._collection: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError("ChromaDB is required for vector storage. Install package 'chromadb'.") from exc

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    @property
    def collection(self) -> Any:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return 0

        self.collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[_metadata_for_chunk(chunk) for chunk in chunks],
        )
        return len(chunks)

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    def count(self) -> int:
        return int(self.collection.count())

    def query(self, query_embedding: list[float], top_k: int = 8) -> list[SourceChunk]:
        if top_k <= 0:
            return []

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = _first_result_list(result.get("ids"))
        documents = _first_result_list(result.get("documents"))
        metadatas = _first_result_list(result.get("metadatas"))
        distances = _first_result_list(result.get("distances"))

        chunks: list[SourceChunk] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
            text = documents[index] if index < len(documents) and isinstance(documents[index], str) else ""
            distance = distances[index] if index < len(distances) else None
            chunks.append(
                SourceChunk(
                    chunk_id=str(chunk_id),
                    file_name=str(metadata.get("file_name", "")),
                    page_start=int(metadata.get("page_start", 0)),
                    page_end=int(metadata.get("page_end", 0)),
                    text=text,
                    score=_score_from_distance(distance),
                )
            )
        return chunks


def _metadata_for_chunk(chunk: TextChunk) -> dict[str, str | int]:
    return {
        "document_id": chunk.document_id,
        "file_name": chunk.file_name,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "file_hash": chunk.file_hash,
        "chunk_index": chunk.chunk_index,
    }


def _first_result_list(value: object) -> list[Any]:
    if not isinstance(value, list) or not value:
        return []
    first = value[0]
    return first if isinstance(first, list) else []


def _score_from_distance(distance: object) -> float:
    if distance is None:
        return 0.0
    try:
        numeric = float(distance)
    except (TypeError, ValueError):
        return 0.0
    if numeric < 0:
        return 0.0
    return 1.0 / (1.0 + numeric)

