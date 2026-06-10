"""End-to-end PDF ingestion into Chroma."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from .chunker import TextChunker
from .embeddings import OllamaEmbeddingsClient
from .pdf_loader import PdfLoader, compute_file_hash, document_id_for_path
from .vector_store import ChromaVectorStore


def backend_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def default_processed_dir() -> Path:
    return backend_dir() / "data" / "processed"


@dataclass(frozen=True)
class DocumentIngestionStatus:
    document_id: str
    file_name: str
    file_hash: str | None = None
    status: str = "pending"
    chunk_count: int = 0
    page_count: int = 0
    low_text_pages: list[int] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class IngestionResult:
    processed: list[DocumentIngestionStatus]
    skipped: list[DocumentIngestionStatus]
    failed: list[DocumentIngestionStatus]
    total_pdfs: int
    force: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed": [asdict(item) for item in self.processed],
            "skipped": [asdict(item) for item in self.skipped],
            "failed": [asdict(item) for item in self.failed],
            "total_pdfs": self.total_pdfs,
            "force": self.force,
        }


class IngestionService:
    def __init__(
        self,
        pdf_loader: PdfLoader | None = None,
        chunker: TextChunker | None = None,
        embeddings_client: OllamaEmbeddingsClient | None = None,
        vector_store: ChromaVectorStore | None = None,
        processed_dir: Path | str | None = None,
    ) -> None:
        self.pdf_loader = pdf_loader or PdfLoader()
        self.chunker = chunker or TextChunker()
        self.embeddings_client = embeddings_client or OllamaEmbeddingsClient()
        self.vector_store = vector_store or ChromaVectorStore()
        self.processed_dir = Path(processed_dir) if processed_dir is not None else default_processed_dir()

    def ingest(self, force: bool = False) -> IngestionResult:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        pdf_paths = self.pdf_loader.discover()
        processed: list[DocumentIngestionStatus] = []
        skipped: list[DocumentIngestionStatus] = []
        failed: list[DocumentIngestionStatus] = []

        for pdf_path in pdf_paths:
            document_id = document_id_for_path(pdf_path)
            file_name = pdf_path.name
            try:
                file_hash = compute_file_hash(pdf_path)
                existing = self._read_document_metadata(document_id)
                if not force and existing and existing.get("file_hash") == file_hash:
                    skipped.append(
                        DocumentIngestionStatus(
                            document_id=document_id,
                            file_name=file_name,
                            file_hash=file_hash,
                            status="skipped",
                            chunk_count=int(existing.get("chunk_count") or 0),
                            page_count=int(existing.get("page_count") or 0),
                            low_text_pages=list(existing.get("low_text_pages") or []),
                        )
                    )
                    continue

                document = self.pdf_loader.load(pdf_path)
                chunks = self.chunker.chunk_document(document)

                self.vector_store.delete_document(document.document_id)
                if chunks:
                    embeddings = self.embeddings_client.embed_texts([chunk.text for chunk in chunks])
                    self.vector_store.add_chunks(chunks, embeddings)

                metadata = {
                    "document_id": document.document_id,
                    "file_name": document.file_name,
                    "file_path": str(document.file_path),
                    "file_hash": document.file_hash,
                    "title": document.title,
                    "page_count": document.page_count,
                    "chunk_count": len(chunks),
                    "low_text_pages": document.low_text_pages,
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                    "pdf_metadata": document.metadata,
                }
                self._write_document_metadata(document.document_id, metadata)
                processed.append(
                    DocumentIngestionStatus(
                        document_id=document.document_id,
                        file_name=document.file_name,
                        file_hash=document.file_hash,
                        status="processed",
                        chunk_count=len(chunks),
                        page_count=document.page_count,
                        low_text_pages=document.low_text_pages,
                    )
                )
            except Exception as exc:
                failed.append(
                    DocumentIngestionStatus(
                        document_id=document_id,
                        file_name=file_name,
                        status="failed",
                        error=str(exc),
                    )
                )

        self._write_index(processed=processed, skipped=skipped, failed=failed)
        return IngestionResult(
            processed=processed,
            skipped=skipped,
            failed=failed,
            total_pdfs=len(pdf_paths),
            force=force,
        )

    def _metadata_path(self, document_id: str) -> Path:
        return self.processed_dir / f"{document_id}.json"

    def _read_document_metadata(self, document_id: str) -> dict[str, Any] | None:
        path = self._metadata_path(document_id)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _write_document_metadata(self, document_id: str, metadata: dict[str, Any]) -> None:
        with self._metadata_path(document_id).open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2, sort_keys=True)

    def _write_index(
        self,
        processed: list[DocumentIngestionStatus],
        skipped: list[DocumentIngestionStatus],
        failed: list[DocumentIngestionStatus],
    ) -> None:
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "processed": [asdict(item) for item in processed],
            "skipped": [asdict(item) for item in skipped],
            "failed": [asdict(item) for item in failed],
        }
        with (self.processed_dir / "index.json").open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)


def ingest_documents(
    settings: object | None = None,
    force: bool = False,
    pdf_dir: Path | str | None = None,
    chroma_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Route-facing adapter for ingestion.

    Keeps the API layer decoupled from service classes while matching the
    existing IngestResponse contract.
    """

    pdf_path = _resolve_backend_relative(pdf_dir or getattr(settings, "pdf_dir", None))
    chroma_path = _resolve_backend_relative(chroma_dir or getattr(settings, "chroma_dir", None))
    processed_path = _resolve_backend_relative(processed_dir or getattr(settings, "processed_dir", None))

    embedding_model = getattr(settings, "embedding_model", None)
    ollama_base_url = getattr(settings, "ollama_base_url", None)

    result = IngestionService(
        pdf_loader=PdfLoader(pdf_dir=pdf_path),
        embeddings_client=OllamaEmbeddingsClient(model=embedding_model, base_url=ollama_base_url),
        vector_store=ChromaVectorStore(persist_dir=chroma_path),
        processed_dir=processed_path,
    ).ingest(force=force)

    documents_indexed = len(result.processed)
    chunks_total = sum(item.chunk_count for item in [*result.processed, *result.skipped])
    if result.failed:
        status = "failed"
    elif documents_indexed == 0:
        status = "skipped"
    else:
        status = "completed"

    return {
        "status": status,
        "documents_total": result.total_pdfs,
        "documents_indexed": documents_indexed,
        "chunks_total": chunks_total,
        "failures": [
            {
                "document_id": item.document_id,
                "file_name": item.file_name,
                "error": item.error,
            }
            for item in result.failed
        ],
    }


def list_documents(
    settings: object | None = None,
    processed_dir: Path | str | None = None,
    chroma_dir: Path | str | None = None,
) -> list[dict[str, object]]:
    del chroma_dir
    processed_path = _resolve_backend_relative(processed_dir or getattr(settings, "processed_dir", None))
    if processed_path is None:
        processed_path = default_processed_dir()
    if not processed_path.exists():
        return []

    documents: list[dict[str, object]] = []
    for path in sorted(processed_path.glob("*.json")):
        if path.name == "index.json":
            continue
        try:
            with path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        documents.append(
            {
                "document_id": str(metadata.get("document_id", path.stem)),
                "file_name": str(metadata.get("file_name", "")),
                "pages": int(metadata.get("page_count") or 0),
                "chunks": int(metadata.get("chunk_count") or 0),
                "indexed_at": metadata.get("indexed_at"),
            }
        )
    return documents


def _resolve_backend_relative(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return backend_dir() / resolved
