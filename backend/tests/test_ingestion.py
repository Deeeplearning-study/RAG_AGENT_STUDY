import json

from backend.app.services.ingestion import IngestionService
from backend.app.services.pdf_loader import PdfDocument, PdfPage, compute_file_hash, document_id_for_path


class FakePdfLoader:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.load_calls = 0

    def discover(self):
        return [self.pdf_path]

    def load(self, path):
        self.load_calls += 1
        file_hash = "changed" if self.load_calls > 1 else "hash"
        return PdfDocument(
            document_id="doc1",
            file_name=path.name,
            file_path=path,
            file_hash=file_hash,
            title="Title",
            page_count=1,
            pages=[PdfPage(page_number=1, text="alpha beta gamma delta", char_count=22)],
            metadata={},
            low_text_pages=[],
        )


class FakeEmbeddings:
    def __init__(self):
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += 1
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeVectorStore:
    def __init__(self):
        self.deleted = []
        self.added = []

    def delete_document(self, document_id):
        self.deleted.append(document_id)

    def add_chunks(self, chunks, embeddings):
        self.added.extend(chunks)
        return len(chunks)


def test_ingestion_skips_unchanged_documents(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"pdf-content")
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    document_id = document_id_for_path(pdf_path)
    (processed_dir / f"{document_id}.json").write_text(
        json.dumps({"file_hash": compute_file_hash(pdf_path), "chunk_count": 3, "page_count": 2}),
        encoding="utf-8",
    )

    loader = FakePdfLoader(pdf_path)
    embeddings = FakeEmbeddings()
    vector_store = FakeVectorStore()

    result = IngestionService(
        pdf_loader=loader,
        embeddings_client=embeddings,
        vector_store=vector_store,
        processed_dir=processed_dir,
    ).ingest(force=False)

    assert len(result.skipped) == 1
    assert result.skipped[0].status == "skipped"
    assert loader.load_calls == 0
    assert embeddings.calls == 0
    assert vector_store.added == []


def test_ingestion_force_reindexes_existing_document(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"pdf-content")
    processed_dir = tmp_path / "processed"

    loader = FakePdfLoader(pdf_path)
    embeddings = FakeEmbeddings()
    vector_store = FakeVectorStore()

    result = IngestionService(
        pdf_loader=loader,
        embeddings_client=embeddings,
        vector_store=vector_store,
        processed_dir=processed_dir,
    ).ingest(force=True)

    assert len(result.processed) == 1
    assert result.processed[0].chunk_count == 1
    assert vector_store.deleted == ["doc1"]
    assert len(vector_store.added) == 1
    assert embeddings.calls == 1
