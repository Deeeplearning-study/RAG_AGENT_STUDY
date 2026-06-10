from backend.app.services.chunker import TextChunker
from backend.app.services.pdf_loader import PdfPage


def test_chunker_preserves_page_ranges_across_overlap():
    pages = [
        PdfPage(page_number=1, text=" ".join(f"a{i}" for i in range(6)), char_count=12),
        PdfPage(page_number=2, text=" ".join(f"b{i}" for i in range(6)), char_count=12),
    ]

    chunks = TextChunker(chunk_size=8, chunk_overlap=2).chunk_pages(
        pages=pages,
        document_id="doc",
        file_name="sample.pdf",
        file_hash="hash",
    )

    assert len(chunks) == 2
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    assert chunks[1].page_start == 2
    assert chunks[1].page_end == 2
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_chunker_returns_no_chunks_for_empty_pages():
    chunks = TextChunker(chunk_size=8, chunk_overlap=2).chunk_pages(
        pages=[PdfPage(page_number=1, text="", char_count=0, low_text=True)],
        document_id="doc",
        file_name="empty.pdf",
        file_hash="hash",
    )

    assert chunks == []

