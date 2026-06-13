from backend.app.services.chunker import TextChunker
from backend.app.services.pdf_loader import PdfPage


def test_chunker_keeps_segments_whole_and_tracks_page_ranges():
    # Each page is a single line/segment of 6 tokens.
    pages = [
        PdfPage(page_number=1, text=" ".join(f"a{i}" for i in range(6)), char_count=12),
        PdfPage(page_number=2, text=" ".join(f"b{i}" for i in range(6)), char_count=12),
    ]

    # chunk_size=8 cannot hold both 6-token lines, and a line is never split,
    # so each line becomes its own chunk on its own page.
    chunks = TextChunker(chunk_size=8, chunk_overlap=2).chunk_pages(
        pages=pages,
        document_id="doc",
        file_name="sample.pdf",
        file_hash="hash",
    )

    assert len(chunks) == 2
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 1)
    assert (chunks[1].page_start, chunks[1].page_end) == (2, 2)
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_chunker_does_not_split_a_criterion_line():
    # A heading line and a numeric criterion line on the same page should stay
    # together in one chunk, with the criterion line kept intact.
    criterion = "수축기 140 이완기 90"
    pages = [
        PdfPage(page_number=1, text=f"고혈압 진단 기준\n{criterion}", char_count=20),
    ]

    chunks = TextChunker(chunk_size=900, chunk_overlap=150).chunk_pages(
        pages=pages,
        document_id="doc",
        file_name="guideline.pdf",
        file_hash="hash",
    )

    assert len(chunks) == 1
    assert criterion in chunks[0].text
    assert "고혈압 진단 기준" in chunks[0].text
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 1)


def test_chunker_chunk_spans_pages_when_lines_pack_together():
    pages = [
        PdfPage(page_number=1, text="첫 페이지 마지막 줄", char_count=10),
        PdfPage(page_number=2, text="둘째 페이지 첫 줄", char_count=10),
    ]

    chunks = TextChunker(chunk_size=900, chunk_overlap=150).chunk_pages(
        pages=pages,
        document_id="doc",
        file_name="sample.pdf",
        file_hash="hash",
    )

    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_chunker_hard_splits_a_single_oversized_line():
    oversized = " ".join(f"t{i}" for i in range(20))
    pages = [PdfPage(page_number=3, text=oversized, char_count=len(oversized))]

    chunks = TextChunker(chunk_size=8, chunk_overlap=2).chunk_pages(
        pages=pages,
        document_id="doc",
        file_name="long.pdf",
        file_hash="hash",
    )

    assert len(chunks) > 1
    assert all(chunk.page_start == 3 and chunk.page_end == 3 for chunk in chunks)


def test_chunker_returns_no_chunks_for_empty_pages():
    chunks = TextChunker(chunk_size=8, chunk_overlap=2).chunk_pages(
        pages=[PdfPage(page_number=1, text="", char_count=0, low_text=True)],
        document_id="doc",
        file_name="empty.pdf",
        file_hash="hash",
    )

    assert chunks == []
