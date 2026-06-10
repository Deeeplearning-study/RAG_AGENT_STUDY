"""PDF discovery, hashing, and page-level text extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any


DEFAULT_LOW_TEXT_CHAR_THRESHOLD = 40


@dataclass(frozen=True)
class PdfPage:
    page_number: int
    text: str
    char_count: int
    low_text: bool = False


@dataclass(frozen=True)
class PdfDocument:
    document_id: str
    file_name: str
    file_path: Path
    file_hash: str
    title: str | None
    page_count: int
    pages: list[PdfPage]
    metadata: dict[str, Any] = field(default_factory=dict)
    low_text_pages: list[int] = field(default_factory=list)


def default_pdf_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "pdf"


def discover_pdfs(pdf_dir: Path | str | None = None) -> list[Path]:
    """Return all PDFs below the configured directory in stable order."""

    base_dir = Path(pdf_dir) if pdf_dir is not None else default_pdf_dir()
    if not base_dir.exists():
        return []
    return sorted(path for path in base_dir.rglob("*.pdf") if path.is_file())


def compute_file_hash(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def document_id_for_path(path: Path | str) -> str:
    """Create a stable document id from the file name/path, not file content."""

    resolved = Path(path)
    key = resolved.name.encode("utf-8", errors="ignore")
    return sha256(key).hexdigest()[:24]


class PdfLoader:
    def __init__(
        self,
        pdf_dir: Path | str | None = None,
        low_text_char_threshold: int = DEFAULT_LOW_TEXT_CHAR_THRESHOLD,
    ) -> None:
        self.pdf_dir = Path(pdf_dir) if pdf_dir is not None else default_pdf_dir()
        self.low_text_char_threshold = low_text_char_threshold

    def discover(self) -> list[Path]:
        return discover_pdfs(self.pdf_dir)

    def load(self, path: Path | str) -> PdfDocument:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required for PDF extraction. Install package 'pymupdf'.") from exc

        pdf_path = Path(path)
        file_hash = compute_file_hash(pdf_path)
        document_id = document_id_for_path(pdf_path)

        pages: list[PdfPage] = []
        with fitz.open(pdf_path) as pdf:
            raw_metadata = dict(pdf.metadata or {})
            title = _clean_optional_text(raw_metadata.get("title"))

            for page_index in range(pdf.page_count):
                page = pdf.load_page(page_index)
                text = _normalize_text(page.get_text("text"))
                char_count = len(text.strip())
                pages.append(
                    PdfPage(
                        page_number=page_index + 1,
                        text=text,
                        char_count=char_count,
                        low_text=char_count < self.low_text_char_threshold,
                    )
                )

            low_text_pages = [page.page_number for page in pages if page.low_text]
            return PdfDocument(
                document_id=document_id,
                file_name=pdf_path.name,
                file_path=pdf_path,
                file_hash=file_hash,
                title=title,
                page_count=pdf.page_count,
                pages=pages,
                metadata=raw_metadata,
                low_text_pages=low_text_pages,
            )


def _clean_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\x00", " ").splitlines()]
    return "\n".join(line for line in lines if line)

