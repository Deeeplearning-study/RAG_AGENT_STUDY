"""Token-window chunking with page range preservation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable
import re

from .pdf_loader import PdfDocument, PdfPage


DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[가-힣]+|[^\s]", re.UNICODE)


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    document_id: str
    file_name: str
    page_start: int
    page_end: int
    text: str
    file_hash: str
    chunk_index: int


@dataclass(frozen=True)
class PageToken:
    text: str
    page_number: int


class TextChunker:
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: PdfDocument) -> list[TextChunk]:
        return self.chunk_pages(
            pages=document.pages,
            document_id=document.document_id,
            file_name=document.file_name,
            file_hash=document.file_hash,
        )

    def chunk_pages(
        self,
        pages: Iterable[PdfPage],
        document_id: str,
        file_name: str,
        file_hash: str,
    ) -> list[TextChunk]:
        tokens = _page_tokens(pages)
        if not tokens:
            return []

        chunks: list[TextChunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            window = tokens[start:end]
            text = _join_tokens(token.text for token in window)
            page_start = min(token.page_number for token in window)
            page_end = max(token.page_number for token in window)
            chunk_id = _chunk_id(document_id, chunk_index, page_start, page_end, text)
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    file_name=file_name,
                    page_start=page_start,
                    page_end=page_end,
                    text=text,
                    file_hash=file_hash,
                    chunk_index=chunk_index,
                )
            )

            if end == len(tokens):
                break
            start += step
            chunk_index += 1

        return chunks


def _page_tokens(pages: Iterable[PdfPage]) -> list[PageToken]:
    tokens: list[PageToken] = []
    for page in pages:
        for match in TOKEN_PATTERN.finditer(page.text):
            tokens.append(PageToken(text=match.group(0), page_number=page.page_number))
    return tokens


def _join_tokens(tokens: Iterable[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%)\]\}])", r"\1", text)
    text = re.sub(r"([(\[\{])\s+", r"\1", text)
    return text.strip()


def _chunk_id(document_id: str, chunk_index: int, page_start: int, page_end: int, text: str) -> str:
    text_hash = sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{document_id}:{chunk_index:06d}:{page_start}-{page_end}:{text_hash}"

