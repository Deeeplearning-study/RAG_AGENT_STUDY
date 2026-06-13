"""Boundary-aware chunking with page range preservation.

Chunks are packed from line-level segments instead of a blind token window, so
a chunk boundary never falls in the middle of a line. This keeps clinical
guideline material intact - e.g. a "수축기 140 / 이완기 90" criterion line stays
in one chunk, and a section heading stays attached to the lines that follow it.
An oversized single line (longer than ``chunk_size``) falls back to a hard token
window so it is never dropped.
"""

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
class Segment:
    text: str
    page_number: int
    token_count: int


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
        segments = _page_segments(pages)
        if not segments:
            return []

        emitted: list[tuple[str, int, int]] = []
        current: list[Segment] = []
        current_tokens = 0
        index = 0

        while index < len(segments):
            segment = segments[index]

            if segment.token_count > self.chunk_size:
                if current:
                    emitted.append(_pack(current))
                    current, current_tokens = [], 0
                emitted.extend(self._split_oversized_segment(segment))
                index += 1
                continue

            if current and current_tokens + segment.token_count > self.chunk_size:
                emitted.append(_pack(current))
                carry = self._overlap_tail(current)
                if carry and _segments_tokens(carry) + segment.token_count > self.chunk_size:
                    carry = []
                current = carry
                current_tokens = _segments_tokens(carry)
                continue

            current.append(segment)
            current_tokens += segment.token_count
            index += 1

        if current:
            emitted.append(_pack(current))

        return self._to_chunks(emitted, document_id, file_name, file_hash)

    def _overlap_tail(self, segments: list[Segment]) -> list[Segment]:
        if self.chunk_overlap <= 0:
            return []
        tail: list[Segment] = []
        tokens = 0
        for segment in reversed(segments):
            if tokens + segment.token_count > self.chunk_overlap:
                break
            tail.insert(0, segment)
            tokens += segment.token_count
        return tail

    def _split_oversized_segment(self, segment: Segment) -> list[tuple[str, int, int]]:
        tokens = TOKEN_PATTERN.findall(segment.text)
        step = self.chunk_size - self.chunk_overlap
        pieces: list[tuple[str, int, int]] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            text = _join_tokens(tokens[start:end])
            if text:
                pieces.append((text, segment.page_number, segment.page_number))
            if end == len(tokens):
                break
            start += step
        return pieces

    def _to_chunks(
        self,
        emitted: list[tuple[str, int, int]],
        document_id: str,
        file_name: str,
        file_hash: str,
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for chunk_index, (text, page_start, page_end) in enumerate(emitted):
            chunks.append(
                TextChunk(
                    chunk_id=_chunk_id(document_id, chunk_index, page_start, page_end, text),
                    document_id=document_id,
                    file_name=file_name,
                    page_start=page_start,
                    page_end=page_end,
                    text=text,
                    file_hash=file_hash,
                    chunk_index=chunk_index,
                )
            )
        return chunks


def _page_segments(pages: Iterable[PdfPage]) -> list[Segment]:
    segments: list[Segment] = []
    for page in pages:
        for line in page.text.split("\n"):
            line = line.strip()
            if not line:
                continue
            token_count = len(TOKEN_PATTERN.findall(line))
            if token_count == 0:
                continue
            segments.append(Segment(text=line, page_number=page.page_number, token_count=token_count))
    return segments


def _pack(segments: list[Segment]) -> tuple[str, int, int]:
    text = _normalize_spaces(" ".join(segment.text for segment in segments))
    page_start = min(segment.page_number for segment in segments)
    page_end = max(segment.page_number for segment in segments)
    return text, page_start, page_end


def _segments_tokens(segments: list[Segment]) -> int:
    return sum(segment.token_count for segment in segments)


def _join_tokens(tokens: Iterable[str]) -> str:
    return _normalize_spaces(" ".join(tokens))


def _normalize_spaces(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?%)\]\}])", r"\1", text)
    text = re.sub(r"([(\[\{])\s+", r"\1", text)
    return text.strip()


def _chunk_id(document_id: str, chunk_index: int, page_start: int, page_end: int, text: str) -> str:
    text_hash = sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{document_id}:{chunk_index:06d}:{page_start}-{page_end}:{text_hash}"
