"""Ollama embedding client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import json
import os
import urllib.error
import urllib.request


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "bge-m3"


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = DEFAULT_EMBEDDING_MODEL
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    timeout_seconds: float = 120.0
    batch_size: int = 16


class OllamaEmbeddingsClient:
    """Small HTTP client for Ollama embeddings.

    Uses `/api/embed` for batches and falls back to `/api/embeddings` for older
    Ollama versions that only accept one prompt at a time.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
        batch_size: int = 16,
    ) -> None:
        self.config = EmbeddingConfig(
            model=model
            or os.getenv("EMBEDDING_MODEL")
            or os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            base_url=(base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)).rstrip("/"),
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
        )

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        embeddings: list[list[float]] = []
        for start in range(0, len(text_list), self.config.batch_size):
            batch = text_list[start : start + self.config.batch_size]
            embeddings.extend(self._embed_batch(batch))
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        embeddings = self.embed_texts([query])
        if not embeddings:
            raise RuntimeError("Ollama returned no embedding for query.")
        return embeddings[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            return self._post_embed(texts)
        except RuntimeError:
            embeddings = [self._post_legacy_embedding(text) for text in texts]
            return embeddings

    def _post_embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.config.model, "input": texts}
        data = self._post_json("/api/embed", payload)
        raw_embeddings = data.get("embeddings")
        if not isinstance(raw_embeddings, list):
            raise RuntimeError("Ollama /api/embed response did not include embeddings.")
        return [_coerce_embedding(embedding) for embedding in raw_embeddings]

    def _post_legacy_embedding(self, text: str) -> list[float]:
        payload = {"model": self.config.model, "prompt": text}
        data = self._post_json("/api/embeddings", payload)
        return _coerce_embedding(data.get("embedding"))

    def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.config.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to call Ollama embeddings endpoint at {self.config.base_url}.") from exc

        try:
            data = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON for embeddings request.") from exc
        if not isinstance(data, dict):
            raise RuntimeError("Ollama embeddings response was not a JSON object.")
        return data


def _coerce_embedding(value: object) -> list[float]:
    if not isinstance(value, list) or not value:
        raise RuntimeError("Ollama returned an empty or invalid embedding.")
    return [float(item) for item in value]
