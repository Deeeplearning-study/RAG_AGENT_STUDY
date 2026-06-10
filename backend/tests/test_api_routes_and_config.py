from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core import service_loader
from app.core.config import Settings, backend_dir, get_settings
from app.core.errors import ServiceNotWiredError
from app.main import create_app


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "pdf_dir": tmp_path / "pdf",
        "chroma_dir": tmp_path / "chroma",
        "processed_dir": tmp_path / "processed",
        "default_top_k": 4,
        "enable_agent_trace": True,
        "main_agent_model": "main-test",
        "sub_agent_model": "sub-test",
        "embedding_model": "embed-test",
    }
    values.update(overrides)
    return Settings(**values)


def make_client(settings: Settings) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_config_loads_env_paths_and_simple_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    absolute_pdf_dir = tmp_path / "pdfs"
    monkeypatch.setenv("PDF_DIR", str(absolute_pdf_dir))
    monkeypatch.setenv("CHROMA_DIR", "custom/chroma")
    monkeypatch.setenv("PROCESSED_DIR", "custom/processed")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:6000, http://127.0.0.1:6000")
    monkeypatch.setenv("DEFAULT_TOP_K", "13")
    monkeypatch.setenv("ENABLE_AGENT_TRACE", "off")
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    monkeypatch.setenv("RERANKER_MODEL", "test-reranker")
    monkeypatch.setenv("RERANKER_DEVICE", "cpu")
    monkeypatch.setenv("RERANKER_BATCH_SIZE", "4")
    monkeypatch.setenv("RERANKER_MAX_LENGTH", "384")
    monkeypatch.setenv("RERANKER_CANDIDATE_MULTIPLIER", "2")
    monkeypatch.setenv("RERANKER_MAX_CANDIDATES", "12")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.pdf_dir == absolute_pdf_dir
    assert settings.chroma_dir == backend_dir() / "custom/chroma"
    assert settings.processed_dir == backend_dir() / "custom/processed"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.cors_origins == ["http://localhost:6000", "http://127.0.0.1:6000"]
    assert settings.default_top_k == 13
    assert settings.enable_agent_trace is False
    assert settings.reranker_enabled is False
    assert settings.reranker_model == "test-reranker"
    assert settings.reranker_device == "cpu"
    assert settings.reranker_batch_size == 4
    assert settings.reranker_max_length == 384
    assert settings.reranker_candidate_multiplier == 2
    assert settings.reranker_max_candidates == 12

    get_settings.cache_clear()


def test_health_route_returns_expected_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_call_service(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["settings"].main_agent_model == "main-test"
        return {
            "status": "degraded",
            "ollama": False,
            "chroma": True,
            "main_model": "main-test",
            "sub_model": "sub-test",
            "embedding_model": "embed-test",
            "indexed_documents": 2,
            "components": {
                "ollama": {"ok": False, "detail": "not running"},
                "chroma": {"ok": True},
            },
        }

    from app.api import routes_health

    monkeypatch.setattr(routes_health, "call_service", fake_call_service)
    client = make_client(make_settings(tmp_path))

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "ollama": False,
        "chroma": True,
        "main_model": "main-test",
        "sub_model": "sub-test",
        "embedding_model": "embed-test",
        "indexed_documents": 2,
        "components": {
            "ollama": {"ok": False, "detail": "not running"},
            "chroma": {"ok": True, "detail": None},
        },
    }


def test_health_route_uses_fallback_when_service_is_not_wired(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_call_service(**_: Any) -> None:
        raise ServiceNotWiredError(["app.services.health.get_health"])

    from app.api import routes_health

    settings = make_settings(tmp_path)
    monkeypatch.setattr(routes_health, "call_service", fake_call_service)
    client = make_client(settings)

    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["main_model"] == "main-test"
    assert payload["sub_model"] == "sub-test"
    assert payload["embedding_model"] == "embed-test"
    assert payload["indexed_documents"] == 0


def test_chat_route_uses_default_top_k_and_returns_response_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    async def fake_call_service(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "answer": "문서 근거에 따른 답변입니다.",
            "sources": [
                {
                    "file_name": "guideline.pdf",
                    "page_start": 2,
                    "page_end": 3,
                    "snippet": "근거 문장",
                }
            ],
            "agent_trace": {
                "query_variants": ["질문", "검색어"],
                "selected_chunk_count": 1,
                "ignored_extra": "not in schema",
            },
        }

    from app.api import routes_chat

    monkeypatch.setattr(routes_chat, "call_service", fake_call_service)
    client = make_client(make_settings(tmp_path, default_top_k=7))

    response = client.post("/api/chat", json={"message": "질문"})

    assert response.status_code == 200
    assert captured["message"] == "질문"
    assert captured["top_k"] == 7
    assert captured["include_agent_trace"] is True
    assert response.json() == {
        "answer": "문서 근거에 따른 답변입니다.",
        "sources": [
            {
                "file_name": "guideline.pdf",
                "page_start": 2,
                "page_end": 3,
                "snippet": "근거 문장",
            }
        ],
        "agent_trace": {
            "query_variants": ["질문", "검색어"],
            "selected_chunk_count": 1,
        },
    }


def test_chat_route_hides_trace_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    async def fake_call_service(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "answer": "trace disabled",
            "sources": [],
            "agent_trace": {"query_variants": ["x"], "selected_chunk_count": 0},
        }

    from app.api import routes_chat

    monkeypatch.setattr(routes_chat, "call_service", fake_call_service)
    client = make_client(make_settings(tmp_path, enable_agent_trace=False))

    response = client.post("/api/chat", json={"message": "질문", "top_k": 3})

    assert response.status_code == 200
    assert captured["top_k"] == 3
    assert captured["include_agent_trace"] is False
    assert response.json()["agent_trace"] is None


def test_chat_route_maps_empty_index_to_api_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class EmptyIndexError(Exception):
        pass

    def fake_service(**_: Any) -> None:
        raise EmptyIndexError("No indexed documents are available.")

    monkeypatch.setattr(service_loader, "_resolve_service", lambda candidates: fake_service)
    client = make_client(make_settings(tmp_path))

    response = client.post("/api/chat", json={"message": "인덱스가 비었나요?"})

    assert response.status_code == 409
    assert response.json() == {
        "error": {
            "code": "empty_index",
            "message": "No indexed documents are available.",
        }
    }


def test_chat_request_validation_uses_error_envelope(tmp_path: Path) -> None:
    client = make_client(make_settings(tmp_path))

    response = client.post("/api/chat", json={"message": "", "top_k": 51})

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["message"] == "Request validation failed."
    assert len(payload["error"]["details"]) >= 2


def test_ingest_route_normalizes_service_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    async def fake_call_service(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "processed": [{"file_name": "a.pdf", "chunk_count": 2}],
            "skipped": [{"file_name": "b.pdf", "chunk_count": 5}],
            "failed": [],
            "total_pdfs": 2,
        }

    from app.api import routes_ingest

    settings = make_settings(tmp_path)
    monkeypatch.setattr(routes_ingest, "call_service", fake_call_service)
    client = make_client(settings)

    response = client.post("/api/ingest", json={"force": True})

    assert response.status_code == 200
    assert captured["force"] is True
    assert captured["pdf_dir"] == settings.pdf_dir
    assert captured["chroma_dir"] == settings.chroma_dir
    assert captured["processed_dir"] == settings.processed_dir
    assert response.json() == {
        "status": "completed",
        "documents_total": 2,
        "documents_indexed": 1,
        "chunks_total": 7,
        "failures": [],
    }


def test_documents_route_normalizes_document_aliases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_call_service(**_: Any) -> dict[str, Any]:
        return {
            "documents": [
                {
                    "document_id": "doc-1",
                    "file_name": "a.pdf",
                    "page_count": 12,
                    "chunk_count": 4,
                    "indexed_at": "2026-06-08T00:00:00Z",
                }
            ]
        }

    from app.api import routes_documents

    monkeypatch.setattr(routes_documents, "call_service", fake_call_service)
    client = make_client(make_settings(tmp_path))

    response = client.get("/api/documents")

    assert response.status_code == 200
    assert response.json() == [
        {
            "document_id": "doc-1",
            "file_name": "a.pdf",
            "pages": 12,
            "chunks": 4,
            "indexed_at": "2026-06-08T00:00:00Z",
        }
    ]
