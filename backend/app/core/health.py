import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from app.core.config import Settings


def get_fallback_health(settings: Settings) -> dict[str, Any]:
    ollama_ok, model_names, ollama_detail = _get_ollama_models(settings.ollama_base_url)
    indexed_documents = _count_indexed_documents(settings.processed_dir)
    chroma_ok = settings.chroma_dir.exists()

    main_model_ok = settings.main_agent_model in model_names
    sub_model_ok = settings.sub_agent_model in model_names
    embedding_model_ok = settings.embedding_model in model_names

    required_ok = [
        ollama_ok,
        chroma_ok,
        main_model_ok,
        sub_model_ok,
        embedding_model_ok,
    ]
    status = "ok" if all(required_ok) else "degraded"

    return {
        "status": status,
        "ollama": ollama_ok,
        "chroma": chroma_ok,
        "main_model": settings.main_agent_model,
        "sub_model": settings.sub_agent_model,
        "embedding_model": settings.embedding_model,
        "indexed_documents": indexed_documents,
        "components": {
            "ollama": {"ok": ollama_ok, "detail": ollama_detail},
            "chroma": {
                "ok": chroma_ok,
                "detail": f"Chroma directory: {settings.chroma_dir}",
            },
            "main_model": {
                "ok": main_model_ok,
                "detail": _model_detail(settings.main_agent_model, model_names),
            },
            "sub_model": {
                "ok": sub_model_ok,
                "detail": _model_detail(settings.sub_agent_model, model_names),
            },
            "embedding_model": {
                "ok": embedding_model_ok,
                "detail": _model_detail(settings.embedding_model, model_names),
            },
            "index": {
                "ok": indexed_documents > 0,
                "detail": f"Indexed documents: {indexed_documents}",
            },
        },
    }


def _get_ollama_models(base_url: str) -> tuple[bool, set[str], str]:
    request = urllib.request.Request(url=f"{base_url.rstrip('/')}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        return False, set(), f"Ollama unavailable at {base_url}: {exc.reason}"
    except TimeoutError:
        return False, set(), f"Ollama health check timed out at {base_url}."

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return False, set(), "Ollama returned invalid JSON from /api/tags."

    raw_models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(raw_models, list):
        return False, set(), "Ollama /api/tags response did not include models."

    model_names = {
        str(item.get("name") or item.get("model"))
        for item in raw_models
        if isinstance(item, dict) and (item.get("name") or item.get("model"))
    }
    return True, model_names, f"Discovered {len(model_names)} Ollama models."


def _count_indexed_documents(processed_dir: Path) -> int:
    if not processed_dir.exists():
        return 0
    return sum(
        1
        for path in processed_dir.glob("*.json")
        if path.name != "index.json" and path.is_file()
    )


def _model_detail(model: str, available_models: set[str]) -> str:
    if model in available_models:
        return f"Model available: {model}"
    return f"Model not found in Ollama: {model}"
