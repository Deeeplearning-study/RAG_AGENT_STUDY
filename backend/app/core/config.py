import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import AnyHttpUrl, BaseModel, Field, field_validator

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv() -> bool:
        return False


class Settings(BaseModel):
    app_name: str = "PDF RAG Agent API"
    app_version: str = "0.1.0"
    environment: str = "development"

    pdf_dir: Path = Path("../pdf")
    chroma_dir: Path = Path("./data/chroma")
    processed_dir: Path = Path("./data/processed")

    ollama_base_url: str = "http://localhost:11434"
    main_agent_model: str = "gemma4:26b"
    sub_agent_model: str = "gemma4:e4b"
    embedding_model: str = "bge-m3"

    default_top_k: Annotated[int, Field(ge=1, le=50)] = 8
    enable_agent_trace: bool = True

    min_rerank_score: float = 0.0

    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "auto"
    reranker_batch_size: Annotated[int, Field(ge=1, le=128)] = 8
    reranker_max_length: Annotated[int, Field(ge=64, le=4096)] = 512
    reranker_candidate_multiplier: Annotated[int, Field(ge=1, le=10)] = 3
    reranker_max_candidates: Annotated[int, Field(ge=1, le=200)] = 40
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:6000",
            "http://127.0.0.1:6000",
        ]
    )

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, value: str) -> str:
        AnyHttpUrl(value)
        return value.rstrip("/")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_cors_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


def backend_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _backend_relative_path(value: str, default: str) -> Path:
    path = Path(os.getenv(value, default))
    if path.is_absolute():
        return path
    return backend_dir() / path


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    values: dict[str, object] = {
        "app_name": os.getenv("APP_NAME", "PDF RAG Agent API"),
        "app_version": os.getenv("APP_VERSION", "0.1.0"),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "pdf_dir": _backend_relative_path("PDF_DIR", "../pdf"),
        "chroma_dir": _backend_relative_path("CHROMA_DIR", "./data/chroma"),
        "processed_dir": _backend_relative_path("PROCESSED_DIR", "./data/processed"),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "main_agent_model": os.getenv("MAIN_AGENT_MODEL", "gemma4:26b"),
        "sub_agent_model": os.getenv("SUB_AGENT_MODEL", "gemma4:e4b"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "bge-m3"),
        "default_top_k": int(os.getenv("DEFAULT_TOP_K", "8")),
        "enable_agent_trace": _get_bool("ENABLE_AGENT_TRACE", True),
        "min_rerank_score": float(os.getenv("RAG_MIN_RERANK_SCORE", "0.0")),
        "reranker_enabled": _get_bool("RERANKER_ENABLED", True),
        "reranker_model": os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        "reranker_device": os.getenv("RERANKER_DEVICE", "auto"),
        "reranker_batch_size": int(os.getenv("RERANKER_BATCH_SIZE", "8")),
        "reranker_max_length": int(os.getenv("RERANKER_MAX_LENGTH", "512")),
        "reranker_candidate_multiplier": int(os.getenv("RERANKER_CANDIDATE_MULTIPLIER", "3")),
        "reranker_max_candidates": int(os.getenv("RERANKER_MAX_CANDIDATES", "40")),
    }
    cors_origins = os.getenv("CORS_ORIGINS")
    if cors_origins is not None:
        values["cors_origins"] = cors_origins
    return Settings(**values)
