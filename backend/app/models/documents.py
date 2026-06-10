from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ComponentStatus(BaseModel):
    ok: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    ollama: bool
    chroma: bool
    main_model: str
    sub_model: str
    embedding_model: str
    indexed_documents: int = Field(default=0, ge=0)
    components: dict[str, ComponentStatus] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    force: bool = False


class IngestResponse(BaseModel):
    status: Literal["completed", "skipped", "failed"]
    documents_total: int = Field(default=0, ge=0)
    documents_indexed: int = Field(default=0, ge=0)
    chunks_total: int = Field(default=0, ge=0)
    failures: list[dict[str, str | None]] = Field(default_factory=list)


class DocumentResponse(BaseModel):
    document_id: str
    file_name: str
    pages: int = Field(..., ge=0)
    chunks: int = Field(..., ge=0)
    indexed_at: datetime | None = None
