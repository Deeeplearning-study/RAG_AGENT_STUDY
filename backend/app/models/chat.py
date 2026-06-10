from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)


class ChatSource(BaseModel):
    file_name: str
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)
    snippet: str


class AgentTrace(BaseModel):
    query_variants: list[str] = Field(default_factory=list)
    selected_chunk_count: int = Field(default=0, ge=0)


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource] = Field(default_factory=list)
    agent_trace: AgentTrace | None = None


class SourceChunk(BaseModel):
    chunk_id: str
    file_name: str
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)
    text: str
    score: float

