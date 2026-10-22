from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=15)
    include_sources: bool = True


class IngestResponse(BaseModel):
    source: str
    pages_loaded: int
    chunks_indexed: int
    status: str


class SourceItem(BaseModel):
    source_id: str | None = None
    source: str
    page: int | None = None
    section: str | None = None
    chunk_id: str | None = None
    excerpt: str
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]
    retrieval_mode: str = "hybrid_rrf"
    retrieval_score_mode: str | None = None
    citation_validation: str | None = None
    citation_audit_latency_seconds: float | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    llm_latency_seconds: float | None = None
    latency_seconds: float
