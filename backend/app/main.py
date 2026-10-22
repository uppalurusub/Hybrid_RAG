import logging

from fastapi import FastAPI

from app.routers.rag_router import router as rag_router
from app.core.llm_config import APP_NAME, LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
app = FastAPI(title=APP_NAME, version="1.0.0", description="FastAPI Hybrid RAG using Chroma + BM25 + RRF")
app.include_router(rag_router)


@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    return {"status": "ok"}
