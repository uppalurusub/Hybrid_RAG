"""Central LLM configuration. Agent implementations must use get_llm()."""
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_DB_PATH = str(
    (PROJECT_ROOT / "data" / "chroma").resolve()
)

load_dotenv()


def _env(*names: str, default: str | None = None) -> str | None:
    """Return the first non-empty environment variable, supporting legacy names."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    model = _env("OPENROUTER_MODEL", "LLM_MODEL", "fallback_model_1")
    api_key = _env("OPENROUTER_API_KEY", "openrouter_api_key")
    base_url = _env(
        "OPENROUTER_BASE_URL", "base_url", default="https://openrouter.ai/api/v1"
    )

    if not model:
        raise RuntimeError(
            "LLM model is not configured. Set OPENROUTER_MODEL (or legacy fallback_model_1)."
        )
    if not api_key:
        raise RuntimeError("OpenRouter API key is not configured. Set OPENROUTER_API_KEY.")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=int(_env("LLM_MAX_TOKENS", default="1200") or "1200"),
        # OpenRouter supports including usage information in the response.
        extra_body={"usage": {"include": True}},
    )


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/chroma")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "8"))
APP_NAME = os.getenv("APP_NAME", "Agentic RAG API")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


