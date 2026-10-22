from pathlib import Path
from typing import Any

from app.core.settings import DOCUMENTS_DIR
from app.implementations.hybrid_rag_impl import get_hybrid_rag


class RAGService:
    """Business orchestration layer: validation, use-case coordination, and errors."""

    def __init__(self) -> None:
        self.impl = get_hybrid_rag()

    def ask(self, question: str, top_k: int, include_sources: bool) -> dict[str, Any]:
        return self.impl.ask(question=question.strip(), top_k=top_k, include_sources=include_sources)

    def ingest_path(self, path: str | Path) -> dict[str, Any]:
        return self.impl.ingest_file(path)

    def ingest_upload(self, filename: str, content: bytes) -> dict[str, Any]:
        safe_name = Path(filename).name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in {".pdf", ".txt", ".md"}:
            raise ValueError("Only PDF, TXT, and MD files are supported")
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        destination = DOCUMENTS_DIR / safe_name
        destination.write_bytes(content)
        return self.impl.ingest_file(destination)
