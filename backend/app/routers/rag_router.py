from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.rag_schemas import AskRequest, AskResponse, IngestResponse
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])


def _service() -> RAGService:
    return RAGService()


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    try:
        result = _service().ask(request.question, request.top_k, request.include_sources)
        return AskResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {exc}") from exc


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")
        result = _service().ingest_upload(file.filename or "document.pdf", content)
        return IngestResponse(**result)
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {exc}") from exc
