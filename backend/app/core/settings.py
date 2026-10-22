from pathlib import Path

from app.core import llm_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_DB_PATH = Path(llm_config.VECTOR_DB_PATH)
COLLECTION_NAME = llm_config.COLLECTION_NAME
EMBEDDING_MODEL = llm_config.EMBEDDING_MODEL
CHUNK_SIZE = llm_config.CHUNK_SIZE
CHUNK_OVERLAP = llm_config.CHUNK_OVERLAP
TOP_K = llm_config.TOP_K
RRF_K = 60
MAX_CONTEXT_CHARS = 18000
