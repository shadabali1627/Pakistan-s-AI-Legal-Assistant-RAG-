import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

# --- Keys & models ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash-002")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "models/text-embedding-004")

# --- Paths (absolute) ---
DATA_PATH = (BASE_DIR / os.getenv("DATA_PATH", "data/pdfs")).resolve()
DATA_PATH.mkdir(parents=True, exist_ok=True)

VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "chroma").lower().strip()
CHROMA_PATH = (BASE_DIR / os.getenv("CHROMA_PATH", "vectorstore/chroma_index")).resolve()
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

# --- IMPORTANT: single collection name everywhere ---
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_docs")

logging.info(f"DATA_PATH: {DATA_PATH}")
logging.info(f"VECTOR_STORE_BACKEND: {VECTOR_STORE_BACKEND}")
logging.info(f"CHROMA_PATH: {CHROMA_PATH}")
