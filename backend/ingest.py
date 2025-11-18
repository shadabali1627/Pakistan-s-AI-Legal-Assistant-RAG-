# backend/ingest.py
from __future__ import annotations
import sys
from pathlib import Path
from typing import List

from langchain_chroma import Chroma  # <-- UPDATED IMPORT
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)

from backend.utils.embedding_utils import embeddings_model
from backend.utils.text_processing import split_documents
from backend.config import DATA_PATH, CHROMA_PATH, COLLECTION_NAME


def _iter_files(root: Path) -> List[Path]:
    # Only the types our loader supports (pdf, docx, doc).
    pats = ["**/*.pdf", "**/*.docx", "**/*.doc"]
    files: List[Path] = []
    for pat in pats:
        files.extend(root.glob(pat))
    # Sort for deterministic processing (helps resume on failure)
    return sorted({f.resolve() for f in files})

def _load_one(path: Path) -> List[Document]:
    # Load exactly one file with the same loaders used in file_loader.py.
    if path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(path)).load()
    elif path.suffix.lower() in (".docx", ".doc"):
        return UnstructuredWordDocumentLoader(str(path)).load()
    else:
        return []

def main():
    print(f"[ingest] DATA_PATH={DATA_PATH}")
    print(f"[ingest] CHROMA_PATH={CHROMA_PATH}")
    print(f"[ingest] COLLECTION_NAME={COLLECTION_NAME}")

    root = Path(DATA_PATH)
    root.mkdir(parents=True, exist_ok=True)

    files = _iter_files(root)
    if not files:
        print("[ingest] No files found (pdf/docx/doc). Nothing to do.")
        return

    # Open (or create) the Chroma collection up front. We will add per-file.
    db = Chroma(
        embedding_function=embeddings_model,
        persist_directory=str(CHROMA_PATH),
        collection_name=COLLECTION_NAME,
    )

    ok = 0
    fail = 0
    skipped_empty = 0

    for idx, f in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] Loading: {f.name}")
        try:
            docs = _load_one(f)
            if not docs:
                print(f"   └─ [skip] unsupported or unreadable: {f.name}")
                skipped_empty += 1
                continue

            chunks = split_documents(docs)   # 1000/200 default chunking
            if not chunks:
                print(f"   └─ [skip] produced 0 chunks: {f.name}")
                skipped_empty += 1
                continue

            # Commit this file now — so a later failure won’t wipe progress
            db.add_documents(chunks)
            db.persist()
            ok += 1
            print(f"   └─ [OK] added {len(chunks)} chunks")
        except Exception as e:
            fail += 1
            # Don’t crash the whole run — just report the file that failed
            print(f"   └─ [FAIL] {f.name} -> {e.__class__.__name__}: {e}")

    # Final stats + collection count
    try:
        count = int(db._collection.count())
    except Exception:
        count = -1

    print("\n[ingest] Done.")
    print(f"         OK={ok}  FAIL={fail}  SKIPPED_EMPTY={skipped_empty}  TOTAL_FILES={len(files)}")
    print(f"         Collection '{COLLECTION_NAME}' count now: {count}")
    print(f"         Persisted at: {CHROMA_PATH}")

if __name__ == "__main__":
    sys.exit(main())