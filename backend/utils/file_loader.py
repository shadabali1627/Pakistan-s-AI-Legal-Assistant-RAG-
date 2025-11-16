from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader
)

def load_all_documents(directory_path: str) -> List[Document]:
    path = Path(directory_path)
    docs: List[Document] = []

    # PDFs
    docs.extend(
        DirectoryLoader(
            path=str(path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
            max_concurrency=4,
        ).load()
    )

    # DOCX/DOC
    for pat in ("**/*.docx", "**/*.doc"):
        docs.extend(
            DirectoryLoader(
                path=str(path),
                glob=pat,
                loader_cls=UnstructuredWordDocumentLoader,
                show_progress=True,
                use_multithreading=True,
                max_concurrency=4,
            ).load()
        )

    return docs
