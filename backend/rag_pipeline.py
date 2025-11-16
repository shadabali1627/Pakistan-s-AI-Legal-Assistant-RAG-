from typing import Iterable, List, Tuple, Dict, Any
import os
import google.generativeai as genai

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from backend.config import CHROMA_PATH, GOOGLE_API_KEY, COLLECTION_NAME
from backend.utils.embedding_utils import embeddings_model

genai.configure(api_key=GOOGLE_API_KEY)

DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "gemini-1.5-flash-002")

def _api_version_for_model(model_name: str) -> str:
    n = (model_name or "").lower()
    if n.startswith("models/"):
        n = n[len("models/"):]
    return "v1alpha" if n.startswith("gemini-2") else "v1"

def _make_model(model_name: str) -> genai.GenerativeModel:
    api_ver = _api_version_for_model(model_name)
    if hasattr(genai, "Client"):
        client = genai.Client(api_key=GOOGLE_API_KEY, http_options={"api_version": api_ver})
        return genai.GenerativeModel(model_name, client=client)
    name = model_name if model_name.startswith("models/") else f"models/{model_name}"
    return genai.GenerativeModel(name)

def _db() -> Chroma:
    return Chroma(
        embedding_function=embeddings_model,
        persist_directory=str(CHROMA_PATH),
        collection_name=COLLECTION_NAME,       # <-- pinned
    )

def _format_docs(docs: Iterable[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def _doc_meta(d: Document) -> Dict[str, Any]:
    meta = dict(getattr(d, "metadata", {}) or {})
    meta["source"] = meta.get("source") or meta.get("file_path") or meta.get("path") or ""
    meta["page"] = meta.get("page") or meta.get("page_number") or meta.get("page_index")
    return meta

def retrieve_with_scores(query: str, k: int = 6) -> List[Tuple[Document, float]]:
    db = _db()
    try:
        return db.similarity_search_with_relevance_scores(query, k=k)
    except Exception:
        return [(d, 0.0) for d in db.similarity_search(query, k=k)]

def retrieve_mmr(query: str, k: int = 8) -> List[Document]:
    db = _db()
    retriever = db.as_retriever(
        search_type="mmr",
        # Tuned parameters: fetch more (20) to find diverse results
        search_kwargs={"k": k, "fetch_k": max(3 * k, 20), "lambda_mult": 0.6},
    )
    return retriever.invoke(query)

def search_docs(query: str, k: int = 8):
    results = retrieve_with_scores(query, k=k)
    out = []
    for i, (d, s) in enumerate(results, 1):
        preview = (d.page_content or "").replace("\n", " ")
        if len(preview) > 300:
            preview = preview[:300] + "..."
        out.append({"rank": i, "score": float(s), "metadata": _doc_meta(d), "preview": preview})
    return out

def answer_query(query: str, k: int = 6, model_name: str = DEFAULT_CHAT_MODEL) -> Dict[str, Any]:
    
    # --- FIX 1: Use MMR (diversity search) first ---
    # This is better for broad queries like "Tell me about Family Law"
    filtered_docs = retrieve_mmr(query, k=max(8, k))
    
    # Fallback to simple similarity search if MMR finds nothing
    if not filtered_docs:
        scored = retrieve_with_scores(query, k=k)
        filtered_docs = [d for d, s in scored if 0.0 <= s <= 1.0 and s >= 0.20] or [d for d, _ in scored]

    if not filtered_docs:
        return {"answer": "I cannot find any documents related to that topic.", "citations": []}

    context = _format_docs(filtered_docs)
    
    # --- FIX 2: A better, more flexible prompt ---
    prompt = (
        "You are an AI Legal Assistant for Pakistan law. Answer the user's question using only the provided context.\n"
        "Your answer must be based *solely* on the text in the context. Do not use outside knowledge.\n"
        "If the context contains the answer, extract it and present it clearly. Be concise.\n"
        "If the answer is not explicitly stated in the context, reply *exactly* with the following text:\n"
        "I cannot find the answer in the provided documents.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:\n"
    )

    model = _make_model(model_name)
    resp = model.generate_content(prompt)
    answer = (getattr(resp, "text", "") or "").strip() or "I cannot find the answer in the provided documents."
    citations = [{"source": _doc_meta(d).get("source"), "page": _doc_meta(d).get("page")} for d in filtered_docs[:5]]
    return {"answer": answer, "citations": citations}

def direct_model_test(model_name: str = DEFAULT_CHAT_MODEL) -> str:
    return (getattr(_make_model(model_name).generate_content("Reply with a single word: pong"), "text", "") or "").strip()

def db_info():
    db = _db()
    try:
        return {
            "count": int(db._collection.count()),
            "name": COLLECTION_NAME,
        }
    except Exception as e:
        return {"error": str(e), "type": e.__class__.__name__}