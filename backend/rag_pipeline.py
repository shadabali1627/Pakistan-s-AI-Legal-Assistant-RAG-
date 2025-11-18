from typing import Iterable, List, Tuple, Dict, Any, Iterator
import os
import google.generativeai as genai

from langchain_chroma import Chroma  # <-- UPDATED IMPORT
from langchain_core.documents import Document

from backend.config import CHROMA_PATH, GOOGLE_API_KEY, COLLECTION_NAME
from backend.utils.embedding_utils import embeddings_model

genai.configure(api_key=GOOGLE_API_KEY)

DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "gemini-1.5-flash-latest")

# --- Helper Functions (Models, DB, Formatting) ---

def _make_model(model_name: str) -> genai.GenerativeModel:
    """
    Creates a GenerativeModel instance.
    """
    name = model_name if model_name.startswith("models/") else f"models/{model_name}"
    return genai.GenerativeModel(name)

def _db() -> Chroma:
    return Chroma(
        embedding_function=embeddings_model,
        persist_directory=str(CHROMA_PATH),
        collection_name=COLLECTION_NAME,
    )

def _format_docs(docs: Iterable[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def _doc_meta(d: Document) -> Dict[str, Any]:
    meta = dict(getattr(d, "metadata", {}) or {})
    meta["source"] = meta.get("source") or meta.get("file_path") or meta.get("path") or ""
    meta["page"] = meta.get("page") or meta.get("page_number") or meta.get("page_index")
    return meta

# --- NEW: Query Correction (Handles Typos) ---

def _correct_query(query: str, model_name: str) -> str:
    """
    Corrects spelling, typos, and abbreviations in the user's query.
    """
    model = _make_model(model_name)
    
    prompt = (
        "You are a query correction assistant. Your task is to fix spelling errors, typos, and expand common abbreviations in the user's question, especially those related to Pakistani law. Respond *only* with the corrected question.\n"
        "If the query is a simple greeting, or seems correct, return it unchanged.\n\n"
        "Examples:\n"
        "User: tell me about lan revenue ac\n"
        "tell me about land revenue act\n\n"
        "User: what is sec 489f of ppc\n"
        "what is section 489f of pakistan penal code\n\n"
        "User: divrce procedure\n"
        "divorce procedure\n\n"
        "User: hi how are you\n"
        "hi how are you\n\n"
        f"User: {query}\n"
    )
    
    try:
        resp = model.generate_content(prompt)
        corrected = (getattr(resp, "text", "") or "").strip()
        if not corrected:
            return query # Return original on empty response
        return corrected
    except Exception:
        return query # Return original on any error

# --- Intent Classification ---

def _classify_intent(query: str, model_name: str) -> str:
    """
    Classifies the user's query as 'LEGAL' or 'GENERAL'.
    Uses the *corrected* query.
    """
    model = _make_model(model_name)
    
    prompt = (
        "You are an intent classifier. Your job is to determine if the user's question is about Pakistani law or a general question/greeting.\n"
        "Respond with only a single word: 'LEGAL' or 'GENERAL'.\n\n"
        "Examples:\n"
        "User: What is the procedure for divorce?\n"
        "LEGAL\n"
        "User: Tell me about Section 489F.\n"
        "LEGAL\n"
        "User: Hi\n"
        "GENERAL\n"
        "User: How are you?\n"
        "GENERAL\n"
        "User: What is the capital of France?\n"
        "GENERAL\n\n"
        f"User: {query}\n"
    )
    
    try:
        resp = model.generate_content(prompt)
        text_resp = (getattr(resp, "text", "") or "").strip().upper()
        if text_resp == "LEGAL":
            return "LEGAL"
        elif text_resp == "GENERAL":
            return "GENERAL"
        else:
            return "LEGAL" # Default to LEGAL
    except Exception:
        return "LEGAL" # Default to LEGAL on error

# --- General Chat Function ---

def _general_chat(query: str, model_name: str) -> Dict[str, Any]:
    """
    Handles general conversation by answering with the LLM directly.
    Uses the *corrected* query.
    """
    model = _make_model(model_name)
    prompt = (
        "You are a helpful and polite AI assistant. You are an expert in Pakistani law, but you can also answer general questions and engage in friendly conversation.\n"
        "Answer the user's question concisely and politely.\n\n"
        f"USER: {query}\n"
        "ASSISTANT:\n"
    )
    
    try:
        resp = model.generate_content(prompt)
        answer = (getattr(resp, "text", "") or "").strip() or "I'm not sure how to respond to that."
        return {"answer": answer, "citations": []}
    except Exception as e:
        return {"answer": f"An error occurred while processing your request: {e}", "citations": []}

# --- NEW: General Chat Function (Streaming) ---

def _general_chat_stream(query: str, model_name: str) -> Iterator[str]:
    """
    Handles general conversation by streaming the LLM response.
    Yields answer tokens as strings.
    """
    model = _make_model(model_name)
    prompt = (
        "You are a helpful and polite AI assistant. You are an expert in Pakistani law, but you can also answer general questions and engage in friendly conversation.\n"
        "Answer the user's question concisely and politely.\n\n"
        f"USER: {query}\n"
        "ASSISTANT:\n"
    )
    
    try:
        resp_stream = model.generate_content(prompt, stream=True)
        for chunk in resp_stream:
            yield (getattr(chunk, "text", "") or "")
    except Exception as e:
        yield f"An error occurred while processing your request: {e}"


# --- RAG Pipeline Functions ---

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
        search_kwargs={"k": k, "fetch_k": max(3 * k, 20), "lambda_mult": 0.6},
    )
    return retriever.invoke(query)

def _rag_query(query: str, k: int, model_name: str) -> Dict[str, Any]:
    """
    The original RAG (Retrieval-Augmented Generation) pipeline.
    Uses the *corrected* query.
    """
    # Use MMR (diversity search) first
    filtered_docs = retrieve_mmr(query, k=max(8, k))
    
    # Fallback to simple similarity search
    if not filtered_docs:
        scored = retrieve_with_scores(query, k=k)
        filtered_docs = [d for d, s in scored if 0.0 <= s <= 1.0 and s >= 0.20] or [d for d, _ in scored]

    if not filtered_docs:
        return {"answer": "I cannot find any documents related to that topic.", "citations": []}

    context = _format_docs(filtered_docs)
    
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


# --- NEW: RAG Query (Streaming) ---

def _rag_query_stream(query: str, k: int, model_name: str) -> Iterator[Dict[str, Any]]:
    """
    RAG pipeline that streams the result.
    Yields dicts: `{"citations": [...]}` first, then `{"token": "..."}`.
    """
    # 1. Retrieval (Blocking)
    filtered_docs = retrieve_mmr(query, k=max(8, k))
    if not filtered_docs:
        scored = retrieve_with_scores(query, k=k)
        filtered_docs = [d for d, s in scored if 0.0 <= s <= 1.0 and s >= 0.20] or [d for d, _ in scored]

    if not filtered_docs:
        yield {"citations": []}
        yield {"token": "I cannot find any documents related to that topic."}
        return

    # 2. Yield citations *first*
    citations = [{"source": _doc_meta(d).get("source"), "page": _doc_meta(d).get("page")} for d in filtered_docs[:5]]
    yield {"citations": citations}

    # 3. Stream the answer
    context = _format_docs(filtered_docs)
    prompt = (
        "You are an AI Legal Assistant for Pakistan law. Answer the user's question using only the provided context.\n"
        "Your answer must be based *solely* on the text in the context. Do not use outside knowledge.\n"
        "If the context contains the answer, extract it and present it clearly. Be concise.\n"
        "If the answer is not explicitly stated in the context, reply *exactly* with the following text:\n"
        "I cannot find the answer in the provided documents.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:\n"
    )

    model = _make_model(model_name)
    try:
        resp_stream = model.generate_content(prompt, stream=True)
        for chunk in resp_stream:
            yield {"token": (getattr(chunk, "text", "") or "")}
    except Exception as e:
        yield {"token": f"An error occurred during streaming: {e}"}


# --- Main Entry Point ---

def answer_query(query: str, k: int = 6, model_name: str = DEFAULT_CHAT_MODEL) -> Dict[str, Any]:
    """
    Main router function. Corrects query, classifies intent, and calls the correct pipeline.
    """
    # --- NEW STEP 1: Correct the query first ---
    corrected_query = _correct_query(query, model_name)
    
    # Step 2: Classify the user's intent *using the corrected query*
    intent = _classify_intent(corrected_query, model_name)
    
    # Step 3: Route to the correct handler *using the corrected query*
    if intent == "GENERAL":
        return _general_chat(corrected_query, model_name)
    else:
        # Default to LEGAL (RAG)
        return _rag_query(corrected_query, k, model_name)

# --- NEW: Main Entry Point (Streaming) ---

def answer_query_stream(query: str, k: int = 6, model_name: str = DEFAULT_CHAT_MODEL) -> Iterator[Dict[str, Any]]:
    """
    Main streaming router. Corrects, classifies, then streams from the correct pipeline.
    Yields dicts: `{"citations": [...]}` first, then `{"token": "..."}`.
    """
    # Step 1: Correct the query (Blocking)
    corrected_query = _correct_query(query, model_name)
    
    # Step 2: Classify intent (Blocking)
    intent = _classify_intent(corrected_query, model_name)
    
    # Step 3: Route to the correct streaming handler
    if intent == "GENERAL":
        # General chat has no citations, so send empty list first
        yield {"citations": []}
        # Yield tokens
        for token in _general_chat_stream(corrected_query, model_name):
            yield {"token": token}
    else:
        # RAG stream handles yielding citations first, then tokens
        for event in _rag_query_stream(corrected_query, k, model_name):
            yield event

# --- Debug/Diagnostics Functions ---

def search_docs(query: str, k: int = 8):
    # This debug function will still use the *original* query
    # To test correction, use the main /chat endpoint
    results = retrieve_with_scores(query, k=k)
    out = []
    for i, (d, s) in enumerate(results, 1):
        preview = (d.page_content or "").replace("\n", " ")
        if len(preview) > 300:
            preview = preview[:300] + "..."
        out.append({"rank": i, "score": float(s), "metadata": _doc_meta(d), "preview": preview})
    return out

def direct_model_test(model_name: str = DEFAULT_CHAT_MODEL) -> str:
    # Use the simplified _make_model function
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