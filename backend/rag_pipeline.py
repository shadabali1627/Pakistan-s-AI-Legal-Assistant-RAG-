from typing import Iterable, List, Tuple, Dict, Any, Iterator, Optional
import os
from google import genai
from google.genai import types

from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.config import CHROMA_PATH, GOOGLE_API_KEY, COLLECTION_NAME
from backend.utils.embedding_utils import embeddings_model
from backend.schemas import ChatMessage

client = genai.Client(api_key=GOOGLE_API_KEY)

DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "gemma-3n-e4b-it")

# --- Helper Functions (Models, DB, Formatting) ---

# --- Helper Functions (Models, DB, Formatting) ---

def _get_model_name(model_name: str) -> str:
    # Ensure no 'models/' prefix for new SDK if strict, but usually it handles it.
    # The warning msg says switch to google.genai
    # We will just pass the model name string.
    return model_name

def _db() -> Chroma:
    return Chroma(
        embedding_function=embeddings_model,
        persist_directory=str(CHROMA_PATH),
        collection_name=COLLECTION_NAME,
    )

def _format_docs(docs: Iterable[Document]) -> str:
    import re
    cleaned_docs = []
    for doc in docs:
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', doc.page_content).strip()
        cleaned_docs.append(content)
    return "\n\n".join(cleaned_docs)

def _doc_meta(d: Document) -> Dict[str, Any]:
    meta = dict(getattr(d, "metadata", {}) or {})
    meta["source"] = meta.get("source") or meta.get("file_path") or meta.get("path") or ""
    meta["page"] = meta.get("page") or meta.get("page_number") or meta.get("page_index")
    return meta

# --- Query Condensing Function ---

# --- Query Condensing Function ---

def _condense_query_with_history(
    history: List[ChatMessage], 
    query: str, 
    model_name: str
) -> str:
    """
    Condenses chat history and a new query into a standalone question.
    """
    if not history:
        return query  # No history, no condensing needed

    # Format history for the prompt
    history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in history])
    
    prompt = (
        "You are a query condensing assistant. Given a chat history and a new follow-up question, "
        "rephrase the follow-up question to be a standalone question that includes all necessary context from the history.\n"
        "If the follow-up is already standalone, return it as is.\n\n"
        "--- CHAT HISTORY ---\n"
        f"{history_str}\n\n"
        f"--- FOLLOW-UP QUESTION ---\n{query}\n\n"
        "--- STANDALONE QUESTION: ---\n"
    )
    
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        condensed = (getattr(resp, "text", "") or "").strip()
        return condensed if condensed else query
    except Exception:
        return query  # Fail safe: return original query on error

# --- Query Correction (Handles Typos) ---

# --- Query Correction (Handles Typos) ---

def _correct_query(query: str, model_name: str) -> str:
    """
    Corrects spelling, typos, and abbreviations in the user's query.
    """
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
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        corrected = (getattr(resp, "text", "") or "").strip()
        if not corrected:
            return query # Return original on empty response
        return corrected
    except Exception:
        return query # Return original on any error

# --- Intent Classification ---

# --- Intent Classification ---

def _classify_intent(query: str, model_name: str) -> str:
    """
    Classifies the user's query as 'LEGAL' or 'GENERAL'.
    Uses the *corrected* query.
    """
    # Simple keyword check for obvious legal terms to save an LLM call or force LEGAL
    obvious_keywords = ["law", "act", "ordinance", "section", "article", "constitution", "court", "judge", "rights", "legal", "ppc", "crpc", "civil", "crime", "punishment"]
    if any(k in query.lower() for k in obvious_keywords):
        return "LEGAL"

    prompt = (
        "You are an intent classifier. Your job is to determine if the user's question is about Pakistani law, legal concepts, human rights, or government procedures.\n"
        "Respond with only a single word: 'LEGAL' or 'GENERAL'.\n\n"
        "Examples:\n"
        "User: What is the procedure for divorce?\n"
        "LEGAL\n"
        "User: Explain human rights.\n" # New Example
        "LEGAL\n"
        "User: rights of arrest person\n"
        "LEGAL\n"
        "User: Tell me about Section 489F.\n"
        "LEGAL\n"
        "User: what is ppc 302\n"
        "LEGAL\n"
        "User: Hi\n"
        "GENERAL\n"
        "User: hello\n"
        "GENERAL\n"
        "User: How are you?\n"
        "GENERAL\n"
        "User: What is the capital of France?\n"
        "GENERAL\n\n"
        f"User: {query}\n"
    )
    
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        text_resp = (getattr(resp, "text", "") or "").strip().upper()
        if text_resp == "LEGAL":
            return "LEGAL"
        elif text_resp == "GENERAL":
            return "GENERAL"
        else:
            return "LEGAL" # Default to LEGAL
    except Exception:
        return "LEGAL" # Default to LEGAL on error

# --- General Chat Function (Blocking) ---

# --- General Chat Function (Blocking) ---

def _general_chat(
    query: str, 
    history: List[ChatMessage], 
    model_name: str
) -> Dict[str, Any]:
    """
    Handles general conversation by answering with the LLM directly.
    """
    # Format history for the model
    # Note: Using "user" and "model" roles for genai API
    formatted_history = []
    for msg in history:
        role = "model" if msg.role == "assistant" else "user"
        formatted_history.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))
    
    system_prompt = (
        "You are a helpful and polite AI assistant. You are an expert in Pakistani law, "
        "but you can also answer general questions and engage in friendly conversation."
    )
    
    full_query = f"{system_prompt}\n\n{query}"
    
    try:
        chat = client.chats.create(
            model=model_name,
            history=formatted_history
        )
        resp = chat.send_message(full_query)
        answer = (getattr(resp, "text", "") or "").strip() or "I'm not sure how to respond to that."
        return {"answer": answer, "citations": []}
    except Exception as e:
        return {"answer": f"An error occurred while processing your request: {e}", "citations": []}

# --- General Chat Function (Streaming) ---

# --- General Chat Function (Streaming) ---

def _general_chat_stream(
    query: str, 
    history: List[ChatMessage], 
    model_name: str
) -> Iterator[str]:
    """
    Handles general conversation by streaming the LLM response.
    """
    # Format history for the model
    # Note: Using "user" and "model" roles for genai API
    formatted_history = []
    for msg in history:
        role = "model" if msg.role == "assistant" else "user"
        formatted_history.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))
    
    system_prompt = (
        "You are a helpful and polite AI assistant. You are an expert in Pakistani law, "
        "but you can also answer general questions and engage in friendly conversation."
    )
    
    full_query = f"{system_prompt}\n\n{query}"

    try:
        chat = client.chats.create(
            model=model_name,
            history=formatted_history
        )
        
        # Buffer for smooth chunking
        text_buffer = ""
        CHUNK_SIZE = 10
        
        # Send message with proper content format
        for chunk in chat.send_message_stream(types.Content(parts=[types.Part(text=full_query)])):
             try:
                 text_chunk = chunk.text
             except Exception:
                 continue
             
             if text_chunk:
                 text_buffer += text_chunk
                 
                 # Emit in smaller chunks
                 while len(text_buffer) >= CHUNK_SIZE:
                     yield text_buffer[:CHUNK_SIZE]
                     text_buffer = text_buffer[CHUNK_SIZE:]
        
        # Emit remaining
        if text_buffer:
            yield text_buffer
            
    except Exception as e:
        yield f"An error occurred while processing your request: {e}"


# --- RAG Pipeline Functions (Retrieval) ---

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

# --- RAG Query (Blocking) ---

def _rag_query(
    query: str, 
    history: List[ChatMessage], 
    k: int, 
    model_name: str
) -> Dict[str, Any]:
    """
    The RAG (Retrieval-Augmented Generation) pipeline.
    Uses the *condensed and corrected* query for retrieval.
    """
    filtered_docs = retrieve_mmr(query, k=max(8, k))
    if not filtered_docs:
        scored = retrieve_with_scores(query, k=k)
        filtered_docs = [d for d, s in scored if 0.0 <= s <= 1.0 and s >= 0.20] or [d for d, _ in scored]

    if not filtered_docs:
        return {"answer": "I cannot find any documents related to that topic.", "citations": []}

    context = _format_docs(filtered_docs)
    history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in history])
    
    # --- THIS PROMPT IS UPDATED ---
    prompt = (
        "You are an AI Legal Assistant for Pakistan law. Answer the user's question using the provided context.\n"
        "IMPORTANT NOTE ON CONTEXT QUALITY:\n"
        "The provided context contains extracted text from PDFs (e.g., Constitution, Acts).\n"
        "1.  **Text Repair**: Some words may be broken (e.g., 'A rticle' -> 'Article'). You MUST mentally repair them.\n"
        "2.  **Synthesis**: The context may not have the *exact* headers you expect. You must read the *content* to find the answer.\n"
        "3.  **Relevance**: If the context talks about the general topic (e.g., 'Human Rights'), use it to answer, even if it doesn't match the query word-for-word.\n\n"
        "**INSTRUCTIONS:**\n"
        "- Answer the question comprehensively using the information found in the context.\n"
        "- If the context mentions relevant laws/articles, cite them.\n"
        "- Do NOT be overly strict. If the documents provide a partial answer or general principles, use them.\n"
        "- Only say you cannot find the answer if the context is completely unrelated (e.g., context is about 'Banking' but query is about 'Murder').\n\n"
        "--- CONTEXT ---\n"
        f"{context}\n\n"
        "--- CHAT HISTORY ---\n"
        f"{history_str}\n\n"
        f"--- QUESTION ---\n{query}\n\n"
        "--- ANSWER ---\n"
    )

    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        answer = (getattr(resp, "text", "") or "").strip() or "I cannot find the answer in the provided documents."
    except Exception as e:
        answer = f"Error generating answer: {e}"

    citations = [{"source": _doc_meta(d).get("source"), "page": _doc_meta(d).get("page")} for d in filtered_docs[:5]]
    return {"answer": answer, "citations": citations}


# --- RAG Query (Streaming) ---

# --- RAG Query (Streaming) ---

def _rag_query_stream(
    query: str, 
    history: List[ChatMessage], 
    k: int, 
    model_name: str
) -> Iterator[Dict[str, Any]]:
    """
    RAG pipeline that streams the result with status updates.
    """
    # 1. Status Update: Searching
    yield {"type": "status", "data": "Searching knowledge base..."}
    
    # 1a. Retrieval (Blocking)
    filtered_docs = retrieve_mmr(query, k=max(8, k))
    if not filtered_docs:
        scored = retrieve_with_scores(query, k=k)
        filtered_docs = [d for d, s in scored if 0.0 <= s <= 1.0 and s >= 0.20] or [d for d, _ in scored]

    if not filtered_docs:
        yield {"type": "status", "data": "No relevant documents found."}
        yield {"type": "citations", "data": []}
        yield {"type": "content", "data": "I cannot find any documents related to that topic."}
        return

    # 1b. Status Update: Found docs
    yield {"type": "status", "data": f"Found {len(filtered_docs)} references. Synthesizing answer..."}

    # 2. Yield citations *first* (Typed Event)
    citations = [{"source": _doc_meta(d).get("source"), "page": _doc_meta(d).get("page")} for d in filtered_docs[:5]]
    yield {"type": "citations", "data": citations}

    # 3. Stream the answer
    context = _format_docs(filtered_docs)
    history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in history])
    
    prompt = (
        "You are an AI Legal Assistant for Pakistan law. Answer the user's question using the provided context.\n"
        "IMPORTANT NOTE ON CONTEXT QUALITY:\n"
        "The provided context contains extracted text from PDFs (e.g., Constitution, Acts).\n"
        "1.  **Text Repair**: Some words may be broken (e.g., 'A rticle' -> 'Article'). You MUST mentally repair them.\n"
        "2.  **Synthesis**: The context may not have the *exact* headers you expect. You must read the *content* to find the answer.\n"
        "3.  **Relevance**: If the context talks about the general topic (e.g., 'Human Rights'), use it to answer, even if it doesn't match the query word-for-word.\n\n"
        "**INSTRUCTIONS:**\n"
        "- Answer the question comprehensively using the information found in the context.\n"
        "- If the context mentions relevant laws/articles, cite them.\n"
        "- Do NOT be overly strict. If the documents provide a partial answer or general principles, use them.\n"
        "- Only say you cannot find the answer if the context is completely unrelated (e.g., context is about 'Banking' but query is about 'Murder').\n\n"
        "--- CONTEXT ---\n"
        f"{context}\n\n"
        "--- CHAT HISTORY ---\n"
        f"{history_str}\n\n"
        f"--- QUESTION ---\n{query}\n\n"
        "--- ANSWER ---\n"
    )



    try:
        # Stream response with character-level granularity for smooth display
        text_buffer = ""
        CHUNK_SIZE = 10  # Emit every 10 characters for smooth streaming
        
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=prompt
        ):
            try:
                text_chunk = chunk.text
            except Exception:
                continue

            if text_chunk:
                text_buffer += text_chunk
                
                # Emit in smaller chunks for smoother frontend display
                while len(text_buffer) >= CHUNK_SIZE:
                    yield {"type": "content", "data": text_buffer[:CHUNK_SIZE]}
                    text_buffer = text_buffer[CHUNK_SIZE:]
        
        # Emit any remaining buffered text
        if text_buffer:
            yield {"type": "content", "data": text_buffer}
            
    except Exception as e:
        yield {"type": "error", "data": f"An error occurred during streaming: {e}"}


# --- Main Entry Point (Blocking) ---
def answer_query(
    query: str, 
    history: Optional[List[ChatMessage]] = None, 
    k: int = 6, 
    model_name: str = DEFAULT_CHAT_MODEL
) -> Dict[str, Any]:
    
    history_list = history or []

    # --- STEP 1: Classify the RAW query first ---
    raw_intent = _classify_intent(query, model_name)
    
    # --- STEP 2: Route based on raw intent ---
    if raw_intent == "GENERAL":
        # It's a greeting or general question, go straight to the fast chat model.
        # We pass the raw query and full history for a natural conversation.
        return _general_chat(query, history_list, model_name)
    
    else:
        # It's a LEGAL query, so now we run the full (slower) RAG pipeline.
        
        # Step 2a: Condense query with history
        condensed_query = _condense_query_with_history(history_list, query, model_name)
        
        # Step 2b: Correct the condensed query
        corrected_query = _correct_query(condensed_query, model_name)
        
        # Step 2c: Run the RAG pipeline
        return _rag_query(corrected_query, history_list, k, model_name)


# --- Main Entry Point (Streaming) ---
def answer_query_stream(
    query: str, 
    history: Optional[List[ChatMessage]] = None, 
    k: int = 6, 
    model_name: str = DEFAULT_CHAT_MODEL
) -> Iterator[Dict[str, Any]]:
    
    history_list = history or []

    # Initial Status
    yield {"type": "status", "data": "Analyzing query..."}

    # --- STEP 1: Classify the RAW query first ---
    raw_intent = _classify_intent(query, model_name)
    
    # --- STEP 2: Route based on raw intent ---
    if raw_intent == "GENERAL":
        # It's a greeting, use the fast path.
        yield {"type": "citations", "data": []}
        
        # General chat stream helper needs adjustment or manual iteration here
        # Re-implementing simplified loop to ensure typed yield
        formatted_history = []
        for msg in history_list:
            role = "model" if msg.role == "assistant" else "user"
            formatted_history.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))
        
    # --- _general_chat fix ---
    # model = genai.GenerativeModel(model_name_for_api) # Simplified
    
    # --- _general_chat_stream fix ---
    
    # --- answer_query_stream (General) fix ---
        
        system_prompt = (
            "You are a helpful and polite AI assistant. You are an expert in Pakistani law, "
            "but you can also answer general questions and engage in friendly conversation."
        )
        
        # Prepend system prompt to the query content effectively
        full_query = f"{system_prompt}\n\n{query}"

        try:
            chat = client.chats.create(
                model=model_name,
                history=formatted_history
            )
            
            # Buffer for smooth chunking
            text_buffer = ""
            CHUNK_SIZE = 10
            
            for chunk in chat.send_message_stream(full_query):
                try:
                     text_chunk = chunk.text
                except Exception:
                     continue
                
                if text_chunk:
                    text_buffer += text_chunk
                    
                    # Emit in smaller chunks for smoother display
                    while len(text_buffer) >= CHUNK_SIZE:
                        yield {"type": "content", "data": text_buffer[:CHUNK_SIZE]}
                        text_buffer = text_buffer[CHUNK_SIZE:]
            
            # Emit remaining
            if text_buffer:
                yield {"type": "content", "data": text_buffer}
                
        except Exception as e:
            yield {"type": "error", "data": str(e)}
    
    else:
        # It's a LEGAL query, run the full RAG pipeline.
        
        # Step 2a: Condense query with history
        yield {"type": "status", "data": "Clarifying context..."}
        condensed_query = _condense_query_with_history(history_list, query, model_name)
        
        # Step 2b: Correct the condensed query
        corrected_query = _correct_query(condensed_query, model_name)
        
        # Step 2c: Run the RAG streaming pipeline
        for event in _rag_query_stream(corrected_query, history_list, k, model_name):
            yield event

# --- Debug/Diagnostics Functions ---

def search_docs(query: str, k: int = 8):
    results = retrieve_with_scores(query, k=k)
    out = []
    for i, (d, s) in enumerate(results, 1):
        preview = (d.page_content or "").replace("\n", " ")
        if len(preview) > 300:
            preview = preview[:300] + "..."
        out.append({"rank": i, "score": float(s), "metadata": _doc_meta(d), "preview": preview})
    return out

def direct_model_test(model_name: str = DEFAULT_CHAT_MODEL) -> str:
    try:
        resp = client.models.generate_content(
            model=model_name, 
            contents="Reply with a single word: pong"
        )
        reply = (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        reply = str(e)
    return f"Model: {model_name} | Reply: {reply}"

def db_info():
    db = _db()
    try:
        return {
            "count": int(db._collection.count()),
            "name": COLLECTION_NAME,
        }
    except Exception as e:
        return {"error": str(e), "type": e.__class__.__name__}