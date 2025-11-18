import logging, traceback
from typing import Iterator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import json

from backend.rag_pipeline import (
    answer_query,
    direct_model_test,
    search_docs,
    db_info,
    answer_query_stream,
    DEFAULT_CHAT_MODEL  # <-- 1. IMPORT THE DEFAULT MODEL NAME
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])


class ChatRequest(BaseModel):
    query: str
    k: int = 6


@router.post("/chat", response_class=JSONResponse)
async def chat(req: ChatRequest):
    """
    Standard (non-streaming) chat endpoint.
    """
    try:
        # Pass the model from config, though answer_query uses it as a default anyway
        result = answer_query(req.query, k=req.k, model_name=DEFAULT_CHAT_MODEL)
        return result  # {"answer": "...", "citations": [...]}
    except Exception as e:
        logger.exception("Chat error")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": e.__class__.__name__, "trace": traceback.format_exc()},
        )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint.
    Yields Server-Sent Events (SSE) as JSON objects.
    Event types:
    - {"citations": [...]} (Once at the beginning)
    - {"token": "..."}     (Repeated for answer)
    - {"error": "..."}    (If an error occurs)
    """
    def event_stream() -> Iterator[str]:
        try:
            # 2. CALL THE STREAMING ROUTER WITH THE DEFAULT MODEL
            stream = answer_query_stream(
                req.query, 
                k=req.k, 
                model_name=DEFAULT_CHAT_MODEL
            )
            
            for event_data in stream:
                # 3. YIELD THE JSON-SERIALIZED DICT
                # event_data is either {"citations": [...]} or {"token": "..."}
                yield f'data: {json.dumps(event_data)}\n\n'
            
            # 4. SEND [DONE] SIGNAL
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.exception("Streaming error")
            # 5. SEND ERROR AS A JSON OBJECT
            error_event = {"error": str(e), "type": e.__class__.__name__}
            yield f'data: {json.dumps(error_event)}\n\n'
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream; charset=utf-8")


# ---- diagnostics ----
@router.get("/debug/ping", response_class=JSONResponse)
async def ping_model():
    try:
        txt = direct_model_test()
        return {"ok": True, "model_reply": txt}
    except Exception as e:
        logger.exception("Ping error")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@router.get("/debug/search", response_class=JSONResponse)
async def debug_search(q: str, k: int = 8):
    try:
        return {"query": q, "k": k, "results": search_docs(q, k=k)}
    except Exception as e:
        logger.exception("Search error")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/debug/dbinfo", response_class=JSONResponse)
async def debug_dbinfo():
    return db_info()