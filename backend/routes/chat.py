import logging, traceback
from typing import Iterator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import json  # <-- 1. IMPORT JSON

from backend.rag_pipeline import (
    answer_query,
    direct_model_test,
    search_docs,
    db_info,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])


class ChatRequest(BaseModel):
    query: str
    k: int = 6


@router.post("/chat", response_class=JSONResponse)
async def chat(req: ChatRequest):
    try:
        result = answer_query(req.query, k=req.k)
        return result  # {"answer": "...", "citations": [...]}
    except Exception as e:
        logger.exception("Chat error")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": e.__class__.__name__, "trace": traceback.format_exc()},
        )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    def event_stream() -> Iterator[str]:
        try:
            result = answer_query(req.query, k=req.k)
            text = result.get("answer", "")
            for token in text.split():
                # 2. USE JSON.DUMPS FOR VALID JSON
                yield f'data: {json.dumps({"token": token + " "})}\n\n'
            
            # 3. SEND CITATIONS AS VALID JSON
            yield f'data: {json.dumps({"citations": result.get("citations", [])})}\n\n'
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception("Streaming error")
            # ALSO USE JSON.DUMPS FOR ERRORS
            yield f'data: {json.dumps({"token": f"Streaming error: {e}"})}\n\n'
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