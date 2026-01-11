import logging, traceback
from typing import Iterator, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import json
import asyncio

from backend.schemas import ChatMessage
from backend.database import get_chat_collection, get_session_collection
from backend.rag_pipeline import (
    answer_query,
    answer_query_stream,
    direct_model_test,
    search_docs,
    db_info,
    DEFAULT_CHAT_MODEL
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])


class ChatRequest(BaseModel):
    query: str
    k: int = 6
    history: List[ChatMessage] = []
    user_email: Optional[str] = None
    session_id: Optional[str] = None


async def ensure_session_exists(session_id: str, user_email: str, title: str):
    """
    Ensures a session document exists. If not, creates one.
    Updates last_interaction if it exists.
    """
    if not session_id or not user_email:
        return

    sessions_col = get_session_collection()
    if sessions_col is None:
        return

    # Try to find
    existing = await sessions_col.find_one({"session_id": session_id})
    timestamp = datetime.utcnow()
    
    if existing:
        # Just update timestamp
        await sessions_col.update_one(
            {"session_id": session_id},
            {"$set": {"last_interaction": timestamp}}
        )
    else:
        # Create new
        await sessions_col.insert_one({
            "session_id": session_id,
            "user_email": user_email,
            "title": title[:50] + "..." if len(title) > 50 else title,
            "created_at": timestamp,
            "last_interaction": timestamp
        })

@router.on_event("startup")
async def migrate_legacy_sessions():
    """
    One-time migration: Populate 'sessions' collection from existing 'chats' messages
    if they don't exist in 'sessions'.
    """
    try:
        chats_col = get_chat_collection()
        sessions_col = get_session_collection()
        if chats_col is None or sessions_col is None:
            return

        logger.info("Checking for legacy sessions to migrate...")
        
        # distinct session_ids (inefficient for HUGE dbs, but ok for this scale)
        # Using aggregation to group by session_id
        pipeline = [
            {"$group": {
                "_id": "$session_id",
                "user_email": {"$first": "$user_email"},
                "first_msg": {"$first": "$timestamp"}, # approximate start
                "last_msg": {"$last": "$timestamp"},
                 # Try to find the first USER message for title
                "msgs": {"$push": {"role": "$role", "content": "$content"}}
            }}
        ]
        
        legacy_groups = await chats_col.aggregate(pipeline).to_list(length=None)
        
        count = 0
        for group in legacy_groups:
            sid = group["_id"]
            if not sid: continue
            
            # Check if exists
            exists = await sessions_col.find_one({"session_id": sid})
            if not exists:
                # Determine title: First user message
                title = "Untitled Conversation"
                for m in group.get("msgs", []):
                    if m.get("role") == "user":
                        content = m.get("content") or m.get("text")
                        if content:
                            title = content[:50]
                            break
                
                await sessions_col.insert_one({
                    "session_id": sid,
                    "user_email": group.get("user_email"),
                    "title": title,
                    "created_at": group.get("first_msg"),
                    "last_interaction": group.get("last_msg") or datetime.utcnow()
                })
                count += 1
        
        if count > 0:
            logger.info(f"Migrated {count} legacy sessions to 'sessions' collection.")
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")


@router.post("/chat", response_class=JSONResponse)
async def chat(req: ChatRequest):
    """
    Standard (non-streaming) chat endpoint.
    """
    try:
        # Update/Create Session Info
        if req.user_email and req.session_id:
            await ensure_session_exists(req.session_id, req.user_email, req.query)

        result = answer_query(
            req.query, 
            history=req.history, 
            k=req.k, 
            model_name=DEFAULT_CHAT_MODEL
        )
        
        # --- Save to MongoDB if we have session info ---
        if req.user_email and req.session_id:
            try:
                chats_col = get_chat_collection()
                if chats_col is not None:
                    await chats_col.insert_one({
                        "session_id": req.session_id,
                        "user_email": req.user_email,
                        "role": "user",
                        "content": req.query,
                        "timestamp": datetime.utcnow()
                    })
                    
                    await chats_col.insert_one({
                        "session_id": req.session_id,
                        "user_email": req.user_email,
                        "role": "assistant",
                        "content": result.get("answer", ""),
                        "citations": result.get("sources", []),
                        "timestamp": datetime.utcnow()
                    })
            except Exception as db_err:
                logger.error(f"Failed to save chat to MongoDB: {db_err}")
        # -----------------------

        return result
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
    """
    # Create/Update Session immediately so it appears in sidebar
    if req.user_email and req.session_id:
         # We fire and forget this check to not block stream? No, better await it fast.
         pass 

    async def event_stream() -> Iterator[str]:
        # Update Session
        if req.user_email and req.session_id:
            await ensure_session_exists(req.session_id, req.user_email, req.query)

        full_answer = ""
        citations = []
        try:
            stream = answer_query_stream(
                req.query,
                history=req.history,
                k=req.k, 
                model_name=DEFAULT_CHAT_MODEL
            )
            
            for event_data in stream:
                if event_data.get("type") == "content":
                    full_answer += event_data.get("data", "")
                elif event_data.get("type") == "citations":
                    citations = event_data.get("data", [])
                
                yield f'data: {json.dumps(event_data)}\n\n'
            
            yield "data: [DONE]\n\n"
            
            # --- Save to MongoDB upon completion ---
            if req.user_email and req.session_id:
                try:
                    chats_col = get_chat_collection()
                    if chats_col is not None:
                        # User message
                        await chats_col.insert_one({
                            "session_id": req.session_id,
                            "user_email": req.user_email,
                            "role": "user",
                            "content": req.query,
                            "timestamp": datetime.utcnow()
                        })
                        # Assistant response
                        await chats_col.insert_one({
                            "session_id": req.session_id,
                            "user_email": req.user_email,
                            "role": "assistant",
                            "content": full_answer,
                            "citations": citations,
                            "timestamp": datetime.utcnow()
                        })
                except Exception as db_err:
                    logger.error(f"Failed to save streamed chat to DB: {db_err}")
            # ---------------------------------------
            
        except Exception as e:
            logger.exception("Streaming error")
            error_event = {"error": str(e), "type": e.__class__.__name__}
            yield f'data: {json.dumps(error_event)}\n\n'
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(), 
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


@router.get("/chat/history", response_class=JSONResponse)
async def get_history(user_email: str):
    """
    Get list of chat sessions from the 'sessions' collection.
    """
    sessions_col = get_session_collection()
    if sessions_col is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    # Query sessions directly
    cursor = sessions_col.find({"user_email": user_email}).sort("last_interaction", -1)
    
    sessions = []
    async for doc in cursor:
        sessions.append({
            "id": doc.get("session_id"),
            "title": doc.get("title") or "Untitled Conversation",
            "last_interaction": doc.get("last_interaction")
        })
        
    return sessions


@router.get("/chat/session/{session_id}", response_class=JSONResponse)
async def get_session(session_id: str):
    """
    Get full history for a specific session ID.
    """
    chats_col = get_chat_collection()
    if chats_col is None:
         raise HTTPException(status_code=500, detail="Database not available")
    
    logger.info(f"Fetching session {session_id}")
    cursor = chats_col.find({"session_id": session_id}).sort("timestamp", 1)
    messages = []
    async for doc in cursor:
        messages.append({
            "id": str(doc.get("_id")),
            "role": doc.get("role"),
            "content": doc.get("content") or doc.get("text") or "", # Fallback to empty string
            "citations": doc.get("citations", [])
        })
    logger.info(f"Found {len(messages)} messages for session {session_id}")
    return messages


@router.delete("/chat/session/{session_id}")
async def delete_session(session_id: str):
    chats_col = get_chat_collection()
    sessions_col = get_session_collection()
    
    if chats_col is None or sessions_col is None:
         raise HTTPException(status_code=500, detail="Database not available")
    
    # Delete from messages
    await chats_col.delete_many({"session_id": session_id})
    # Delete from sessions
    res = await sessions_col.delete_one({"session_id": session_id})
    
    if res.deleted_count == 0:
        # It might be in legacy only, but we deleted messages so return success/warning
        pass
        
    return {"status": "success", "session_id": session_id}

@router.put("/chat/session/{session_id}")
async def rename_session(session_id: str, body: dict):
    sessions_col = get_session_collection()
    if sessions_col is None:
         raise HTTPException(status_code=500, detail="Database not available")
         
    new_title = body.get("title")
    if not new_title:
        raise HTTPException(status_code=400, detail="Title is required")
        
    res = await sessions_col.update_one(
        {"session_id": session_id},
        {"$set": {"title": new_title}}
    )
    
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
        
    return {"status": "success", "title": new_title}


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
