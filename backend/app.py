from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import chat  # registers endpoints
from backend.config import DATA_PATH, CHROMA_PATH, VECTOR_STORE_BACKEND

app = FastAPI(title="AI Legal Assistant RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "AI Legal Assistant API is running!"}

# Mount routes
app.include_router(chat.router, prefix="/api")
