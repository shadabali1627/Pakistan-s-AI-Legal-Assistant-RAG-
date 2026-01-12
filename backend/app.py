from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.routes import chat  # registers endpoints
from backend.routes import auth  # registers endpoints
from backend.config import DATA_PATH, CHROMA_PATH, VECTOR_STORE_BACKEND

app = FastAPI(title="AI Legal Assistant RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)





# Mount routes
app.include_router(chat.router, prefix="/api")
app.include_router(auth.router, prefix="/api")

# Mount frontend static files
# Enable robust path resolution regardless of where the script is run from
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
# BASE_DIR is .../backend
# We want .../frontend/dist
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"
STATIC_DIR = FRONTEND_DIST / "assets"
INDEX_PATH = FRONTEND_DIST / "index.html"

if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR)), name="assets")

# Catch-all route for SPA
@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    if full_path.startswith("api/"):
        return {"error": "Not Found", "status": 404}
    
    if INDEX_PATH.exists():
        return FileResponse(str(INDEX_PATH))
    return {"error": "Frontend not built", "path_tried": str(INDEX_PATH), "cwd": os.getcwd()}
