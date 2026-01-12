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
# Ensure the directory exists, otherwise this might raise an error if not built locally yet.
# for now we assume it will exist in prod.
import os
static_dir = "../frontend/dist/assets"
if os.path.exists(static_dir):
    app.mount("/assets", StaticFiles(directory=static_dir), name="assets")

# Catch-all route for SPA
@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    # If it's an API call that wasn't matched above, return 404 (handled by FastAPI usually, but good to be explicit if needed)
    # Actually, FastAPI matches in order. access to /api/... that fails above will NOT match this if it's strictly defined?
    # No, {full_path:path} matches everything.
    # So we need to ensure API 404s don't return HTML.
    if full_path.startswith("api/"):
        return {"error": "Not Found", "status": 404}
    
    # Serve index.html for any other route (SPA)
    index_path = "../frontend/dist/index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Frontend not built or not found"}
