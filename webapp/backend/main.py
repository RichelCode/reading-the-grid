"""FastAPI backend for the Reading the Grid web app.

Scaffold only: exposes a health check and a placeholder hello endpoint under ``/api``, and
serves the built React frontend as static files when present. The ML model is wired in a
later step. In production (Docker) this process serves both the API and the frontend on a
single port; in local development the frontend runs on the Vite dev server and proxies
``/api`` requests here.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Reading the Grid API")

# Permissive CORS so the Vite dev server (a different origin during development) can call
# the API directly if it is not using the proxy. The app has no auth and stores nothing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe used by the frontend's online/offline indicator."""
    return {"status": "ok"}


@app.get("/api/hello")
def hello() -> dict[str, str]:
    """Placeholder endpoint, replaced by model inference in a later step."""
    return {"message": "Reading the Grid backend online"}


# Serve the built frontend when it exists. The Docker build copies the compiled React app
# to ./static; in local dev that directory is absent, so this mount is skipped and the
# Vite dev server serves the UI instead. Mounted last so it does not shadow /api routes.
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
