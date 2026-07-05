"""FastAPI application entry point.

Run with:
    uvicorn api.main:app --reload --app-dir .
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

from api.routers.edit import clef, time_sig, key_sig, tempo, instr, rm_staff, rm_measure

# Ensure the project root is on sys.path so ml/ imports resolve
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Also set working directory to project root so relative paths (storage/) work
os.chdir(PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routers import infill, generate, append

app = FastAPI(title="Muse Waffle API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(infill.router, prefix="/api")
app.include_router(generate.router, prefix="/api")
app.include_router(append.router, prefix="/api")
app.include_router(clef.router, prefix="/api")
app.include_router(time_sig.router, prefix="/api")
app.include_router(key_sig.router, prefix="/api")
app.include_router(tempo.router, prefix="/api")
app.include_router(instr.router, prefix="/api")
app.include_router(rm_staff.router, prefix="/api")
app.include_router(rm_measure.router, prefix="/api")

# Serve the built React frontend in production (frontend/dist)
_dist = PROJECT_ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="static")
