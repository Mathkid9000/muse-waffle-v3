"""FastAPI application entry point.

Run with:
    uvicorn api.main:app --reload --app-dir .
"""

from __future__ import annotations
import logging
import os
import sys
from pathlib import Path

# without this the routers' own progress lines (and anything ml/ logs below WARNING) go nowhere,
# since uvicorn only configures its own loggers
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s: %(message)s")

from api.routers.edit import clef, time_sig, key_sig, tempo, instr, rm_staff, rm_measure, write_score
from api.routers.templates import fill_template, score_to_template

# Ensure the project root is on sys.path so ml/ imports resolve
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# ml/ modules import each other by bare name (they are run as scripts too), so ml/ has to be
# importable in its own right; appended rather than inserted so nothing in there can shadow a
# package the app depends on -- ml/api.py in particular must not outrank the api/ package
sys.path.append(str(PROJECT_ROOT / "ml"))
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
app.include_router(write_score.router, prefix="/api")
app.include_router(score_to_template.router, prefix="/api")
app.include_router(fill_template.router, prefix="/api")

# Serve the built React frontend in production (frontend/dist)
_dist = PROJECT_ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="static")
