"""
PINN-Cal Backend — FastAPI Application

This is the entry point for the backend server.

Run from D:\\pinn-cal with:
    uvicorn backend.app.main:app --reload --port 8000

The working directory MUST be D:\\pinn-cal so that relative paths
to data/ and models/ resolve correctly.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.simulation import router as simulation_router
from backend.app.api.robustness import router as robustness_router
from backend.app.api.system import router as system_router
from backend.app.services.model_service import ModelService


# ---------------------------------------------------------------------------
# Lifespan — load PINN model once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm up the model singleton
    model_svc = ModelService.get()
    if model_svc.loaded:
        print(f"[PINN-Cal] PINN model loaded in {model_svc._load_time_s:.2f}s")
    else:
        print(f"[PINN-Cal] PINN model NOT loaded: {model_svc.load_error}")

    yield

    # Shutdown
    print("[PINN-Cal] Backend shutting down.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PINN-Cal API",
    description=(
        "Physics-Informed Neural Calibration for 5G Massive MIMO Antenna Arrays. "
        "Backend API providing live computation of antenna simulation, calibration, "
        "beam patterns, and robustness analysis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow Vite dev server, production builds, or configured origins
# ---------------------------------------------------------------------------

import os

default_origins = [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:5174",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

env_origins = os.getenv("ALLOWED_ORIGINS")
allowed_origins = [origin.strip() for origin in env_origins.split(",") if origin.strip()] if env_origins else default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(system_router, prefix="/api")
app.include_router(simulation_router, prefix="/api")
app.include_router(robustness_router, prefix="/api")


@app.get("/")
def root():
    return {
        "service": "PINN-Cal API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }
