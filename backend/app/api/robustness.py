"""
PINN-Cal Backend — Robustness API route.
"""

from __future__ import annotations

import asyncio
from fastapi import APIRouter, HTTPException

from backend.app.schemas.robustness import RobustnessRequest, RobustnessResponse
from backend.app.services.robustness_service import run_robustness_sweep

router = APIRouter()


@router.post("/robustness")
async def robustness(req: RobustnessRequest):
    """
    Execute a robustness sweep across SNR, coupling mismatch, or steering angle.

    Generates fresh datasets at each level and evaluates both analytical
    and PINN calibration. Results reflect actual computation; no cached
    static data is returned.

    Note: Offloaded to worker thread via asyncio.to_thread so heavy Monte Carlo
    loops do not block concurrent requests on the FastAPI async event loop.
    """
    try:
        result = await asyncio.to_thread(run_robustness_sweep, req)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Robustness sweep error: {exc}")

    return {
        "sweep_type": result["sweep_type"],
        "levels": result["levels"],
        "points": [p.model_dump() for p in result["points"]],
        "pinn_available": result["pinn_available"],
        "note": result["note"],
    }
