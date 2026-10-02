"""
PINN-Cal Backend — Simulation API route.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas.simulation import SimulationRequest, SimulationResponse
from backend.app.services.simulation_service import run_simulation

router = APIRouter()


@router.post("/simulation", response_model=None)
def simulation(req: SimulationRequest):
    """
    Execute the full physics simulation + calibration pipeline.

    Returns a unified response containing:
    - Per-element complex states (ideal, hardware, observed, corrections)
    - Beam patterns (ideal, uncalibrated, analytical, PINN)
    - Calibration metrics for both methods
    - Scalar summary metrics

    The PINN calibration is only available when num_elements=64 and
    frequency_hz is 28 GHz (the checkpoint's training configuration).
    """
    try:
        result = run_simulation(req)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}")

    return {
        "request": req.model_dump(),
        "pinn_available": result["pinn_available"],
        "pinn_compatibility_message": result["pinn_compatibility_message"],
        "elements": [e.model_dump() for e in result["elements"]],
        "beam": result["beam"],
        "calibration": result["calibration"],
        "metrics": result["metrics"],
    }
