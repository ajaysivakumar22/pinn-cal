"""
PINN-Cal Backend — Robustness schemas.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RobustnessRequest(BaseModel):
    sweep_type: Literal["snr", "coupling", "steering"] = "snr"
    levels: list[float] | None = None
    num_samples: int = Field(default=50, ge=10, le=200)
    seed: int = Field(default=4200, ge=0)


class RobustnessPoint(BaseModel):
    level: float
    analytical_nmse_db: float
    pinn_nmse_db: float | None
    analytical_phase_error_deg: float
    pinn_phase_error_deg: float | None
    analytical_gain_error_db: float
    pinn_gain_error_db: float | None
    analytical_relative_error_pct: float
    pinn_relative_error_pct: float | None


class RobustnessResponse(BaseModel):
    sweep_type: str
    levels: list[float]
    points: list[RobustnessPoint]
    pinn_available: bool
    note: str = ""
