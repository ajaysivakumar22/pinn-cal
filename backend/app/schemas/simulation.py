"""
PINN-Cal — Simulation schemas (request + response).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SimulationRequest(BaseModel):
    """
    Parameters for a single physics simulation.

    When num_elements != 64 or frequency_hz != 28 GHz:
        - Physics simulation runs normally
        - PINN calibration is unavailable (checkpoint is trained for 64 × 28 GHz)
    """

    num_elements: int = Field(
        default=64,
        ge=2,
        le=256,
        description="Number of antenna elements.",
    )
    frequency_hz: float = Field(
        default=28e9,
        gt=0,
        description="Operating frequency in Hz.",
    )
    element_spacing_lambda: float = Field(
        default=0.5,
        gt=0,
        le=2.0,
        description="Element spacing in wavelengths.",
    )
    target_angle_deg: float = Field(
        default=0.0,
        ge=-85.0,
        le=85.0,
        description="Target steering angle in degrees.",
    )
    phase_sigma_deg: float = Field(
        default=20.0,
        ge=0.0,
        le=90.0,
        description="Hardware phase error standard deviation (degrees).",
    )
    gain_sigma_db: float = Field(
        default=2.0,
        ge=0.0,
        le=10.0,
        description="Hardware gain error standard deviation (dB).",
    )
    snr_db: float = Field(
        default=20.0,
        ge=-10.0,
        le=60.0,
        description="Measurement SNR in dB.",
    )
    coupling_mismatch_fraction: float = Field(
        default=0.20,
        ge=0.0,
        lt=1.0,
        description="Hidden coupling mismatch fraction (0 = no mismatch).",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for hardware error and noise generation.",
    )

    @model_validator(mode="after")
    def validate_angle_range(self) -> "SimulationRequest":
        if not (-85.0 <= self.target_angle_deg <= 85.0):
            raise ValueError(
                f"target_angle_deg must be between -85 and 85°, "
                f"got {self.target_angle_deg:.1f}°."
            )
        return self

    @property
    def pinn_compatible(self) -> bool:
        """True if the PINN checkpoint is compatible with this configuration."""
        return (
            self.num_elements == 64
            and abs(self.frequency_hz - 28e9) < 1e6
        )


# ---------------------------------------------------------------------------
# Per-element data
# ---------------------------------------------------------------------------

class ElementState(BaseModel):
    """Per-element complex state for all four modes, including RF diagnostics."""

    element_index: int
    # Ideal excitation
    ideal_real: float
    ideal_imag: float
    ideal_magnitude: float
    ideal_phase_deg: float
    # Hardware (impaired) excitation
    hardware_real: float
    hardware_imag: float
    hardware_magnitude: float
    hardware_phase_deg: float
    # Hardware errors (Simulation Ground Truth)
    gain_error_db: float
    phase_error_deg: float
    true_gain_error_db: float = 0.0
    true_phase_error_deg: float = 0.0
    # Observed (noisy) current
    observed_real: float
    observed_imag: float
    # Analytical correction coefficient
    analytical_correction_real: float
    analytical_correction_imag: float
    analytical_correction_magnitude: float
    analytical_correction_phase_deg: float
    # Analytical estimated impairments & residuals
    analytical_estimated_phase_error_deg: float = 0.0
    analytical_estimated_gain_error_db: float = 0.0
    analytical_phase_residual_deg: float = 0.0
    analytical_gain_residual_db: float = 0.0
    # PINN correction coefficient (None if unavailable)
    pinn_correction_real: float | None = None
    pinn_correction_imag: float | None = None
    pinn_correction_magnitude: float | None = None
    pinn_correction_phase_deg: float | None = None
    # PINN estimated impairments & residuals
    pinn_estimated_phase_error_deg: float | None = None
    pinn_estimated_gain_error_db: float | None = None
    phase_error_residual_deg: float | None = None
    gain_error_residual_db: float | None = None
    # Compensation applied (opposite of estimated impairment)
    compensation_phase_deg: float | None = None
    compensation_gain_db: float | None = None


# ---------------------------------------------------------------------------
# Beam pattern
# ---------------------------------------------------------------------------

class BeamTrace(BaseModel):
    """A single beam pattern trace."""

    label: str
    angles_deg: list[float]
    power_db: list[float]
    peak_angle_deg: float
    pointing_error_deg: float
    beamwidth_3db_deg: float | None
    psll_db: float | None
    beam_correlation: float | None


class BeamPatternResult(BaseModel):
    """All four beam pattern traces for one simulation."""

    ideal: BeamTrace
    uncalibrated: BeamTrace
    analytical: BeamTrace
    pinn: BeamTrace | None = None


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

class CalibrationMetrics(BaseModel):
    """Calibration performance metrics for one method."""

    nmse_db: float
    mean_phase_error_deg: float
    mean_gain_error_db: float
    mean_relative_error_pct: float
    current_residual_pct: float | None = None


class CalibrationComparison(BaseModel):
    """Side-by-side calibration metrics."""

    analytical: CalibrationMetrics
    pinn: CalibrationMetrics | None = None
    pinn_available: bool


# ---------------------------------------------------------------------------
# Unified simulation response
# ---------------------------------------------------------------------------

class SimulationResponse(BaseModel):
    """
    Unified response from POST /api/simulation.

    Contains all data needed to update the entire dashboard in one request.
    """

    request: SimulationRequest
    pinn_available: bool
    pinn_compatibility_message: str

    # Per-element states for the antenna visualization
    elements: list[ElementState]

    # Beam pattern results
    beam: BeamPatternResult

    # Calibration metrics comparison
    calibration: CalibrationComparison

    # Scalar summary metrics for right panel
    target_angle_deg: float
    beam_peak_angle_deg: float
    pointing_error_deg: float
    beam_correlation: float
    beamwidth_3db_deg: float | None
    psll_db: float | None
    analytical_nmse_db: float
    pinn_nmse_db: float | None
