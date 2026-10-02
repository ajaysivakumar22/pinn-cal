"""
PINN-Cal Backend — Beam Service

Wraps physics_core.py beam-pattern functions.
No physics equations are reimplemented here.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from backend.app.scientific.adapter import (
    DEFAULT_CONFIG,
    ArrayConfig,
    steering_vector,
    power_pattern,
)
from backend.app.core.config import (
    BEAM_ANGLE_MIN,
    BEAM_ANGLE_MAX,
    BEAM_ANGLE_POINTS,
)

# Angular grid — matches evaluate_beam_metrics.py exactly (0.1° resolution)
ANGLE_GRID_DEG = np.linspace(BEAM_ANGLE_MIN, BEAM_ANGLE_MAX, BEAM_ANGLE_POINTS)


def _get_angle_grid(device) -> torch.Tensor:
    return torch.tensor(ANGLE_GRID_DEG, dtype=torch.float64, device=device)


def compute_power_pattern_db(
    weights: torch.Tensor,
    config=None,
) -> np.ndarray:
    """
    Compute normalized dB power pattern.

    Returns 1-D array of length BEAM_ANGLE_POINTS (dB, 0 dB = peak).
    """
    if config is None:
        config = DEFAULT_CONFIG

    angle_grid = _get_angle_grid(weights.device)

    # weights may be 1-D or batch; power_pattern handles both
    if weights.ndim == 1:
        weights = weights.unsqueeze(0)

    pat = power_pattern(weights, angle_grid, config)  # shape (N, K)
    pat_np = pat.detach().cpu().numpy()

    if pat_np.ndim == 2:
        pat_np = pat_np[0]  # take first sample

    # Already normalized to peak=1 by power_pattern; clip then log
    pat_db = 10.0 * np.log10(pat_np.clip(min=1e-6))
    return pat_db


def _beamwidth_3db(pattern_linear: np.ndarray, peak_idx: int) -> Optional[float]:
    """Find -3 dB beamwidth around peak_idx using linear interpolation."""
    threshold = 0.5

    left_crossing = None
    for i in range(peak_idx - 1, -1, -1):
        if pattern_linear[i] < threshold <= pattern_linear[i + 1]:
            den = pattern_linear[i + 1] - pattern_linear[i]
            if abs(den) > 1e-15:
                frac = (threshold - pattern_linear[i]) / den
                left_crossing = ANGLE_GRID_DEG[i] + frac * (ANGLE_GRID_DEG[i + 1] - ANGLE_GRID_DEG[i])
            else:
                left_crossing = 0.5 * (ANGLE_GRID_DEG[i] + ANGLE_GRID_DEG[i + 1])
            break

    right_crossing = None
    for i in range(peak_idx, len(pattern_linear) - 1):
        if pattern_linear[i] >= threshold > pattern_linear[i + 1]:
            den = pattern_linear[i + 1] - pattern_linear[i]
            if abs(den) > 1e-15:
                frac = (threshold - pattern_linear[i]) / den
                right_crossing = ANGLE_GRID_DEG[i] + frac * (ANGLE_GRID_DEG[i + 1] - ANGLE_GRID_DEG[i])
            else:
                right_crossing = 0.5 * (ANGLE_GRID_DEG[i] + ANGLE_GRID_DEG[i + 1])
            break

    if left_crossing is None or right_crossing is None:
        return None
    return float(right_crossing - left_crossing)


def _peak_sidelobe_level_db(pattern_linear: np.ndarray, peak_idx: int) -> Optional[float]:
    """Estimate PSLL by excluding the main lobe between its nearest nulls."""
    left_boundary = None
    for i in range(peak_idx - 1, 0, -1):
        if pattern_linear[i] <= pattern_linear[i - 1] and pattern_linear[i] <= pattern_linear[i + 1]:
            left_boundary = i
            break

    right_boundary = None
    for i in range(peak_idx + 1, len(pattern_linear) - 1):
        if pattern_linear[i] <= pattern_linear[i - 1] and pattern_linear[i] <= pattern_linear[i + 1]:
            right_boundary = i
            break

    if left_boundary is None or right_boundary is None:
        return None

    sidelobe = np.concatenate([
        pattern_linear[:left_boundary],
        pattern_linear[right_boundary + 1:],
    ])

    if sidelobe.size == 0:
        return None

    return float(10.0 * np.log10(max(float(np.max(sidelobe)), 1e-12)))


def _beam_correlation(pattern: np.ndarray, ideal_pattern: np.ndarray) -> float:
    num = float(np.sum(pattern * ideal_pattern))
    den = float(np.linalg.norm(pattern) * np.linalg.norm(ideal_pattern))
    if den <= 1e-12:
        return float("nan")
    return float(num / den)


def compute_trace(
    weights: torch.Tensor,
    label: str,
    target_angle_deg: float,
    ideal_linear: Optional[np.ndarray] = None,
    config=None,
) -> dict:
    """Compute a single beam pattern trace dict."""
    if config is None:
        config = DEFAULT_CONFIG

    pat_db = compute_power_pattern_db(weights, config)
    pat_linear = 10.0 ** (pat_db / 10.0)  # already 0-dB normalized

    peak_idx = int(np.argmax(pat_linear))
    peak_angle = float(ANGLE_GRID_DEG[peak_idx])
    pointing_error = float(peak_angle - target_angle_deg)
    bw = _beamwidth_3db(pat_linear, peak_idx)
    psll = _peak_sidelobe_level_db(pat_linear, peak_idx)

    corr: Optional[float] = None
    if ideal_linear is not None:
        corr = _beam_correlation(pat_linear, ideal_linear)

    return {
        "label": label,
        "angles_deg": ANGLE_GRID_DEG.tolist(),
        "power_db": pat_db.tolist(),
        "peak_angle_deg": peak_angle,
        "pointing_error_deg": pointing_error,
        "beamwidth_3db_deg": bw,
        "psll_db": psll,
        "beam_correlation": corr,
    }


def compute_all_beam_patterns(
    ideal_weights: torch.Tensor,
    uncalibrated_weights: torch.Tensor,
    analytical_weights: torch.Tensor,
    pinn_weights: Optional[torch.Tensor],
    target_angle_deg: float,
    config=None,
) -> dict:
    """
    Compute beam traces for ideal, uncalibrated, analytical, and PINN modes.

    Returns a dict with keys: ideal, uncalibrated, analytical, pinn (or None).
    """
    if config is None:
        config = DEFAULT_CONFIG

    ideal_trace = compute_trace(ideal_weights, "Ideal", target_angle_deg, config=config)
    ideal_linear = 10.0 ** (np.array(ideal_trace["power_db"]) / 10.0)

    # Self-correlation of ideal = 1.0
    ideal_trace["beam_correlation"] = 1.0

    uncal_trace = compute_trace(
        uncalibrated_weights, "Uncalibrated", target_angle_deg,
        ideal_linear=ideal_linear, config=config,
    )
    ana_trace = compute_trace(
        analytical_weights, "Analytical", target_angle_deg,
        ideal_linear=ideal_linear, config=config,
    )

    pinn_trace = None
    if pinn_weights is not None:
        pinn_trace = compute_trace(
            pinn_weights, "PINN", target_angle_deg,
            ideal_linear=ideal_linear, config=config,
        )

    return {
        "ideal": ideal_trace,
        "uncalibrated": uncal_trace,
        "analytical": ana_trace,
        "pinn": pinn_trace,
    }
