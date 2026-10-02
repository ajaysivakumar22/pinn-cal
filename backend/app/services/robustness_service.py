"""
PINN-Cal Backend — Robustness Service

Generates fresh datasets at each sweep level and evaluates both
analytical and PINN calibration.

IMPORTANT:
    Uses calibration_dataset.generate_dataset() from the frozen scientific core.
    No physics equations are implemented here.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from typing import Optional

import numpy as np
import torch

from backend.app.scientific.adapter import (
    SCIENTIFIC_CORE_AVAILABLE,
    DEFAULT_CONFIG,
    mutual_impedance_matrix,
    steering_vector,
    DatasetConfig,
    generate_dataset,
)
from backend.app.core.config import (
    SELF_IMPEDANCE,
    NEAREST_NEIGHBOR_COUPLING,
    DECAY,
    PHASE_STEP_RAD,
    SNR_LEVELS,
    COUPLING_LEVELS,
)
from backend.app.services.model_service import ModelService
from backend.app.schemas.robustness import RobustnessRequest, RobustnessPoint


def _nmse_db(predicted: np.ndarray, target: np.ndarray) -> float:
    num = np.sum(np.abs(predicted - target) ** 2)
    den = np.sum(np.abs(target) ** 2)
    return float(10.0 * np.log10(num / (den + 1e-30)))


def _phase_error_deg(predicted: np.ndarray, target: np.ndarray) -> float:
    diff = np.angle(predicted * np.conj(target))
    return float(np.degrees(np.mean(np.abs(diff))))


def _gain_error_db(predicted: np.ndarray, target: np.ndarray) -> float:
    pred_db = 20.0 * np.log10(np.maximum(np.abs(predicted), 1e-12))
    tgt_db = 20.0 * np.log10(np.maximum(np.abs(target), 1e-12))
    return float(np.mean(np.abs(pred_db - tgt_db)))


def _relative_error(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - target) / (np.abs(target) + 1e-12)))


def _analytical_correction_batch(
    X: np.ndarray,
    steering_angles_deg: np.ndarray,
    z_nom_np: np.ndarray,
) -> np.ndarray:
    """Analytical calibration for a batch of samples."""
    observed_current = X[:, :64] + 1j * X[:, 64:128]
    corrections = []

    for i in range(len(X)):
        v_eff = z_nom_np @ observed_current[i]
        w_hw = v_eff / SELF_IMPEDANCE
        w_ideal = (
            steering_vector(
                float(steering_angles_deg[i]), DEFAULT_CONFIG
            )
            .detach()
            .cpu()
            .numpy()
        )
        corr = w_ideal / (w_hw + 1e-30)
        corrections.append(corr)

    return np.array(corrections, dtype=np.complex128)


def _evaluate_at_level(
    dataset_config_kwargs: dict,
    seed: int,
    num_samples: int,
    z_nom_np: np.ndarray,
    model_svc: Optional[object],
    pinn_enabled: bool,
) -> dict:
    """Generate a fresh dataset at one level and evaluate both methods."""
    ds_cfg = DatasetConfig(
        num_samples=num_samples,
        seed=seed,
        **dataset_config_kwargs,
    )

    data = generate_dataset(config=ds_cfg)
    X = data["X"]
    Y = data["Y"]
    angles = data["steering_angles"]

    # Ground-truth complex correction
    c_true = Y[:, :64] + 1j * Y[:, 64:]

    # Analytical calibration
    c_analytical = _analytical_correction_batch(X, angles, z_nom_np)

    ana_metrics = {
        "nmse_db": _nmse_db(c_analytical, c_true),
        "phase_error_deg": _phase_error_deg(c_analytical, c_true),
        "gain_error_db": _gain_error_db(c_analytical, c_true),
        "relative_error_pct": 100.0 * _relative_error(c_analytical, c_true),
    }

    pinn_m = None
    if pinn_enabled and model_svc is not None and model_svc.loaded:
        pred = model_svc.predict(X)  # (N, 128)
        c_pinn = pred[:, :64] + 1j * pred[:, 64:]

        pinn_m = {
            "nmse_db": _nmse_db(c_pinn, c_true),
            "phase_error_deg": _phase_error_deg(c_pinn, c_true),
            "gain_error_db": _gain_error_db(c_pinn, c_true),
            "relative_error_pct": 100.0 * _relative_error(c_pinn, c_true),
        }

    return {"analytical": ana_metrics, "pinn": pinn_m}


def run_robustness_sweep(req: RobustnessRequest) -> dict:
    """
    Execute a robustness sweep and return per-level metrics.
    """
    if not SCIENTIFIC_CORE_AVAILABLE:
        raise RuntimeError("Scientific core unavailable")

    z_nom = mutual_impedance_matrix(
        config=DEFAULT_CONFIG,
        self_impedance_ohm=SELF_IMPEDANCE,
        nearest_neighbor_coupling=NEAREST_NEIGHBOR_COUPLING,
        decay=DECAY,
        phase_step_rad=PHASE_STEP_RAD,
    )
    z_nom_np = z_nom.detach().cpu().numpy()

    model_svc = ModelService.get()
    pinn_enabled = model_svc.loaded

    if req.levels is None:
        if req.sweep_type == "snr":
            levels = SNR_LEVELS
        elif req.sweep_type == "coupling":
            levels = COUPLING_LEVELS
        else:  # steering
            levels = [-50.0, -30.0, -10.0, 10.0, 30.0, 50.0]
    else:
        levels = req.levels

    points: list[RobustnessPoint] = []

    for i, level in enumerate(levels):
        level_seed = req.seed + i * 1000

        if req.sweep_type == "snr":
            kwargs = {
                "snr_db": float(level),
                "coupling_mismatch_fraction": 0.20,
            }
        elif req.sweep_type == "coupling":
            kwargs = {
                "snr_db": 20.0,
                "coupling_mismatch_fraction": float(level),
            }
        else:  # steering — vary angle center
            target_angle = float(level)
            kwargs = {
                "snr_db": 20.0,
                "coupling_mismatch_fraction": 0.20,
                "min_angle_deg": target_angle - 10.0,
                "max_angle_deg": target_angle + 10.0,
            }

        result = _evaluate_at_level(
            dataset_config_kwargs=kwargs,
            seed=level_seed,
            num_samples=req.num_samples,
            z_nom_np=z_nom_np,
            model_svc=model_svc,
            pinn_enabled=pinn_enabled,
        )

        ana = result["analytical"]
        pinn = result["pinn"]

        points.append(RobustnessPoint(
            level=float(level),
            analytical_nmse_db=ana["nmse_db"],
            pinn_nmse_db=pinn["nmse_db"] if pinn else None,
            analytical_phase_error_deg=ana["phase_error_deg"],
            pinn_phase_error_deg=pinn["phase_error_deg"] if pinn else None,
            analytical_gain_error_db=ana["gain_error_db"],
            pinn_gain_error_db=pinn["gain_error_db"] if pinn else None,
            analytical_relative_error_pct=ana["relative_error_pct"],
            pinn_relative_error_pct=pinn["relative_error_pct"] if pinn else None,
        ))

    return {
        "sweep_type": req.sweep_type,
        "levels": [float(l) for l in levels],
        "points": points,
        "pinn_available": pinn_enabled,
        "note": (
            "Results computed from fresh datasets at each level. "
            "PINN checkpoint: models/pinn_cal_final_best.pth."
        ),
    }
