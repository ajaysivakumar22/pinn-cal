"""
PINN-Cal Backend — System/Health API routes.
"""

from __future__ import annotations

import sys
import time
import platform

import torch
from fastapi import APIRouter

from backend.app.services.model_service import ModelService
from backend.app.scientific.adapter import SCIENTIFIC_CORE_AVAILABLE, SCIENTIFIC_CORE_ERROR
from backend.app.core.config import VALIDATED_RESULTS

router = APIRouter()

_START_TIME = time.monotonic()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": time.time(),
        "uptime_s": round(time.monotonic() - _START_TIME, 2),
        "scientific_core_available": SCIENTIFIC_CORE_AVAILABLE,
        "scientific_core_error": SCIENTIFIC_CORE_ERROR,
        "pinn_model_loaded": ModelService.get().loaded,
        "pinn_model_error": ModelService.get().load_error,
    }


@router.get("/system-status")
def system_status():
    return {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "cuda_device_count": torch.cuda.device_count(),
        "platform": platform.platform(),
        "scientific_core_available": SCIENTIFIC_CORE_AVAILABLE,
        "scientific_core_error": SCIENTIFIC_CORE_ERROR,
        "backend_uptime_s": round(time.monotonic() - _START_TIME, 2),
    }


@router.get("/model-status")
def model_status():
    status = ModelService.get().status_dict()
    status["validated_results"] = VALIDATED_RESULTS
    status["honesty_note"] = (
        "Current validated results show the analytical baseline achieves lower "
        "correction error than the PINN for this configuration. Both methods are "
        "shown neutrally. The PINN results are highly reproducible (multi-seed tested)."
    )
    return status


@router.get("/config")
def get_config():
    from backend.app.core.config import (
        DEFAULT_NUM_ELEMENTS,
        DEFAULT_FREQUENCY_HZ,
        DEFAULT_ELEMENT_SPACING_LAMBDA,
        MIN_ANGLE_DEG,
        MAX_ANGLE_DEG,
        SNR_LEVELS,
        COUPLING_LEVELS,
        BEAM_ANGLE_POINTS,
    )
    return {
        "pinn_compatible": {
            "num_elements": DEFAULT_NUM_ELEMENTS,
            "frequency_hz": DEFAULT_FREQUENCY_HZ,
            "element_spacing_lambda": DEFAULT_ELEMENT_SPACING_LAMBDA,
        },
        "steering_range_deg": [MIN_ANGLE_DEG, MAX_ANGLE_DEG],
        "robustness": {
            "snr_levels_db": SNR_LEVELS,
            "coupling_levels": COUPLING_LEVELS,
        },
        "beam_angle_points": BEAM_ANGLE_POINTS,
        "supported_num_elements": list(range(2, 257)),
        "note": (
            "PINN calibration is only available for exactly 64 elements at 28 GHz. "
            "All other configurations run physics simulation only."
        ),
    }
