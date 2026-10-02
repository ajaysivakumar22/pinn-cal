"""
Scientific Core Adapter

Adds the repository root to sys.path so frozen scientific modules can be
imported without modification.

Path layout:
    D:/pinn-cal/                    ← repo root (parents[3])
        backend/app/scientific/adapter.py  ← this file
        physics_core.py             ← frozen scientific core
        pinn_cal_final.py
        calibration_dataset.py
        coupling_model.py
        analytical_baseline.py

DO NOT copy, re-implement, or modify scientific equations here.
This module is purely a path adapter + safe import wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root — parents[3] from backend/app/scientific/adapter.py
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Import frozen scientific modules
# If any import fails, SCIENTIFIC_CORE_AVAILABLE becomes False and the
# error is surfaced through /api/health.
# ---------------------------------------------------------------------------

SCIENTIFIC_CORE_AVAILABLE: bool
SCIENTIFIC_CORE_ERROR: str | None

try:
    from physics_core import (  # type: ignore[import]
        ArrayConfig,
        DEFAULT_CONFIG,
        element_positions,
        steering_vector,
        apply_hardware_errors,
        generate_hardware_errors,
        mutual_impedance_matrix,
        apply_mutual_coupling,
        array_factor,
        power_pattern,
        beam_peak_angle,
        normalize_power,
    )

    from pinn_cal_final import (  # type: ignore[import]
        PINNCalV3,
        DEVICE,
        INPUT_DIM,
        OUTPUT_DIM,
        M,
    )

    from calibration_dataset import (  # type: ignore[import]
        DatasetConfig,
        generate_sample,
        generate_dataset,
    )

    from coupling_model import (  # type: ignore[import]
        generate_true_coupling_matrix,
    )

    SCIENTIFIC_CORE_AVAILABLE = True
    SCIENTIFIC_CORE_ERROR = None

except Exception as _exc:
    SCIENTIFIC_CORE_AVAILABLE = False
    SCIENTIFIC_CORE_ERROR = str(_exc)

    # Provide stub symbols so other modules can still import without
    # crashing at import time.  Runtime calls will raise descriptive errors.
    class _Stub:
        pass

    ArrayConfig = _Stub  # type: ignore
    DEFAULT_CONFIG = None  # type: ignore
    PINNCalV3 = None  # type: ignore
    DEVICE = None  # type: ignore
    INPUT_DIM = 130  # type: ignore
    OUTPUT_DIM = 128  # type: ignore
    M = 64  # type: ignore
    DatasetConfig = _Stub  # type: ignore
    generate_sample = None  # type: ignore
    generate_dataset = None  # type: ignore
    generate_true_coupling_matrix = None  # type: ignore

    element_positions = None  # type: ignore
    steering_vector = None  # type: ignore
    apply_hardware_errors = None  # type: ignore
    generate_hardware_errors = None  # type: ignore
    mutual_impedance_matrix = None  # type: ignore
    apply_mutual_coupling = None  # type: ignore
    array_factor = None  # type: ignore
    power_pattern = None  # type: ignore
    beam_peak_angle = None  # type: ignore
    normalize_power = None  # type: ignore
