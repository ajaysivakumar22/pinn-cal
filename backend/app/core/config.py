"""
PINN-Cal Backend — Application Configuration

All paths are resolved relative to the repository root (D:/pinn-cal),
which is the working directory when running the backend.

Environment information is detected dynamically at runtime.
Do not hard-code Python version, PyTorch version, or CUDA state.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root — parents[3] from backend/app/core/config.py
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Scientific core paths
# ---------------------------------------------------------------------------

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"

PINN_BEST_CHECKPOINT = MODELS_DIR / "pinn_cal_final_best.pth"
PINN_LAST_CHECKPOINT = MODELS_DIR / "pinn_cal_final_last.pth"

TEST_DATA_PATH = DATA_DIR / "physics_test_200.npy"
TRAIN_DATA_PATH = DATA_DIR / "physics_train_800.npy"
VAL_DATA_PATH = DATA_DIR / "physics_val_200.npy"

# ---------------------------------------------------------------------------
# Model metadata (for display in model-status panel — not fed into model)
# ---------------------------------------------------------------------------

MODEL_PARAMS = 83_904
MODEL_BEST_EPOCH = 40
TRAIN_SAMPLES = 800
VAL_SAMPLES = 200
TEST_SAMPLES = 200
SPLIT_SEED = 2026

# Multi-seed reproducibility results (for display — read-only scientific record)
VALIDATED_RESULTS = {
    "analytical": {
        "nmse_db": -14.4545,
        "mean_phase_error_deg": 3.9201,
        "mean_gain_error_db": 1.0404,
        "mean_relative_error_pct": 15.1264,
    },
    "pinn": {
        "nmse_db": -7.7489,
        "mean_phase_error_deg": 16.1990,
        "mean_gain_error_db": 1.7084,
        "mean_relative_error_pct": 36.7887,
    },
    "multiseed": {
        "nmse_mean_db": -7.7402,
        "nmse_std_db": 0.0061,
        "phase_mean_deg": 16.2042,
        "phase_std_deg": 0.0139,
        "gain_mean_db": 1.7248,
        "gain_std_db": 0.0135,
        "relative_mean_pct": 36.6049,
        "relative_std_pct": 0.1305,
    },
}

# ---------------------------------------------------------------------------
# Physics constants — must match physics_core.py exactly
# ---------------------------------------------------------------------------

SELF_IMPEDANCE = 73.1 + 42.5j
NEAREST_NEIGHBOR_COUPLING = 0.12
DECAY = 0.55
PHASE_STEP_RAD = 0.35

# Default array — PINN checkpoint is trained for these exact parameters
DEFAULT_NUM_ELEMENTS = 64
DEFAULT_FREQUENCY_HZ = 28e9
DEFAULT_ELEMENT_SPACING_LAMBDA = 0.5

# Steering range used in training
MIN_ANGLE_DEG = -60.0
MAX_ANGLE_DEG = 60.0

# Angular grid for beam evaluation (matches evaluate_beam_metrics.py: 0.1° resolution)
BEAM_ANGLE_MIN = -90.0
BEAM_ANGLE_MAX = 90.0
BEAM_ANGLE_POINTS = 1801

# Robustness sweep levels
SNR_LEVELS = [40.0, 30.0, 20.0, 15.0, 10.0]
COUPLING_LEVELS = [0.0, 0.10, 0.20, 0.30, 0.50]
STEERING_REGIONS = [
    (-60.0, -40.0),
    (-40.0, -20.0),
    (-20.0, 0.0),
    (0.0, 20.0),
    (20.0, 40.0),
    (40.0, 60.0),
]
