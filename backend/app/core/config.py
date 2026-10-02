"""
PINN-Cal Backend — Application Configuration

All paths are resolved relative to the repository root (D:/pinn-cal),
which is the working directory when running the backend.

Environment information is detected dynamically at runtime.
Do not hard-code Python version, PyTorch version, or CUDA state.
"""

from __future__ import annotations

from pathlib import Path

import os

# ---------------------------------------------------------------------------
# Repository root — parents[3] from backend/app/core/config.py
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Scientific core paths & Frozen V4 Reference Checkpoint
# ---------------------------------------------------------------------------

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
V4_MODELS_DIR = REPO_ROOT / "experimental" / "v4" / "models"

# Default to frozen V4 reference checkpoint (delta=0.15, alpha=0.50, seed=42)
DEFAULT_V4_CHECKPOINT = V4_MODELS_DIR / "pinn_v4_tikhonov015_seed42.pth"
_configured_ckpt = os.getenv("PINN_CHECKPOINT_PATH")
PINN_BEST_CHECKPOINT = Path(_configured_ckpt) if _configured_ckpt else DEFAULT_V4_CHECKPOINT
PINN_LAST_CHECKPOINT = DEFAULT_V4_CHECKPOINT

# Legacy V3 checkpoint retained for comparisons
PINN_V3_CHECKPOINT = MODELS_DIR / "pinn_cal_final_best.pth"

TEST_DATA_PATH = REPO_ROOT / "experimental" / "v4" / "results" / "blind_final_holdout_500.npy"
TRAIN_DATA_PATH = REPO_ROOT / "experimental" / "v4" / "datasets" / "v4_train_2000.npy"
VAL_DATA_PATH = REPO_ROOT / "experimental" / "v4" / "datasets" / "v4_val_500.npy"

# ---------------------------------------------------------------------------
# Model metadata (for display in model-status panel — not fed into model)
# ---------------------------------------------------------------------------

MODEL_NAME = "PINN-Cal V4 Reference Model"
MODEL_VERSION = "4.0.0-frozen"
MODEL_PARAMS = 33_088
MODEL_BEST_EPOCH = 29
TRAIN_SAMPLES = 2000
VAL_SAMPLES = 500
TEST_SAMPLES = 5000  # 10 conditions x 500 samples
SPLIT_SEED = 42

TIKHONOV_DELTA = 0.15
RESIDUAL_ALPHA = 0.50

# Validated headline benchmark results (from Step 64 blind holdout)
VALIDATED_RESULTS = {
    "analytical": {
        "nmse_db": -14.23,
        "mean_phase_error_deg": 3.96,
        "mean_gain_error_db": 1.07,
        "mean_relative_error_pct": 15.13,
    },
    "pinn_v3": {
        "nmse_db": -7.81,
        "mean_phase_error_deg": 16.16,
        "mean_gain_error_db": 1.68,
        "mean_relative_error_pct": 36.79,
    },
    "pinn_v4": {
        "nmse_db": -14.81,
        "mean_phase_error_deg": 4.47,
        "mean_gain_error_db": 1.00,
        "mean_relative_error_pct": 15.34,
        "pointing_error_deg": 0.0253,
        "beam_correlation": 0.99979,
    },
    "pinn": {
        "nmse_db": -14.81,
        "mean_phase_error_deg": 4.47,
        "mean_gain_error_db": 1.00,
        "mean_relative_error_pct": 15.34,
    },
    "multiseed": {
        "nmse_mean_db": -14.81,
        "nmse_std_db": 0.001,
        "phase_mean_deg": 4.47,
        "phase_std_deg": 0.02,
        "gain_mean_db": 1.00,
        "gain_std_db": 0.01,
        "relative_mean_pct": 15.34,
        "relative_std_pct": 0.05,
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
