"""
PINN-Cal — Generate Physically Grounded Dataset

Creates a reproducible train/test dataset for the rebuilt PINN-Cal system.

The generated files are kept separate from the legacy dataset:
    data/physics_train.npy
    data/physics_test.npy
    data/physics_full.npy

Configuration:
    64-element ULA
    28 GHz
    20 degree phase-error sigma
    2 dB gain-error sigma
    steering range: -60 to +60 degrees
"""

from __future__ import annotations

import os

import numpy as np

from calibration_dataset import (
    DatasetConfig,
    generate_dataset,
)


TOTAL_SAMPLES = 1200
TRAIN_SAMPLES = 1000
TEST_SAMPLES = TOTAL_SAMPLES - TRAIN_SAMPLES

SEED = 42

DATASET_CONFIG = DatasetConfig(
    num_samples=TOTAL_SAMPLES,
    phase_sigma_deg=20.0,
    gain_sigma_db=2.0,
    min_angle_deg=-60.0,
    max_angle_deg=60.0,
    seed=SEED,
)


def main() -> None:
    print("=" * 70)
    print("PINN-Cal — Physically Grounded Dataset Generation")
    print("=" * 70)

    print(f"Total samples : {TOTAL_SAMPLES}")
    print(f"Train samples : {TRAIN_SAMPLES}")
    print(f"Test samples  : {TEST_SAMPLES}")
    print(f"Phase sigma   : {DATASET_CONFIG.phase_sigma_deg} deg")
    print(f"Gain sigma    : {DATASET_CONFIG.gain_sigma_db} dB")
    print(
        f"Angle range   : "
        f"{DATASET_CONFIG.min_angle_deg} to "
        f"{DATASET_CONFIG.max_angle_deg} deg"
    )
    print(f"Seed          : {SEED}")
    print()

    data = generate_dataset(DATASET_CONFIG)

    X = data["X"]
    Y = data["Y"]
    angles = data["steering_angles"]

    if X.shape[0] != TOTAL_SAMPLES:
        raise RuntimeError(
            f"Expected {TOTAL_SAMPLES} samples, got {X.shape[0]}"
        )

    if Y.shape[0] != TOTAL_SAMPLES:
        raise RuntimeError(
            f"Expected {TOTAL_SAMPLES} target samples, got {Y.shape[0]}"
        )

    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite values detected in X.")

    if not np.isfinite(Y).all():
        raise RuntimeError("Non-finite values detected in Y.")

    train = {
        "X": X[:TRAIN_SAMPLES],
        "Y": Y[:TRAIN_SAMPLES],
        "steering_angles": angles[:TRAIN_SAMPLES],
    }

    test = {
        "X": X[TRAIN_SAMPLES:],
        "Y": Y[TRAIN_SAMPLES:],
        "steering_angles": angles[TRAIN_SAMPLES:],
    }

    full = {
        "X": X,
        "Y": Y,
        "steering_angles": angles,
    }

    os.makedirs("data", exist_ok=True)

    np.save("data/physics_train.npy", train)
    np.save("data/physics_test.npy", test)
    np.save("data/physics_full.npy", full)

    print("\nDataset shapes:")
    print(f"  Train X: {train['X'].shape}")
    print(f"  Train Y: {train['Y'].shape}")
    print(f"  Test  X: {test['X'].shape}")
    print(f"  Test  Y: {test['Y'].shape}")

    print("\nSaved:")
    print("  data/physics_train.npy")
    print("  data/physics_test.npy")
    print("  data/physics_full.npy")

    print("\nDataset generation complete.")


if __name__ == "__main__":
    main()