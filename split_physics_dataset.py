"""
PINN-Cal — Controlled Dataset Split

Creates a deterministic:
    800 train
    200 validation
    200 test

split from data/physics_full.npy.

The original physics dataset files are preserved.

Outputs:
    data/physics_train_800.npy
    data/physics_val_200.npy
    data/physics_test_200.npy
"""

from __future__ import annotations

import os

import numpy as np


INPUT_PATH = "data/physics_full.npy"

TRAIN_SAMPLES = 800
VAL_SAMPLES = 200
TEST_SAMPLES = 200

SPLIT_SEED = 2026


def main() -> None:
    print("=" * 70)
    print("PINN-Cal — Controlled Dataset Split")
    print("=" * 70)

    data = np.load(
        INPUT_PATH,
        allow_pickle=True,
    ).item()

    X = data["X"]
    Y = data["Y"]
    angles = data["steering_angles"]

    total_samples = len(X)
    expected_total = (
        TRAIN_SAMPLES
        + VAL_SAMPLES
        + TEST_SAMPLES
    )

    if total_samples != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} samples, "
            f"found {total_samples}."
        )

    if len(Y) != total_samples:
        raise RuntimeError("X/Y sample counts do not match.")

    if len(angles) != total_samples:
        raise RuntimeError(
            "X/angles sample counts do not match."
        )

    if not np.isfinite(X).all():
        raise RuntimeError("X contains non-finite values.")

    if not np.isfinite(Y).all():
        raise RuntimeError("Y contains non-finite values.")

    if not np.isfinite(angles).all():
        raise RuntimeError(
            "Steering angles contain non-finite values."
        )

    # ---------------------------------------------------------------
    # Deterministic permutation.
    # ---------------------------------------------------------------
    rng = np.random.default_rng(SPLIT_SEED)

    indices = rng.permutation(total_samples)

    train_idx = indices[:TRAIN_SAMPLES]

    val_idx = indices[
        TRAIN_SAMPLES:
        TRAIN_SAMPLES + VAL_SAMPLES
    ]

    test_idx = indices[
        TRAIN_SAMPLES + VAL_SAMPLES:
    ]

    # ---------------------------------------------------------------
    # Safety checks: no overlap.
    # ---------------------------------------------------------------
    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    test_set = set(test_idx.tolist())

    if train_set & val_set:
        raise RuntimeError(
            "Train/validation overlap detected."
        )

    if train_set & test_set:
        raise RuntimeError(
            "Train/test overlap detected."
        )

    if val_set & test_set:
        raise RuntimeError(
            "Validation/test overlap detected."
        )

    if (
        len(train_set)
        + len(val_set)
        + len(test_set)
        != total_samples
    ):
        raise RuntimeError(
            "Split does not contain every sample exactly once."
        )

    train = {
        "X": X[train_idx],
        "Y": Y[train_idx],
        "steering_angles": angles[train_idx],
    }

    validation = {
        "X": X[val_idx],
        "Y": Y[val_idx],
        "steering_angles": angles[val_idx],
    }

    test = {
        "X": X[test_idx],
        "Y": Y[test_idx],
        "steering_angles": angles[test_idx],
    }

    os.makedirs("data", exist_ok=True)

    np.save(
        "data/physics_train_800.npy",
        train,
    )

    np.save(
        "data/physics_val_200.npy",
        validation,
    )

    np.save(
        "data/physics_test_200.npy",
        test,
    )

    print("\nSplit created successfully:")
    print(
        f"  Train      : {len(train['X'])} samples"
    )
    print(
        f"  Validation : {len(validation['X'])} samples"
    )
    print(
        f"  Test       : {len(test['X'])} samples"
    )

    print("\nFeature shapes:")
    print(
        f"  Train X: {train['X'].shape} | "
        f"Train Y: {train['Y'].shape}"
    )
    print(
        f"  Val   X: {validation['X'].shape} | "
        f"Val   Y: {validation['Y'].shape}"
    )
    print(
        f"  Test  X: {test['X'].shape} | "
        f"Test  Y: {test['Y'].shape}"
    )

    print("\nSteering-angle ranges:")

    for name, subset in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        print(
            f"  {name:10s}: "
            f"{subset['steering_angles'].min():8.3f}° "
            f"to "
            f"{subset['steering_angles'].max():8.3f}°"
        )

    print("\nSaved:")
    print("  data/physics_train_800.npy")
    print("  data/physics_val_200.npy")
    print("  data/physics_test_200.npy")

    print("\nSplit seed:", SPLIT_SEED)
    print("Original physics_full.npy was not modified.")


if __name__ == "__main__":
    main()