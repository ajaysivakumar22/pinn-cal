"""
PINN-Cal v4: Multi-SNR Dataset Generation
Generates a 2,500-sample dataset (2,000 train, 500 validation) with multi-SNR diversity:
SNR distributed uniformly over [5.0, 30.0] dB (covering 5dB, 10dB, 15dB, 20dB, 25dB, 30dB).

Hardware impairments:
- Phase error sigma: 20 deg
- Gain error sigma: 2.0 dB
- Coupling mismatch: 20%
- Steering angle range: [-60.0, 60.0] deg

All generated data is saved in experimental/v4/datasets/
Frozen benchmark data in data/ is strictly untouched.
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from calibration_dataset import DatasetConfig, generate_sample
from physics_core import DEFAULT_CONFIG

TOTAL_SAMPLES = 2500
TRAIN_SAMPLES = 2000
VAL_SAMPLES = 500
SEED = 202604  # Explicit distinct seed for v4 experimental dataset

OUTPUT_DIR = Path("experimental/v4/datasets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_v4_dataset():
    print("=" * 70)
    print("PINN-Cal v4 Multi-SNR Dataset Generation")
    print("=" * 70)
    print(f"Total samples : {TOTAL_SAMPLES} (Train: {TRAIN_SAMPLES}, Val: {VAL_SAMPLES})")
    print(f"SNR range     : 5.0 dB to 30.0 dB (continuous multi-SNR)")
    print(f"Phase sigma   : 20.0 deg")
    print(f"Gain sigma    : 2.0 dB")
    print(f"Coupling err  : 20% mismatch")
    print(f"Steering      : -60.0 to +60.0 deg")
    print(f"Generation seed: {SEED}")

    rng = np.random.default_rng(SEED)
    angles = rng.uniform(-60.0, 60.0, size=TOTAL_SAMPLES)
    snrs = rng.uniform(5.0, 30.0, size=TOTAL_SAMPLES)

    X_list = []
    Y_list = []

    for i in range(TOTAL_SAMPLES):
        sample_seed = SEED + i
        cfg = DatasetConfig(
            num_samples=1,
            phase_sigma_deg=20.0,
            gain_sigma_db=2.0,
            snr_db=float(snrs[i]),
            coupling_mismatch_fraction=0.20,
            min_angle_deg=-60.0,
            max_angle_deg=60.0,
            seed=sample_seed,
        )
        x, y = generate_sample(float(angles[i]), sample_seed, config=DEFAULT_CONFIG, dataset_config=cfg)
        X_list.append(x)
        Y_list.append(y)

        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{TOTAL_SAMPLES} samples...")

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    angles = np.array(angles, dtype=np.float64)
    snrs = np.array(snrs, dtype=np.float64)

    # Train / Val Split
    train_data = {
        "X": X[:TRAIN_SAMPLES],
        "Y": Y[:TRAIN_SAMPLES],
        "steering_angles": angles[:TRAIN_SAMPLES],
        "snr_db": snrs[:TRAIN_SAMPLES],
    }

    val_data = {
        "X": X[TRAIN_SAMPLES:],
        "Y": Y[TRAIN_SAMPLES:],
        "steering_angles": angles[TRAIN_SAMPLES:],
        "snr_db": snrs[TRAIN_SAMPLES:],
    }

    full_data = {
        "X": X,
        "Y": Y,
        "steering_angles": angles,
        "snr_db": snrs,
    }

    np.save(OUTPUT_DIR / "v4_train_2000.npy", train_data)
    np.save(OUTPUT_DIR / "v4_val_500.npy", val_data)
    np.save(OUTPUT_DIR / "v4_full_2500.npy", full_data)

    print("\nDataset successfully generated and saved:")
    print(f"  Train : {OUTPUT_DIR / 'v4_train_2000.npy'} (X: {train_data['X'].shape})")
    print(f"  Val   : {OUTPUT_DIR / 'v4_val_500.npy'} (X: {val_data['X'].shape})")
    print(f"  SNR range in train: [{train_data['snr_db'].min():.1f}, {train_data['snr_db'].max():.1f}] dB")
    print(f"  SNR range in val  : [{val_data['snr_db'].min():.1f}, {val_data['snr_db'].max():.1f}] dB")
    print("=" * 70)

if __name__ == "__main__":
    generate_v4_dataset()
