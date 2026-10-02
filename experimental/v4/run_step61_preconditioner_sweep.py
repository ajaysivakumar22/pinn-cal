"""
PINN-Cal v4: STEP 61 Tikhonov-Regularized Physics Preconditioner Sweep
Evaluates:
- delta = 0.00 (Unregularized control)
- delta = 0.025 (0.25 * delta_noise)
- delta = 0.050 (0.50 * delta_noise)
- delta = 0.100 (1.00 * delta_noise)
- delta = 0.150 (1.50 * delta_noise)
- delta = 0.200 (2.00 * delta_noise)
- delta = 0.300 (3.00 * delta_noise)

Evaluates on:
1. Frozen nominal 20 dB benchmark (data/physics_test_200.npy)
2. 10 dB SNR
3. 5 dB SNR
4. Phase 35 deg stress
5. Gain 4.0 dB stress
6. Coupling 40% stress
7. Multi-SNR dataset distribution analysis (experimental/v4/datasets/v4_train_2000.npy)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from physics_core import DEFAULT_CONFIG, steering_vector, mutual_impedance_matrix
from analytical_baseline import (
    nmse_db,
    phase_error_deg,
    gain_error_db,
    relative_error,
)
from calibration_dataset import DatasetConfig, generate_sample
from experimental.v4.train_and_evaluate_v4 import (
    compute_beam_metrics,
    evaluate_full_metrics,
    FROZEN_TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
    CONFIG,
    M,
)

DELTA_NOISE = 0.10  # Derived from nominal 20 dB SNR noise variance on V_hw
DELTAS = [0.00, 0.025, 0.050, 0.100, 0.150, 0.200, 0.300]

def compute_tikhonov_prior(X_np, angles_deg, delta: float, config=CONFIG):
    I_obs = torch.complex(torch.tensor(X_np[:, :M], dtype=torch.float64), torch.tensor(X_np[:, M:2*M], dtype=torch.float64))
    Z = mutual_impedance_matrix(config).to(torch.complex128)
    Z_self = Z.diagonal().mean()
    V_eff = I_obs @ Z.T
    V_hw = V_eff / Z_self

    angles_t = torch.tensor(angles_deg, dtype=torch.float64)
    V_ideal = steering_vector(angles_t, config).to(torch.complex128)

    if delta == 0.0:
        C_prior = V_ideal / (V_hw + 1e-12)
    else:
        denom = torch.abs(V_hw)**2 + delta**2
        C_prior = (V_ideal * torch.conj(V_hw)) / denom

    return C_prior.numpy(), V_hw.numpy()

def generate_test_split(overrides: dict, num_samples: int = 100, base_seed: int = 888000):
    rng = np.random.default_rng(base_seed + int(overrides.get("snr_db", 0)) + int(overrides.get("phase_sigma_deg", 0)))
    X_s, Y_s, ang_s = [], [], []

    for i in range(num_samples):
        sample_seed = base_seed + i
        angle = float(rng.uniform(-60.0, 60.0))
        cfg = DatasetConfig(
            num_samples=1,
            phase_sigma_deg=overrides.get("phase_sigma_deg", 20.0),
            gain_sigma_db=overrides.get("gain_sigma_db", 2.0),
            snr_db=overrides.get("snr_db", 20.0),
            coupling_mismatch_fraction=overrides.get("coupling_mismatch_fraction", 0.20),
            min_angle_deg=-60.0,
            max_angle_deg=60.0,
            seed=sample_seed,
        )
        x, y = generate_sample(angle, sample_seed, config=CONFIG, dataset_config=cfg)
        X_s.append(x)
        Y_s.append(y)
        ang_s.append(angle)

    return (
        np.array(X_s, dtype=np.float32),
        np.array(Y_s, dtype=np.float32),
        np.array(ang_s, dtype=np.float64),
    )

def analyze_stability_distribution(delta_values, train_path=TRAIN_PATH):
    train_data = np.load(train_path, allow_pickle=True).item()
    X = train_data["X"]
    Y = train_data["Y"]
    angles = train_data["steering_angles"]
    snrs = train_data["snr_db"]
    C_true = Y[:, :64] + 1j * Y[:, 64:]

    # First inspect V_hw
    _, V_hw = compute_tikhonov_prior(X, angles, delta=0.0)
    v_mag = np.abs(V_hw).flatten()

    v_hw_stats = {
        "min": float(np.min(v_mag)),
        "mean": float(np.mean(v_mag)),
        "median": float(np.median(v_mag)),
        "p001": float(np.percentile(v_mag, 0.1)),
        "p01": float(np.percentile(v_mag, 1.0)),
        "p05": float(np.percentile(v_mag, 5.0)),
        "near_zero_001_count": int(np.sum(v_mag < 0.01)),
        "near_zero_005_count": int(np.sum(v_mag < 0.05)),
        "near_zero_010_count": int(np.sum(v_mag < 0.10)),
    }

    # Now for each delta, analyze C_prior distribution and sample MSEs
    c_stats = {}
    for d in delta_values:
        C_prior, _ = compute_tikhonov_prior(X, angles, delta=d)
        c_mag = np.abs(C_prior).flatten()
        sample_err = np.mean(np.abs(C_prior - C_true)**2, axis=1) / 2.0  # real/imag MSE

        c_stats[f"delta_{d:.3f}"] = {
            "delta": d,
            "max_c": float(np.max(c_mag)),
            "p95_c": float(np.percentile(c_mag, 95.0)),
            "p99_c": float(np.percentile(c_mag, 99.0)),
            "p999_c": float(np.percentile(c_mag, 99.9)),
            "mean_sample_mse": float(np.mean(sample_err)),
            "median_sample_mse": float(np.median(sample_err)),
            "p95_sample_mse": float(np.percentile(sample_err, 95.0)),
            "p99_sample_mse": float(np.percentile(sample_err, 99.0)),
            "max_sample_mse": float(np.max(sample_err)),
        }

    return v_hw_stats, c_stats

def main():
    print("=" * 85)
    print("PINN-CAL v4: STEP 61 TIKHONOV-REGULARIZED PRECONDITIONER EVALUATION")
    print("=" * 85)

    # 1. Numerical Stability & Distribution Analysis on Multi-SNR Train Set
    print("\n--- Part B: Numerical Stability & Singularity Analysis on Multi-SNR Dataset ---")
    v_stats, c_stats = analyze_stability_distribution(DELTAS)
    print(f"|V_hw| Distribution across {2000*64} elements:")
    print(f"  min={v_stats['min']:.6e} | p0.1%={v_stats['p001']:.4f} | p1%={v_stats['p01']:.4f} | median={v_stats['median']:.4f} | mean={v_stats['mean']:.4f}")
    print(f"  Near-zero count (<0.01): {v_stats['near_zero_001_count']} | (<0.05): {v_stats['near_zero_005_count']} | (<0.10): {v_stats['near_zero_010_count']}")

    print("\n|C_prior| and Sample MSE by delta:")
    print(f"  {'delta':8s} | {'max |C|':10s} | {'p99.9%':8s} | {'p99%':8s} | {'Mean MSE':10s} | {'Median MSE':10s} | {'Max MSE':10s}")
    print("  " + "-" * 75)
    for d in DELTAS:
        cs = c_stats[f"delta_{d:.3f}"]
        print(f"  {cs['delta']:8.3f} | {cs['max_c']:10.2f} | {cs['p999_c']:8.2f} | {cs['p99_c']:8.2f} | {cs['mean_sample_mse']:10.4f} | {cs['median_sample_mse']:10.4f} | {cs['max_sample_mse']:10.2f}")

    # 2. Benchmark & Stress Test Evaluations
    conditions = [
        ("Frozen 20 dB", None),  # Loaded from frozen test set
        ("10 dB SNR", {"snr_db": 10.0}),
        ("5 dB SNR", {"snr_db": 5.0}),
        ("Phase 35 deg", {"phase_sigma_deg": 35.0}),
        ("Gain 4.0 dB", {"gain_sigma_db": 4.0}),
        ("Coupling 40%", {"coupling_mismatch_fraction": 0.40}),
    ]

    datasets = {}
    frozen_test = np.load(FROZEN_TEST_PATH, allow_pickle=True).item()
    datasets["Frozen 20 dB"] = (
        frozen_test["X"],
        frozen_test["Y"],
        frozen_test["steering_angles"],
    )

    for cond_name, overrides in conditions[1:]:
        datasets[cond_name] = generate_test_split(overrides, num_samples=100)

    # Evaluate each delta across conditions
    all_eval_results = {}
    for d in DELTAS:
        all_eval_results[f"delta_{d:.3f}"] = {}
        for cond_name, (X, Y, ang) in datasets.items():
            C_prior, _ = compute_tikhonov_prior(X, ang, delta=d)
            C_true = Y[:, :64] + 1j * Y[:, 64:]
            metrics = evaluate_full_metrics(C_prior, C_true, ang)
            all_eval_results[f"delta_{d:.3f}"][cond_name] = metrics

    # 3. Print Preconditioner Quality & Nominal Preservation
    print("\n" + "=" * 85)
    print("PART D & E: PRECONDITIONER QUALITY AND NOMINAL PRESERVATION")
    print("=" * 85)

    for cond_name in datasets:
        print(f"\n--- Condition: {cond_name} ---")
        print(f"  {'delta':8s} | {'NMSE (dB)':10s} | {'Phase (deg)':12s} | {'Gain (dB)':10s} | {'Rel Err (%)':12s} | {'Point (deg)':12s} | {'Beam Corr':10s}")
        print("  " + "-" * 80)
        for d in DELTAS:
            m = all_eval_results[f"delta_{d:.3f}"][cond_name]
            print(f"  {d:8.3f} | {m['nmse_db']:10.2f} | {m['phase_error_deg']:12.2f} | {m['gain_error_db']:10.2f} | {m['relative_error']:12.2f} | {m['pointing_error_deg']:12.4f} | {m['beam_correlation']:10.6f}")

    # 4. Detailed Nominal Preservation Analysis (delta = 0 vs delta > 0 at Frozen 20 dB)
    print("\n" + "=" * 85)
    print("PART E: DETAILED NOMINAL PRESERVATION ANALYSIS AT FROZEN 20 dB BENCHMARK")
    print("=" * 85)
    base_m = all_eval_results["delta_0.000"]["Frozen 20 dB"]
    print(f"Baseline (delta=0): NMSE={base_m['nmse_db']:.4f} dB | Phase={base_m['phase_error_deg']:.4f} deg | Gain={base_m['gain_error_db']:.4f} dB | Point={base_m['pointing_error_deg']:.4f} deg | Corr={base_m['beam_correlation']:.6f}")
    print("\nDifferences relative to delta=0 (Positive NMSE diff = degradation, Negative = improvement):")
    print(f"  {'delta':8s} | {'Delta NMSE':12s} | {'Delta Phase':12s} | {'Delta Gain':12s} | {'Delta Point':12s} | {'Delta Corr':12s}")
    print("  " + "-" * 70)
    for d in DELTAS[1:]:
        m = all_eval_results[f"delta_{d:.3f}"]["Frozen 20 dB"]
        d_nmse = m['nmse_db'] - base_m['nmse_db']
        d_phase = m['phase_error_deg'] - base_m['phase_error_deg']
        d_gain = m['gain_error_db'] - base_m['gain_error_db']
        d_point = m['pointing_error_deg'] - base_m['pointing_error_deg']
        d_corr = m['beam_correlation'] - base_m['beam_correlation']
        print(f"  {d:8.3f} | {d_nmse:+12.4f} | {d_phase:+12.4f} | {d_gain:+12.4f} | {d_point:+12.4f} | {d_corr:+12.6f}")

    # 5. Save all results
    results_path = Path("experimental/v4/results/step61_preconditioner_sweep.npy")
    np.save(results_path, {
        "v_hw_stats": v_stats,
        "stability_stats": c_stats,
        "eval_results": all_eval_results,
        "deltas": DELTAS,
        "delta_noise": DELTA_NOISE,
    })
    print(f"\nStep 61 results successfully saved to: {results_path}")

if __name__ == "__main__":
    main()
