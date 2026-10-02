"""
STEP 59: Reproduction and Low-SNR Evaluation across 3 Seeds (42, 123, 456)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from physics_core import DEFAULT_CONFIG
from analytical_baseline import (
    analytical_correction,
    nmse_db,
    phase_error_deg,
    gain_error_db,
    relative_error,
)
from calibration_dataset import DatasetConfig, generate_sample
from experimental.v4.pinn_v4_model import PINNCalV4, CONFIG, M
from experimental.v4.train_and_evaluate_v4 import compute_beam_metrics, evaluate_full_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FROZEN_TEST_PATH = Path("data/physics_test_200.npy")

def evaluate_seed_frozen(model_path: str, seed: int):
    # Load model
    model = PINNCalV4(bounded=True, alpha=0.5).to(DEVICE)
    weights = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(weights)
    model.eval()

    # Load history
    hist_path = Path(f"experimental/v4/histories/pinn_v4_bounded_seed{seed}_history.npy")
    hist = np.load(hist_path, allow_pickle=True).item()
    best_val_loss = float(np.min(hist["val_loss"]))
    best_ep = int(np.argmin(hist["val_loss"])) + 1
    train_loss = float(hist["train_loss"][best_ep - 1])

    # Frozen test set
    frozen_test = np.load(FROZEN_TEST_PATH, allow_pickle=True).item()
    X_frozen = frozen_test["X"]
    Y_frozen = frozen_test["Y"]
    angles_frozen = frozen_test["steering_angles"]
    C_true_frozen = Y_frozen[:, :64] + 1j * Y_frozen[:, 64:]

    with torch.no_grad():
        c_hat_t, delta_c_t, x_prior_t = model(torch.tensor(X_frozen, dtype=torch.float32, device=DEVICE))
        c_hat = c_hat_t.cpu().numpy()

    metrics = evaluate_full_metrics(c_hat, C_true_frozen, angles_frozen)

    return {
        "seed": seed,
        "best_epoch": best_ep,
        "train_loss": train_loss,
        "val_loss": best_val_loss,
        "nmse_db": metrics["nmse_db"],
        "phase_error_deg": metrics["phase_error_deg"],
        "gain_error_db": metrics["gain_error_db"],
        "relative_error": metrics["relative_error"],
        "pointing_error_deg": metrics["pointing_error_deg"],
        "beam_correlation": metrics["beam_correlation"],
        "model": model,
    }

def generate_snr_testset(snr_db: float, num_samples: int = 100, base_seed: int = 999000):
    rng = np.random.default_rng(base_seed + int(snr_db * 10))
    X_list, Y_list, ang_list = [], [], []

    for i in range(num_samples):
        sample_seed = base_seed + i
        angle = float(rng.uniform(-60.0, 60.0))
        cfg = DatasetConfig(
            num_samples=1,
            phase_sigma_deg=20.0,
            gain_sigma_db=2.0,
            snr_db=snr_db,
            coupling_mismatch_fraction=0.20,
            min_angle_deg=-60.0,
            max_angle_deg=60.0,
            seed=sample_seed,
        )
        x, y = generate_sample(angle, sample_seed, config=CONFIG, dataset_config=cfg)
        X_list.append(x)
        Y_list.append(y)
        ang_list.append(angle)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(Y_list, dtype=np.float32),
        np.array(ang_list, dtype=np.float64),
    )

def evaluate_snr_condition(models: dict, snr_db: float):
    X, Y, ang = generate_snr_testset(snr_db=snr_db, num_samples=100)
    C_true = Y[:, :64] + 1j * Y[:, 64:]

    # Analytical baseline
    c_ana = analytical_correction(X[:, :64] + 1j * X[:, 64:128], ang)
    m_ana = evaluate_full_metrics(c_ana, C_true, ang)

    results = {
        "analytical": {
            "nmse_db": m_ana["nmse_db"],
            "phase_error_deg": m_ana["phase_error_deg"],
            "pointing_error_deg": m_ana["pointing_error_deg"],
            "beam_correlation": m_ana["beam_correlation"],
        }
    }

    for seed, model in models.items():
        with torch.no_grad():
            c_hat_t, _, _ = model(torch.tensor(X, dtype=torch.float32, device=DEVICE))
            c_hat = c_hat_t.cpu().numpy()
        m_v4 = evaluate_full_metrics(c_hat, C_true, ang)
        results[f"v4_seed{seed}"] = {
            "nmse_db": m_v4["nmse_db"],
            "phase_error_deg": m_v4["phase_error_deg"],
            "pointing_error_deg": m_v4["pointing_error_deg"],
            "beam_correlation": m_v4["beam_correlation"],
        }

    return results

def main():
    print("=" * 80)
    print("STEP 59: PART B — 3-SEED REPRODUCTION (SEEDS 42, 123, 456)")
    print("=" * 80)

    seeds = [42, 123, 456]
    results_b = []
    models = {}

    for s in seeds:
        res = evaluate_seed_frozen(f"experimental/v4/models/pinn_v4_bounded_seed{s}.pth", s)
        results_b.append(res)
        models[s] = res["model"]
        print(f"Seed {s:3d} | Best Ep {res['best_epoch']:2d} | TrainLoss={res['train_loss']:.6f} | ValLoss={res['val_loss']:.6f} | "
              f"NMSE={res['nmse_db']:.2f} dB | PhaseErr={res['phase_error_deg']:.2f} deg | GainErr={res['gain_error_db']:.2f} dB | "
              f"RelErr={res['relative_error']:.2f}% | PointErr={res['pointing_error_deg']:.4f} deg | BeamCorr={res['beam_correlation']:.6f}")

    # Calculate Mean +/- Std
    keys = ["best_epoch", "train_loss", "val_loss", "nmse_db", "phase_error_deg", "gain_error_db", "relative_error", "pointing_error_deg", "beam_correlation"]
    stats = {}
    for k in keys:
        vals = [r[k] for r in results_b]
        stats[k] = (float(np.mean(vals)), float(np.std(vals)))

    print("\n--- Summary Statistics (Mean +/- Std) ---")
    for k in keys:
        m, s = stats[k]
        print(f"  {k:20s}: {m:.6f} +/- {s:.6f}")

    print("\n" + "=" * 80)
    print("STEP 59: PART C — LOW-SNR CHECK (10 dB & 5 dB SNR)")
    print("=" * 80)

    res_10db = evaluate_snr_condition(models, snr_db=10.0)
    res_5db = evaluate_snr_condition(models, snr_db=5.0)

    print("\n[Condition: 10 dB SNR]")
    print(f"  Analytical Baseline : NMSE={res_10db['analytical']['nmse_db']:6.2f} dB | Phase={res_10db['analytical']['phase_error_deg']:5.2f} deg | Point={res_10db['analytical']['pointing_error_deg']:.4f} deg | Corr={res_10db['analytical']['beam_correlation']:.6f}")
    for s in seeds:
        m = res_10db[f"v4_seed{s}"]
        print(f"  V4 (Seed {s:3d})       : NMSE={m['nmse_db']:6.2f} dB | Phase={m['phase_error_deg']:5.2f} deg | Point={m['pointing_error_deg']:.4f} deg | Corr={m['beam_correlation']:.6f}")

    # Aggregate 10 dB across seeds
    v4_10_nmse = [res_10db[f"v4_seed{s}"]['nmse_db'] for s in seeds]
    v4_10_phase = [res_10db[f"v4_seed{s}"]['phase_error_deg'] for s in seeds]
    v4_10_point = [res_10db[f"v4_seed{s}"]['pointing_error_deg'] for s in seeds]
    v4_10_corr = [res_10db[f"v4_seed{s}"]['beam_correlation'] for s in seeds]
    print(f"  V4 10 dB Mean+/-Std : NMSE={np.mean(v4_10_nmse):.2f}+/-{np.std(v4_10_nmse):.2f} dB | Phase={np.mean(v4_10_phase):.2f}+/-{np.std(v4_10_phase):.2f} deg | Point={np.mean(v4_10_point):.4f}+/-{np.std(v4_10_point):.4f} deg | Corr={np.mean(v4_10_corr):.6f}+/-{np.std(v4_10_corr):.6f}")

    print("\n[Condition: 5 dB SNR]")
    print(f"  Analytical Baseline : NMSE={res_5db['analytical']['nmse_db']:6.2f} dB | Phase={res_5db['analytical']['phase_error_deg']:5.2f} deg | Point={res_5db['analytical']['pointing_error_deg']:.4f} deg | Corr={res_5db['analytical']['beam_correlation']:.6f}")
    for s in seeds:
        m = res_5db[f"v4_seed{s}"]
        print(f"  V4 (Seed {s:3d})       : NMSE={m['nmse_db']:6.2f} dB | Phase={m['phase_error_deg']:5.2f} deg | Point={m['pointing_error_deg']:.4f} deg | Corr={m['beam_correlation']:.6f}")

    # Aggregate 5 dB across seeds
    v4_5_nmse = [res_5db[f"v4_seed{s}"]['nmse_db'] for s in seeds]
    v4_5_phase = [res_5db[f"v4_seed{s}"]['phase_error_deg'] for s in seeds]
    v4_5_point = [res_5db[f"v4_seed{s}"]['pointing_error_deg'] for s in seeds]
    v4_5_corr = [res_5db[f"v4_seed{s}"]['beam_correlation'] for s in seeds]
    print(f"  V4 5 dB Mean+/-Std  : NMSE={np.mean(v4_5_nmse):.2f}+/-{np.std(v4_5_nmse):.2f} dB | Phase={np.mean(v4_5_phase):.2f}+/-{np.std(v4_5_phase):.2f} deg | Point={np.mean(v4_5_point):.4f}+/-{np.std(v4_5_point):.4f} deg | Corr={np.mean(v4_5_corr):.6f}+/-{np.std(v4_5_corr):.6f}")

    # Save all Step 59 reproduction results
    step59_data = {
        "part_b_frozen_seeds": results_b,
        "part_b_stats": stats,
        "part_c_10db": res_10db,
        "part_c_5db": res_5db,
    }
    out_file = Path("experimental/v4/results/step59_reproduction_results.npy")
    np.save(out_file, step59_data)
    print(f"\nStep 59 results successfully saved to: {out_file}")

if __name__ == "__main__":
    main()
