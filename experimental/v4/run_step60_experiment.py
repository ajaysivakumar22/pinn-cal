"""
PINN-Cal v4: STEP 60 Controlled SNR-Aware Residual-Bound Experiment
Evaluates:
- Candidate 1: Fixed alpha = 0.50 (Control)
- Candidate 2: Mild adaptive alpha in [0.30, 0.80]
- Candidate 3: Stronger adaptive alpha in [0.25, 1.00]
Across seeds: 42, 123, 456
"""
import sys
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from physics_core import DEFAULT_CONFIG, steering_vector
from analytical_baseline import (
    analytical_correction,
    nmse_db,
    phase_error_deg,
    gain_error_db,
    relative_error,
)
from evaluate_beam_metrics import calculate_patterns, evaluate_patterns
from calibration_dataset import DatasetConfig, generate_sample
from experimental.v4.pinn_v4_model import PINNCalV4, CONFIG, M
from experimental.v4.train_and_evaluate_v4 import (
    compute_beam_metrics,
    evaluate_full_metrics,
    TRAIN_PATH,
    VAL_PATH,
    FROZEN_TEST_PATH,
    DEVICE,
)

def train_candidate_model(
    name: str,
    alpha_mode: str = "fixed",
    alpha: float = 0.5,
    alpha_bounds: tuple[float, float] | None = None,
    seed: int = 42,
    epochs: int = 100,
    patience: int = 15,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
):
    print(f"\n--- Training {name} (Seed {seed}, Mode={alpha_mode}, Bounds={alpha_bounds or (alpha, alpha)}) ---")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = np.load(TRAIN_PATH, allow_pickle=True).item()
    val_data = np.load(VAL_PATH, allow_pickle=True).item()

    X_tr = torch.tensor(train_data["X"], dtype=torch.float32)
    Y_tr = torch.tensor(train_data["Y"], dtype=torch.float32)

    X_va = torch.tensor(val_data["X"], dtype=torch.float32)
    Y_va = torch.tensor(val_data["Y"], dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, Y_tr), batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_va, Y_va), batch_size=32, shuffle=False
    )

    model = PINNCalV4(
        bounded=True,
        alpha=alpha,
        alpha_mode=alpha_mode,
        alpha_bounds=alpha_bounds,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = -1
    no_improve_epochs = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "gen_gap": [],
    }

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        b_losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            c_hat, delta_c_real, x_prior = model(xb)

            c_hat_real = torch.cat([c_hat.real.to(torch.float32), c_hat.imag.to(torch.float32)], dim=-1)
            loss = criterion(c_hat_real, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            b_losses.append(loss.item())

        train_loss = float(np.mean(b_losses))

        # Evaluate validation
        model.eval()
        v_losses = []
        with torch.no_grad():
            for xvb, yvb in val_loader:
                xvb = xvb.to(DEVICE)
                yvb = yvb.to(DEVICE)
                c_hat_v, _, _ = model(xvb)
                c_hat_v_real = torch.cat([c_hat_v.real.to(torch.float32), c_hat_v.imag.to(torch.float32)], dim=-1)
                v_losses.append(criterion(c_hat_v_real, yvb).item())

        val_loss = float(np.mean(v_losses))
        gen_gap = val_loss - train_loss

        history["epoch"].append(ep)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["gen_gap"].append(gen_gap)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ep
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"  Early stopping at epoch {ep} (Best Ep {best_epoch}, ValLoss={best_val_loss:.6e})")
                break

    elapsed = time.time() - t0
    ckpt_path = Path(f"experimental/v4/models/{name}.pth")
    torch.save(best_weights, ckpt_path)
    np.save(f"experimental/v4/histories/{name}_history.npy", history)

    print(f"  Finished in {elapsed:.1f}s | Best Ep {best_epoch:2d}/{epochs} | ValLoss={best_val_loss:.6e} | TrainLoss={history['train_loss'][best_epoch-1]:.6e}")

    model.load_state_dict(best_weights)
    return model, {
        "name": name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_loss_at_best": history["train_loss"][best_epoch - 1],
        "gen_gap": history["gen_gap"][best_epoch - 1],
        "elapsed_seconds": elapsed,
        "checkpoint": str(ckpt_path),
    }

def compute_residual_and_saturation_stats(model, X_np):
    model.eval()
    with torch.no_grad():
        c_hat, delta_c_real, _, eff_alpha = model(
            torch.tensor(X_np, dtype=torch.float32, device=DEVICE),
            return_alpha=True,
        )
        delta_c = delta_c_real.cpu().numpy()
        eff_alpha_np = eff_alpha.cpu().numpy() if isinstance(eff_alpha, torch.Tensor) else float(eff_alpha)

    # Complex magnitude of residual per element
    delta_c_complex = delta_c[:, :64] + 1j * delta_c[:, 64:]
    res_mag = np.abs(delta_c_complex)

    # Saturation: element real or imag residual reaching >= 95% of local alpha bound
    if isinstance(eff_alpha_np, np.ndarray) and eff_alpha_np.ndim > 0:
        alpha_bound = eff_alpha_np  # (B, 1)
    else:
        alpha_bound = float(eff_alpha_np)

    # Saturation threshold: >= 0.95 * alpha
    sat_mask = np.abs(delta_c) >= (0.95 * alpha_bound)
    sat_percentage = 100.0 * float(np.mean(sat_mask))

    return {
        "mean_residual_mag": float(np.mean(res_mag)),
        "max_residual_mag": float(np.max(res_mag)),
        "mean_alpha": float(np.mean(eff_alpha_np)),
        "min_alpha": float(np.min(eff_alpha_np)),
        "max_alpha": float(np.max(eff_alpha_np)),
        "saturation_pct": sat_percentage,
    }

def generate_eval_dataset(condition_name: str, overrides: dict, num_samples: int = 100, base_seed: int = 888000):
    rng = np.random.default_rng(base_seed + int(overrides.get("snr_db", 0)) + int(overrides.get("phase_sigma_deg", 0)))
    X_s, Y_s, ang_s = [], [], []

    min_ang = overrides.get("min_angle_deg", -60.0)
    max_ang = overrides.get("max_angle_deg", 60.0)

    for i in range(num_samples):
        sample_seed = base_seed + i
        angle = float(rng.uniform(min_ang, max_ang))
        cfg = DatasetConfig(
            num_samples=1,
            phase_sigma_deg=overrides.get("phase_sigma_deg", 20.0),
            gain_sigma_db=overrides.get("gain_sigma_db", 2.0),
            snr_db=overrides.get("snr_db", 20.0),
            coupling_mismatch_fraction=overrides.get("coupling_mismatch_fraction", 0.20),
            min_angle_deg=min_ang,
            max_angle_deg=max_ang,
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

def evaluate_model_on_dataset(model, X, Y, ang):
    model.eval()
    C_true = Y[:, :64] + 1j * Y[:, 64:]
    with torch.no_grad():
        c_hat_t, _, _ = model(torch.tensor(X, dtype=torch.float32, device=DEVICE))
        c_hat = c_hat_t.cpu().numpy()
    return evaluate_full_metrics(c_hat, C_true, ang)

def main():
    print("=" * 80)
    print("PINN-CAL v4: STEP 60 CONTROLLED SNR-AWARE RESIDUAL EXPERIMENT")
    print("=" * 80)

    seeds = [42, 123, 456]

    candidates = {
        "Candidate 1 (Fixed alpha=0.50)": {
            "mode": "fixed",
            "alpha": 0.5,
            "bounds": (0.5, 0.5),
            "prefix": "pinn_v4_bounded",  # Pre-existing from Step 59
            "retrain": False,
        },
        "Candidate 2 (Mild [0.30, 0.80])": {
            "mode": "mild",
            "alpha": 0.5,
            "bounds": (0.30, 0.80),
            "prefix": "pinn_v4_mild_adaptive",
            "retrain": True,
        },
        "Candidate 3 (Strong [0.25, 1.00])": {
            "mode": "strong",
            "alpha": 0.5,
            "bounds": (0.25, 1.00),
            "prefix": "pinn_v4_strong_adaptive",
            "retrain": True,
        },
    }

    # 1. Train or load models
    trained_models = {}
    training_info = {}

    for c_name, c_cfg in candidates.items():
        trained_models[c_name] = {}
        training_info[c_name] = {}
        for s in seeds:
            model_name = f"{c_cfg['prefix']}_seed{s}"
            ckpt_path = Path(f"experimental/v4/models/{model_name}.pth")

            if c_cfg["retrain"] or not ckpt_path.exists():
                model, info = train_candidate_model(
                    name=model_name,
                    alpha_mode=c_cfg["mode"],
                    alpha=c_cfg["alpha"],
                    alpha_bounds=c_cfg["bounds"],
                    seed=s,
                )
            else:
                print(f"Loading pre-existing {model_name}...")
                model = PINNCalV4(
                    bounded=True,
                    alpha=c_cfg["alpha"],
                    alpha_mode=c_cfg["mode"],
                    alpha_bounds=c_cfg["bounds"],
                ).to(DEVICE)
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                hist = np.load(f"experimental/v4/histories/{model_name}_history.npy", allow_pickle=True).item()
                best_ep = int(np.argmin(hist["val_loss"])) + 1
                info = {
                    "name": model_name,
                    "best_epoch": best_ep,
                    "best_val_loss": float(np.min(hist["val_loss"])),
                    "train_loss_at_best": float(hist["train_loss"][best_ep - 1]),
                    "gen_gap": float(hist["gen_gap"][best_ep - 1]),
                }

            trained_models[c_name][s] = model
            training_info[c_name][s] = info

    # 2. Build Evaluation Conditions
    eval_conditions = [
        ("Frozen 20 dB", None),  # Special: loads FROZEN_TEST_PATH
        ("10 dB SNR", {"snr_db": 10.0}),
        ("5 dB SNR", {"snr_db": 5.0}),
        ("Phase 35 deg", {"phase_sigma_deg": 35.0}),
        ("Gain 4.0 dB", {"gain_sigma_db": 4.0}),
        ("Coupling 40%", {"coupling_mismatch_fraction": 0.40}),
        ("Steering Stress", {"min_angle_deg": -65.0, "max_angle_deg": 65.0}),
    ]

    # Pre-generate or load datasets for consistency across candidates
    datasets = {}
    frozen_test = np.load(FROZEN_TEST_PATH, allow_pickle=True).item()
    datasets["Frozen 20 dB"] = (
        frozen_test["X"],
        frozen_test["Y"],
        frozen_test["steering_angles"],
    )

    for cond_name, overrides in eval_conditions[1:]:
        datasets[cond_name] = generate_eval_dataset(cond_name, overrides, num_samples=100)

    # 3. Evaluate Analytical Baseline
    analytical_results = {}
    for cond_name, (X, Y, ang) in datasets.items():
        C_true = Y[:, :64] + 1j * Y[:, 64:]
        c_ana = analytical_correction(X[:, :64] + 1j * X[:, 64:128], ang)
        analytical_results[cond_name] = evaluate_full_metrics(c_ana, C_true, ang)

    # 4. Evaluate all candidates across all conditions
    full_eval_results = {}

    for c_name in candidates:
        full_eval_results[c_name] = {}
        for cond_name, (X, Y, ang) in datasets.items():
            full_eval_results[c_name][cond_name] = {}
            seed_metrics = []
            for s in seeds:
                model = trained_models[c_name][s]
                m = evaluate_model_on_dataset(model, X, Y, ang)
                full_eval_results[c_name][cond_name][f"seed_{s}"] = m
                seed_metrics.append(m)

            # Compute mean and std for this condition
            agg = {}
            for k in ["nmse_db", "phase_error_deg", "gain_error_db", "relative_error", "pointing_error_deg", "beam_correlation"]:
                vals = [m[k] for m in seed_metrics]
                agg[f"{k}_mean"] = float(np.mean(vals))
                agg[f"{k}_std"] = float(np.std(vals))
            full_eval_results[c_name][cond_name]["aggregate"] = agg

        # Compute Residual & Saturation statistics on Frozen Test Set
        X_frozen, _, _ = datasets["Frozen 20 dB"]
        res_stats_seeds = []
        for s in seeds:
            st = compute_residual_and_saturation_stats(trained_models[c_name][s], X_frozen)
            res_stats_seeds.append(st)

        agg_res = {}
        for k in ["mean_residual_mag", "max_residual_mag", "mean_alpha", "min_alpha", "max_alpha", "saturation_pct"]:
            vals = [st[k] for st in res_stats_seeds]
            agg_res[f"{k}_mean"] = float(np.mean(vals))
            agg_res[f"{k}_std"] = float(np.std(vals))
        full_eval_results[c_name]["residual_stats"] = agg_res

    # 5. Print comprehensive comparative table
    print("\n" + "=" * 90)
    print("STEP 60: COMPARATIVE EVALUATION RESULTS (MEAN +/- STD OVER 3 SEEDS)")
    print("=" * 90)

    for cond_name in datasets:
        print(f"\n--- Condition: {cond_name} ---")
        ana = analytical_results[cond_name]
        print(f"  Analytical Baseline : NMSE={ana['nmse_db']:6.2f} dB | Phase={ana['phase_error_deg']:5.2f}° | Point={ana['pointing_error_deg']:.4f}° | Corr={ana['beam_correlation']:.6f}")
        for c_name in candidates:
            agg = full_eval_results[c_name][cond_name]["aggregate"]
            print(f"  {c_name:32s}: NMSE={agg['nmse_db_mean']:6.2f}±{agg['nmse_db_std']:.2f} dB | Phase={agg['phase_error_deg_mean']:5.2f}±{agg['phase_error_deg_std']:.2f}° | Point={agg['pointing_error_deg_mean']:.4f}±{agg['pointing_error_deg_std']:.4f}° | Corr={agg['beam_correlation_mean']:.6f}±{agg['beam_correlation_std']:.6f}")

    print("\n" + "=" * 90)
    print("RESIDUAL AND SATURATION STATISTICS ON FROZEN 20 dB TEST SET")
    print("=" * 90)
    for c_name in candidates:
        st = full_eval_results[c_name]["residual_stats"]
        print(f"  {c_name:32s}: Mean |Delta C|={st['mean_residual_mag_mean']:.4f}±{st['mean_residual_mag_std']:.4f} | Max |Delta C|={st['max_residual_mag_mean']:.4f} | Saturation={st['saturation_pct_mean']:.2f}% | Mean Alpha={st['mean_alpha_mean']:.4f}")

    # Save complete artifacts
    output_path = Path("experimental/v4/results/step60_complete_results.npy")
    np.save(output_path, {
        "candidates": full_eval_results,
        "analytical": analytical_results,
        "training_info": training_info,
    })
    print(f"\nAll Step 60 results saved to {output_path}")

if __name__ == "__main__":
    main()
