"""
PINN-Cal v4: STEP 62 Tikhonov-Regularized Combined Model Experiment
Evaluates:
- Control: V4-FixedAlpha-Delta0 (delta = 0.00, alpha = 0.50)
- Candidate A: V4-FixedAlpha-Delta010 (delta = 0.10, alpha = 0.50)
- Candidate B: V4-FixedAlpha-Delta015 (delta = 0.15, alpha = 0.50)
- Standalone Physics Preconditioners (delta = 0.00, 0.10, 0.15)
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

from physics_core import DEFAULT_CONFIG, steering_vector, mutual_impedance_matrix
from analytical_baseline import (
    analytical_correction,
    nmse_db,
    phase_error_deg,
    gain_error_db,
    relative_error,
)
from calibration_dataset import DatasetConfig, generate_sample
from experimental.v4.pinn_v4_model import PINNCalV4, FixedPhysicsPreconditioner, CONFIG, M
from experimental.v4.train_and_evaluate_v4 import (
    evaluate_full_metrics,
    TRAIN_PATH,
    VAL_PATH,
    FROZEN_TEST_PATH,
    DEVICE,
)

def train_tikhonov_v4_model(
    name: str,
    delta: float = 0.10,
    alpha: float = 0.5,
    seed: int = 42,
    epochs: int = 100,
    patience: int = 15,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
):
    print(f"\n--- Training {name} (Seed {seed}, delta={delta:.2f}, alpha={alpha:.2f}) ---")
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
        alpha_mode="fixed",
        delta=delta,
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

def compute_prior_and_residual_stats(model, X_np, alpha=0.5):
    model.eval()
    with torch.no_grad():
        c_hat, delta_c_real, x_prior = model(torch.tensor(X_np, dtype=torch.float32, device=DEVICE))
        delta_c = delta_c_real.cpu().numpy()
        x_prior_np = x_prior.cpu().numpy()

    c_prior_complex = x_prior_np[:, :64] + 1j * x_prior_np[:, 64:]
    prior_mag = np.abs(c_prior_complex).flatten()

    delta_c_complex = delta_c[:, :64] + 1j * delta_c[:, 64:]
    res_mag = np.abs(delta_c_complex).flatten()

    sat_mask = np.abs(delta_c) >= (0.95 * alpha)
    sat_percentage = 100.0 * float(np.mean(sat_mask))

    return {
        "mean_prior_mag": float(np.mean(prior_mag)),
        "max_prior_mag": float(np.max(prior_mag)),
        "p99_prior_mag": float(np.percentile(prior_mag, 99.0)),
        "p999_prior_mag": float(np.percentile(prior_mag, 99.9)),
        "mean_residual_mag": float(np.mean(res_mag)),
        "max_residual_mag": float(np.max(res_mag)),
        "saturation_pct": sat_percentage,
    }

def generate_robustness_dataset(overrides: dict, num_samples: int = 100, base_seed: int = 999000):
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

def evaluate_model_on_data(model, X, Y, ang):
    model.eval()
    C_true = Y[:, :64] + 1j * Y[:, 64:]
    with torch.no_grad():
        c_hat_t, _, _ = model(torch.tensor(X, dtype=torch.float32, device=DEVICE))
        c_hat = c_hat_t.cpu().numpy()
    return evaluate_full_metrics(c_hat, C_true, ang)

def evaluate_preconditioner_only(delta: float, X, Y, ang):
    precond = FixedPhysicsPreconditioner(CONFIG, delta=delta).to(DEVICE)
    with torch.no_grad():
        c_prior_t, _ = precond(torch.tensor(X, dtype=torch.float32, device=DEVICE))
        c_prior = c_prior_t.cpu().numpy()
    C_true = Y[:, :64] + 1j * Y[:, 64:]
    return evaluate_full_metrics(c_prior, C_true, ang)

def main():
    print("=" * 85)
    print("PINN-CAL v4: STEP 62 TIKHONOV COMBINED MODEL (PRECONDITIONER + DENOISER)")
    print("=" * 85)

    seeds = [42, 123, 456]

    candidates = {
        "Control (Delta=0.00)": {
            "delta": 0.00,
            "prefix": "pinn_v4_bounded",  # Pre-existing from Step 59
            "retrain": False,
        },
        "Candidate A (Delta=0.10)": {
            "delta": 0.10,
            "prefix": "pinn_v4_tikhonov010",
            "retrain": True,
        },
        "Candidate B (Delta=0.15)": {
            "delta": 0.15,
            "prefix": "pinn_v4_tikhonov015",
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
                model, info = train_tikhonov_v4_model(
                    name=model_name,
                    delta=c_cfg["delta"],
                    alpha=0.5,
                    seed=s,
                )
            else:
                print(f"Loading pre-existing {model_name}...")
                model = PINNCalV4(
                    bounded=True,
                    alpha=0.5,
                    alpha_mode="fixed",
                    delta=c_cfg["delta"],
                ).to(DEVICE)
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
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

    # 2. Build Evaluation Conditions (Frozen Benchmark + Independent Robustness Sets)
    test_conditions = [
        ("Frozen 20 dB", None),  # Benchmark
        ("Independent 20 dB", {"snr_db": 20.0}),
        ("10 dB SNR", {"snr_db": 10.0}),
        ("5 dB SNR", {"snr_db": 5.0}),
        ("Phase 35 deg", {"phase_sigma_deg": 35.0}),
        ("Gain 4.0 dB", {"gain_sigma_db": 4.0}),
        ("Coupling 40%", {"coupling_mismatch_fraction": 0.40}),
        ("Steering Stress", {"min_angle_deg": -65.0, "max_angle_deg": 65.0}),
    ]

    datasets = {}
    frozen_test = np.load(FROZEN_TEST_PATH, allow_pickle=True).item()
    datasets["Frozen 20 dB"] = (
        frozen_test["X"],
        frozen_test["Y"],
        frozen_test["steering_angles"],
    )

    for cond_name, overrides in test_conditions[1:]:
        datasets[cond_name] = generate_robustness_dataset(overrides, num_samples=100)

    # 3. Evaluate Preconditioner Alone (No neural network)
    preconditioner_alone_results = {
        "Delta=0.00 Alone": {},
        "Delta=0.10 Alone": {},
        "Delta=0.15 Alone": {},
    }
    for cond_name, (X, Y, ang) in datasets.items():
        preconditioner_alone_results["Delta=0.00 Alone"][cond_name] = evaluate_preconditioner_only(0.00, X, Y, ang)
        preconditioner_alone_results["Delta=0.10 Alone"][cond_name] = evaluate_preconditioner_only(0.10, X, Y, ang)
        preconditioner_alone_results["Delta=0.15 Alone"][cond_name] = evaluate_preconditioner_only(0.15, X, Y, ang)

    # 4. Evaluate Full Combined Models across seeds
    combined_eval_results = {}
    for c_name in candidates:
        combined_eval_results[c_name] = {}
        for cond_name, (X, Y, ang) in datasets.items():
            combined_eval_results[c_name][cond_name] = {}
            seed_metrics = []
            for s in seeds:
                model = trained_models[c_name][s]
                m = evaluate_model_on_data(model, X, Y, ang)
                combined_eval_results[c_name][cond_name][f"seed_{s}"] = m
                seed_metrics.append(m)

            # Aggregate
            agg = {}
            for k in ["nmse_db", "phase_error_deg", "gain_error_db", "relative_error", "pointing_error_deg", "beam_correlation"]:
                vals = [m[k] for m in seed_metrics]
                agg[f"{k}_mean"] = float(np.mean(vals))
                agg[f"{k}_std"] = float(np.std(vals))
            combined_eval_results[c_name][cond_name]["aggregate"] = agg

        # Prior & Residual stats on Frozen 20 dB set
        X_frozen, _, _ = datasets["Frozen 20 dB"]
        stats_seeds = []
        for s in seeds:
            st = compute_prior_and_residual_stats(trained_models[c_name][s], X_frozen, alpha=0.5)
            stats_seeds.append(st)

        agg_stats = {}
        for k in ["mean_prior_mag", "max_prior_mag", "p99_prior_mag", "p999_prior_mag", "mean_residual_mag", "max_residual_mag", "saturation_pct"]:
            vals = [st[k] for st in stats_seeds]
            agg_stats[f"{k}_mean"] = float(np.mean(vals))
            agg_stats[f"{k}_std"] = float(np.std(vals))
        combined_eval_results[c_name]["prior_residual_stats"] = agg_stats

    # 5. Display Full Comparative Results Table
    print("\n" + "=" * 90)
    print("STEP 62: FULL COMPARATIVE RESULTS (MEAN +/- STD OVER 3 SEEDS)")
    print("=" * 90)

    for cond_name in datasets:
        print(f"\n--- Condition: {cond_name} ---")
        p0 = preconditioner_alone_results["Delta=0.00 Alone"][cond_name]
        p10 = preconditioner_alone_results["Delta=0.10 Alone"][cond_name]
        p15 = preconditioner_alone_results["Delta=0.15 Alone"][cond_name]
        print(f"  Preconditioner Delta=0.00 Alone: NMSE={p0['nmse_db']:6.2f} dB | Phase={p0['phase_error_deg']:5.2f}° | Point={p0['pointing_error_deg']:.4f}° | Corr={p0['beam_correlation']:.6f}")
        print(f"  Preconditioner Delta=0.10 Alone: NMSE={p10['nmse_db']:6.2f} dB | Phase={p10['phase_error_deg']:5.2f}° | Point={p10['pointing_error_deg']:.4f}° | Corr={p10['beam_correlation']:.6f}")
        print(f"  Preconditioner Delta=0.15 Alone: NMSE={p15['nmse_db']:6.2f} dB | Phase={p15['phase_error_deg']:5.2f}° | Point={p15['pointing_error_deg']:.4f}° | Corr={p15['beam_correlation']:.6f}")
        for c_name in candidates:
            agg = combined_eval_results[c_name][cond_name]["aggregate"]
            print(f"  {c_name:32s}: NMSE={agg['nmse_db_mean']:6.2f}±{agg['nmse_db_std']:.2f} dB | Phase={agg['phase_error_deg_mean']:5.2f}±{agg['phase_error_deg_std']:.2f}° | Point={agg['pointing_error_deg_mean']:.4f}±{agg['pointing_error_deg_std']:.4f}° | Corr={agg['beam_correlation_mean']:.6f}±{agg['beam_correlation_std']:.6f}")

    print("\n" + "=" * 90)
    print("PRIOR AND RESIDUAL STATISTICS ON FROZEN 20 dB TEST SET")
    print("=" * 90)
    for c_name in candidates:
        st = combined_eval_results[c_name]["prior_residual_stats"]
        ti = [training_info[c_name][s] for s in seeds]
        mean_tr_loss = float(np.mean([t["train_loss_at_best"] for t in ti]))
        mean_val_loss = float(np.mean([t["best_val_loss"] for t in ti]))
        print(f"  {c_name:30s}: Max |C_prior|={st['max_prior_mag_mean']:6.2f} | p99%={st['p99_prior_mag_mean']:5.2f} | Mean |Delta C|={st['mean_residual_mag_mean']:.4f} | Saturation={st['saturation_pct_mean']:.2f}% | TrainLoss={mean_tr_loss:.4f} | ValLoss={mean_val_loss:.4f}")

    # 6. Preconditioner Alone vs Combined Model Comparison
    print("\n" + "=" * 90)
    print("NEURAL ADDED VALUE ANALYSIS: PRECONDITIONER ALONE vs PRECONDITIONER + NEURAL DENOISER")
    print("=" * 90)
    for cond_name in ["Frozen 20 dB", "10 dB SNR", "5 dB SNR", "Phase 35 deg", "Gain 4.0 dB", "Coupling 40%"]:
        p10 = preconditioner_alone_results["Delta=0.10 Alone"][cond_name]["nmse_db"]
        m10 = combined_eval_results["Candidate A (Delta=0.10)"][cond_name]["aggregate"]["nmse_db_mean"]
        diff = m10 - p10
        print(f"  [{cond_name:18s}] Delta=0.10: Precond Alone={p10:6.2f} dB -> Combined Model={m10:6.2f} dB (Added value: {diff:+.2f} dB)")

    # Save complete artifacts
    output_path = Path("experimental/v4/results/step62_combined_results.npy")
    np.save(output_path, {
        "combined_results": combined_eval_results,
        "preconditioner_alone": preconditioner_alone_results,
        "training_info": training_info,
        "seeds": seeds,
    })
    print(f"\nAll Step 62 results successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
