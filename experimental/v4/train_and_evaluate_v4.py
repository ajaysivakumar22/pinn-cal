"""
PINN-Cal v4: Complete Training and Evaluation Suite
Executes:
1. Multi-seed training (Seeds 42, 123, 456) for primary Bounded V4 model.
2. Controlled Ablation:
   - Variant A: Analytical Baseline Only
   - Variant B: Physics Prior + Bounded Neural Residual (alpha = 0.5 * tanh)
   - Variant C: Physics Prior + Unbounded Neural Residual (linear output)
3. Rigorous Evaluation:
   - Frozen 20 dB Test Set (data/physics_test_200.npy)
   - Low-SNR Test Sets: 10 dB and 5 dB
   - High-Impairment Test Sets: Phase sigma=35 deg, Gain sigma=4 dB, Coupling mismatch=40%
   - Far-Field Beam Pattern Metrics (Pointing Error, Beam Correlation, Beamwidth, PSLL)
4. Saves all checkpoints, histories, metrics, and residual statistics in experimental/v4/
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
from analytical_baseline import analytical_correction, nmse_db, phase_error_deg, gain_error_db, relative_error
from evaluate_beam_metrics import calculate_patterns, evaluate_patterns
from calibration_dataset import DatasetConfig, generate_sample
from experimental.v4.pinn_v4_model import PINNCalV4, CONFIG, M

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_PATH = Path("experimental/v4/datasets/v4_train_2000.npy")
VAL_PATH = Path("experimental/v4/datasets/v4_val_500.npy")
FROZEN_TEST_PATH = Path("data/physics_test_200.npy")

def compute_beam_metrics(corrected_C_np, target_C_np, angles_deg):
    angles_t = torch.tensor(angles_deg, dtype=torch.float64, device=DEVICE)
    ideal_weights = steering_vector(angles_t, CONFIG).to(torch.complex128)

    target_C_t = torch.tensor(target_C_np, dtype=torch.complex128, device=DEVICE)
    true_hardware = ideal_weights / (target_C_t + 1e-12)

    corr_C_t = torch.tensor(corrected_C_np, dtype=torch.complex128, device=DEVICE)
    corrected_weights = true_hardware * corr_C_t

    ideal_patterns = calculate_patterns(ideal_weights)
    beam_res = evaluate_patterns(corrected_weights, ideal_patterns, angles_deg)

    pointing_err = float(np.nanmean(np.abs(beam_res["pointing_errors"])))
    beam_corr = float(np.nanmean(beam_res["correlations"]))
    bw_3db = float(np.nanmean(beam_res["beamwidths"]))
    psll = float(np.nanmean(beam_res["psll"]))

    return {
        "pointing_error_deg": pointing_err,
        "beam_correlation": beam_corr,
        "beamwidth_3db": bw_3db,
        "psll_db": psll,
    }

def evaluate_full_metrics(c_pred_np, c_target_np, angles_deg):
    nmse = nmse_db(c_pred_np, c_target_np)
    phase_err = phase_error_deg(c_pred_np, c_target_np)
    gain_err = gain_error_db(c_pred_np, c_target_np)
    rel_err = 100.0 * relative_error(c_pred_np, c_target_np)
    beam_m = compute_beam_metrics(c_pred_np, c_target_np, angles_deg)

    return {
        "nmse_db": nmse,
        "phase_error_deg": phase_err,
        "gain_error_db": gain_err,
        "relative_error": rel_err,
        **beam_m,
    }

def train_v4_model(
    name: str,
    bounded: bool = True,
    alpha: float = 0.5,
    seed: int = 42,
    epochs: int = 100,
    patience: int = 15,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
):
    print(f"\n--- Training {name} (Seed {seed}, Bounded={bounded}, alpha={alpha}) ---")
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

    model = PINNCalV4(bounded=bounded, alpha=alpha).to(DEVICE)
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

            # Target is Y (real and imag parts of C_true)
            # c_hat in real representation:
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

    print(f"  Finished training in {elapsed:.1f}s | Best Ep {best_epoch:2d}/{epochs} | ValLoss={best_val_loss:.6e} | GenGap={history['gen_gap'][best_epoch-1]:.6e}")

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

def run_evaluation_suite(model, model_name: str):
    """
    Evaluates model across:
    1. Frozen 20 dB Test Set
    2. Low-SNR Test Sets (10 dB, 5 dB)
    3. Severe Phase (35 deg), Severe Gain (4 dB), Coupling Mismatch (40%)
    """
    model.eval()

    # 1. Frozen 20 dB Test Set
    frozen_test = np.load(FROZEN_TEST_PATH, allow_pickle=True).item()
    X_frozen = frozen_test["X"]
    Y_frozen = frozen_test["Y"]
    angles_frozen = frozen_test["steering_angles"]
    C_true_frozen = Y_frozen[:, :64] + 1j * Y_frozen[:, 64:]

    with torch.no_grad():
        c_hat_t, delta_c_t, x_prior_t = model(torch.tensor(X_frozen, dtype=torch.float32, device=DEVICE))
        c_hat_frozen = c_hat_t.cpu().numpy()
        delta_c_np = delta_c_t.cpu().numpy()
        c_prior_np = (x_prior_t[:, :64] + 1j * x_prior_t[:, 64:]).cpu().numpy()

    frozen_metrics = evaluate_full_metrics(c_hat_frozen, C_true_frozen, angles_frozen)

    # Also compute analytical baseline on frozen test set for comparison
    analytical_frozen = analytical_correction(X_frozen[:, :64] + 1j * X_frozen[:, 64:128], angles_frozen)
    analytical_metrics = evaluate_full_metrics(analytical_frozen, C_true_frozen, angles_frozen)

    # 2. Stress / OOD conditions
    stress_conditions = [
        ("Low SNR (10 dB)", {"snr_db": 10.0}),
        ("Very Low SNR (5 dB)", {"snr_db": 5.0}),
        ("Severe Phase (35 deg)", {"phase_sigma_deg": 35.0}),
        ("Severe Gain (4.0 dB)", {"gain_sigma_db": 4.0}),
        ("Severe Coupling (40%)", {"coupling_mismatch_fraction": 0.40}),
    ]

    stress_results = {}
    for cond_name, overrides in stress_conditions:
        # Generate 100 independent evaluation samples
        X_s, Y_s, ang_s = [], [], []
        rng = np.random.default_rng(777000 + int(overrides.get("snr_db", 0)) + int(overrides.get("phase_sigma_deg", 0)))
        for i in range(100):
            seed = 888000 + i
            angle = float(rng.uniform(-60.0, 60.0))
            cfg = DatasetConfig(
                num_samples=1,
                phase_sigma_deg=overrides.get("phase_sigma_deg", 20.0),
                gain_sigma_db=overrides.get("gain_sigma_db", 2.0),
                snr_db=overrides.get("snr_db", 20.0),
                coupling_mismatch_fraction=overrides.get("coupling_mismatch_fraction", 0.20),
                min_angle_deg=-60.0,
                max_angle_deg=60.0,
                seed=seed,
            )
            x, y = generate_sample(angle, seed, config=CONFIG, dataset_config=cfg)
            X_s.append(x)
            Y_s.append(y)
            ang_s.append(angle)

        X_s = np.array(X_s, dtype=np.float32)
        Y_s = np.array(Y_s, dtype=np.float32)
        ang_s = np.array(ang_s, dtype=np.float64)
        C_true_s = Y_s[:, :64] + 1j * Y_s[:, 64:]

        # Model prediction
        with torch.no_grad():
            c_hat_s, _, _ = model(torch.tensor(X_s, dtype=torch.float32, device=DEVICE))
            c_hat_s_np = c_hat_s.cpu().numpy()

        m_v4 = evaluate_full_metrics(c_hat_s_np, C_true_s, ang_s)

        # Analytical prediction
        c_ana_s = analytical_correction(X_s[:, :64] + 1j * X_s[:, 64:128], ang_s)
        m_ana = evaluate_full_metrics(c_ana_s, C_true_s, ang_s)

        stress_results[cond_name] = {
            "v4": m_v4,
            "analytical": m_ana,
        }

    # Residual distribution statistics on frozen test
    res_mag = np.abs(delta_c_np[:, :64] + 1j * delta_c_np[:, 64:])
    residual_stats = {
        "mean_residual_mag": float(np.mean(res_mag)),
        "std_residual_mag": float(np.std(res_mag)),
        "max_residual_mag": float(np.max(res_mag)),
        "min_residual_mag": float(np.min(res_mag)),
    }

    return {
        "model_name": model_name,
        "frozen_test": frozen_metrics,
        "analytical_frozen": analytical_metrics,
        "stress_results": stress_results,
        "residual_stats": residual_stats,
    }

def main():
    print("=" * 70)
    print("PINN-CAL v4 EXPERIMENTAL PIPELINE EXECUTION")
    print("=" * 70)

    # 1. Ablation comparison: Bounded vs Unbounded vs Analytical
    # A. Train Bounded V4 (Primary design: alpha = 0.5 * tanh)
    model_bounded, info_bounded = train_v4_model(
        "pinn_v4_bounded_seed42", bounded=True, alpha=0.5, seed=42
    )
    eval_bounded = run_evaluation_suite(model_bounded, "pinn_v4_bounded_seed42")

    # B. Train Unbounded V4 (Linear output)
    model_unbounded, info_unbounded = train_v4_model(
        "pinn_v4_unbounded_seed42", bounded=False, seed=42
    )
    eval_unbounded = run_evaluation_suite(model_unbounded, "pinn_v4_unbounded_seed42")

    # 2. Multi-Seed Stability for Primary Bounded V4 (Seeds 123, 456)
    model_s123, info_s123 = train_v4_model("pinn_v4_bounded_seed123", bounded=True, alpha=0.5, seed=123)
    eval_s123 = run_evaluation_suite(model_s123, "pinn_v4_bounded_seed123")

    model_s456, info_s456 = train_v4_model("pinn_v4_bounded_seed456", bounded=True, alpha=0.5, seed=456)
    eval_s456 = run_evaluation_suite(model_s456, "pinn_v4_bounded_seed456")

    all_v4_results = {
        "bounded_seed42": {**info_bounded, **eval_bounded},
        "unbounded_seed42": {**info_unbounded, **eval_unbounded},
        "bounded_seed123": {**info_s123, **eval_s123},
        "bounded_seed456": {**info_s456, **eval_s456},
    }

    np.save("experimental/v4/results/v4_complete_results.npy", all_v4_results)

    print("\n" + "=" * 70)
    print("PINN-CAL v4 EVALUATION SUMMARY ON FROZEN TEST SET")
    print("=" * 70)
    ana = eval_bounded["analytical_frozen"]
    b42 = eval_bounded["frozen_test"]
    ub42 = eval_unbounded["frozen_test"]

    print(f"Analytical Baseline      : NMSE={ana['nmse_db']:.2f} dB | PhaseErr={ana['phase_error_deg']:.2f} deg | PointingErr={ana['pointing_error_deg']:.4f} deg | BeamCorr={ana['beam_correlation']:.6f}")
    print(f"V4 Bounded (Seed 42)     : NMSE={b42['nmse_db']:.2f} dB | PhaseErr={b42['phase_error_deg']:.2f} deg | PointingErr={b42['pointing_error_deg']:.4f} deg | BeamCorr={b42['beam_correlation']:.6f}")
    print(f"V4 Unbounded (Seed 42)   : NMSE={ub42['nmse_db']:.2f} dB | PhaseErr={ub42['phase_error_deg']:.2f} deg | PointingErr={ub42['pointing_error_deg']:.4f} deg | BeamCorr={ub42['beam_correlation']:.6f}")

    # Multi-seed aggregate
    ms_nmses = [all_v4_results[k]["frozen_test"]["nmse_db"] for k in ["bounded_seed42", "bounded_seed123", "bounded_seed456"]]
    ms_phases = [all_v4_results[k]["frozen_test"]["phase_error_deg"] for k in ["bounded_seed42", "bounded_seed123", "bounded_seed456"]]
    ms_pointings = [all_v4_results[k]["frozen_test"]["pointing_error_deg"] for k in ["bounded_seed42", "bounded_seed123", "bounded_seed456"]]
    ms_corrs = [all_v4_results[k]["frozen_test"]["beam_correlation"] for k in ["bounded_seed42", "bounded_seed123", "bounded_seed456"]]

    print("\nMulti-Seed Bounded V4 Aggregate (Seeds 42, 123, 456):")
    print(f"  NMSE           : {np.mean(ms_nmses):.4f} +/- {np.std(ms_nmses):.4f} dB")
    print(f"  Phase Error    : {np.mean(ms_phases):.4f} +/- {np.std(ms_phases):.4f} deg")
    print(f"  Pointing Error : {np.mean(ms_pointings):.4f} +/- {np.std(ms_pointings):.4f} deg")
    print(f"  Beam Corr      : {np.mean(ms_corrs):.6f} +/- {np.std(ms_corrs):.6f}")

    print("\nLow-SNR & Stress Summary (Bounded V4 vs Analytical):")
    for cond_name, res in eval_bounded["stress_results"].items():
        v4_m = res["v4"]
        an_m = res["analytical"]
        print(f"  [{cond_name:24s}] V4: NMSE={v4_m['nmse_db']:6.2f} dB (Phase={v4_m['phase_error_deg']:5.2f} deg, Point={v4_m['pointing_error_deg']:.4f} deg, Corr={v4_m['beam_correlation']:.6f})")
        print(f"  {' '*26} Ana: NMSE={an_m['nmse_db']:6.2f} dB (Phase={an_m['phase_error_deg']:5.2f} deg, Point={an_m['pointing_error_deg']:.4f} deg, Corr={an_m['beam_correlation']:.6f})")

    print("\nExecution complete. Results saved in experimental/v4/results/v4_complete_results.npy")

if __name__ == "__main__":
    main()
