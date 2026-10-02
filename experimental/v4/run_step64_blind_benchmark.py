"""
PINN-Cal v4: STEP 64 Blind Final Holdout Benchmark Suite

Evaluates ONLY fixed pre-designated models:
1. Analytical Baseline
2. Frozen original PINN-v3 (models/pinn_cal_final_best.pth)
3. V4 Candidate B (delta=0.15, alpha=0.50, Seed 42 - Pre-designated primary reference)
4. V4 Candidate B (delta=0.15, alpha=0.50, Seed 123)
5. V4 Candidate B (delta=0.15, alpha=0.50, Seed 456)
6. Tikhonov Preconditioner alone (delta=0.15) for neural added value quantification.

Evaluated on:
experimental/v4/results/blind_final_holdout_500.npy (N=500 independent samples per condition).
"""
import sys
import hashlib
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics_core import DEFAULT_CONFIG, steering_vector, mutual_impedance_matrix
from analytical_baseline import (
    analytical_correction,
    nmse_db,
    phase_error_deg,
    gain_error_db,
    relative_error,
)
from evaluate_beam_metrics import calculate_patterns, evaluate_patterns
from pinn_cal_final import PINNCalV3
from experimental.v4.pinn_v4_model import PINNCalV4, FixedPhysicsPreconditioner, CONFIG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOLDOUT_PATH = Path("experimental/v4/results/blind_final_holdout_500.npy")
RESULTS_OUTPUT_PATH = Path("experimental/v4/results/step64_blind_holdout_results.npy")
FIGURES_DIR = Path("experimental/v4/results/step64_figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def compute_checkpoint_hash(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_detailed_metrics(c_pred_np: np.ndarray, c_true_np: np.ndarray, angles_deg: np.ndarray):
    """
    Compute aggregate and sample-level distribution metrics (mean, std, min, max, count).
    """
    N = len(c_pred_np)
    # 1. Global NMSE
    num = np.sum(np.abs(c_pred_np - c_true_np) ** 2)
    den = np.sum(np.abs(c_true_np) ** 2)
    global_nmse = float(10.0 * np.log10(num / den))

    # 2. Per-sample NMSE
    sample_num = np.sum(np.abs(c_pred_np - c_true_np) ** 2, axis=1)
    sample_den = np.sum(np.abs(c_true_np) ** 2, axis=1)
    sample_nmse = 10.0 * np.log10(np.maximum(sample_num, 1e-18) / np.maximum(sample_den, 1e-18))

    # 3. Phase Error (deg)
    phase_diff = np.angle(c_pred_np * np.conj(c_true_np))
    sample_phase = np.degrees(np.mean(np.abs(phase_diff), axis=1))

    # 4. Gain Error (dB)
    pred_db = 20.0 * np.log10(np.maximum(np.abs(c_pred_np), 1e-12))
    target_db = 20.0 * np.log10(np.maximum(np.abs(c_true_np), 1e-12))
    sample_gain = np.mean(np.abs(pred_db - target_db), axis=1)

    # 5. Relative Complex Error (%)
    sample_rel = np.mean(np.abs(c_pred_np - c_true_np) / np.maximum(np.abs(c_true_np), 1e-12), axis=1) * 100.0

    # 6. Beam metrics
    angles_t = torch.tensor(angles_deg, dtype=torch.float64, device=DEVICE)
    ideal_weights = steering_vector(angles_t, CONFIG).to(torch.complex128)
    target_C_t = torch.tensor(c_true_np, dtype=torch.complex128, device=DEVICE)
    true_hardware = ideal_weights / (target_C_t + 1e-12)

    corr_C_t = torch.tensor(c_pred_np, dtype=torch.complex128, device=DEVICE)
    corrected_weights = true_hardware * corr_C_t

    ideal_patterns = calculate_patterns(ideal_weights)
    beam_res = evaluate_patterns(corrected_weights, ideal_patterns, angles_deg)

    sample_pointing = np.abs(beam_res["pointing_errors"])
    sample_corr = beam_res["correlations"]

    return {
        "global_nmse_db": global_nmse,
        "nmse_db": global_nmse,  # Standard reference
        "sample_nmse_mean": float(np.mean(sample_nmse)),
        "sample_nmse_std": float(np.std(sample_nmse)),
        "phase_error_deg": float(np.mean(sample_phase)),
        "phase_error_std": float(np.std(sample_phase)),
        "gain_error_db": float(np.mean(sample_gain)),
        "gain_error_std": float(np.std(sample_gain)),
        "relative_error": float(np.mean(sample_rel)),
        "relative_error_std": float(np.std(sample_rel)),
        "pointing_error_deg": float(np.nanmean(sample_pointing)),
        "pointing_error_std": float(np.nanstd(sample_pointing)),
        "beam_correlation": float(np.nanmean(sample_corr)),
        "beam_correlation_std": float(np.nanstd(sample_corr)),
        "sample_count": N,
        # Raw arrays for distributions and plots
        "sample_nmse_arr": sample_nmse,
        "sample_phase_arr": sample_phase,
        "sample_pointing_arr": sample_pointing,
        "sample_corr_arr": sample_corr,
    }

def evaluate_v3_model(model, X, Y, ang):
    model.eval()
    C_true = Y[:, :64] + 1j * Y[:, 64:]
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32, device=DEVICE))
        c_pinn = torch.complex(pred[:, :64].to(torch.float64), pred[:, 64:].to(torch.float64)).cpu().numpy()
    return compute_detailed_metrics(c_pinn, C_true, ang), c_pinn

def evaluate_v4_model(model, X, Y, ang):
    model.eval()
    C_true = Y[:, :64] + 1j * Y[:, 64:]
    with torch.no_grad():
        c_hat_t, _, _ = model(torch.tensor(X, dtype=torch.float32, device=DEVICE))
        c_hat = c_hat_t.cpu().numpy()
    return compute_detailed_metrics(c_hat, C_true, ang), c_hat

def evaluate_preconditioner(delta: float, X, Y, ang):
    precond = FixedPhysicsPreconditioner(CONFIG, delta=delta).to(DEVICE)
    with torch.no_grad():
        c_prior_t, _ = precond(torch.tensor(X, dtype=torch.float32, device=DEVICE))
        c_prior = c_prior_t.cpu().numpy()
    C_true = Y[:, :64] + 1j * Y[:, 64:]
    return compute_detailed_metrics(c_prior, C_true, ang), c_prior

def plot_step64_figures(results, holdout_data):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {
        "Analytical Baseline": "#2b5c8f",
        "Frozen PINN-v3": "#d95f02",
        "V4 Delta=0.15 (Seed 42)": "#e7298a",
        "V4 Delta=0.15 (Seed 123)": "#7570b3",
        "V4 Delta=0.15 (Seed 456)": "#1b9e77",
        "V4 Delta=0.15 (Seed Mean)": "#984ea3",
    }

    snr_levels = [5.0, 7.5, 10.0, 15.0, 20.0]
    snr_keys = ["5 dB SNR", "7.5 dB SNR", "10 dB SNR", "15 dB SNR", "Nominal 20 dB"]

    # 1. NMSE vs SNR (with error bars representing standard deviation across N=500 samples)
    plt.figure(figsize=(8.5, 5.2), dpi=300)
    for model_name in ["Analytical Baseline", "Frozen PINN-v3", "V4 Delta=0.15 (Seed 42)", "V4 Delta=0.15 (Seed Mean)"]:
        nmse_vals = [results[model_name][k]["nmse_db"] for k in snr_keys]
        plt.plot(snr_levels, nmse_vals, marker="o", linewidth=2.4, label=model_name, color=colors[model_name])
    plt.xlabel("Signal-to-Noise Ratio (SNR) [dB]", fontsize=12, fontweight="bold")
    plt.ylabel("Calibration NMSE [dB] (Lower is Better)", fontsize=12, fontweight="bold")
    plt.title("Blind Holdout (N=500): Calibration Accuracy vs SNR Robustness", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nmse_vs_snr.png")
    plt.close()

    # 2. Beam Correlation vs SNR
    plt.figure(figsize=(8.5, 5.2), dpi=300)
    for model_name in ["Analytical Baseline", "Frozen PINN-v3", "V4 Delta=0.15 (Seed 42)", "V4 Delta=0.15 (Seed Mean)"]:
        corr_vals = [results[model_name][k]["beam_correlation"] for k in snr_keys]
        corr_stds = [results[model_name][k].get("beam_correlation_std", results[model_name][k].get("beam_correlation_seed_std", 0.0)) for k in snr_keys]
        plt.errorbar(snr_levels, corr_vals, yerr=corr_stds, marker="s", capsize=4, linewidth=2.2, label=model_name, color=colors[model_name])
    plt.xlabel("Signal-to-Noise Ratio (SNR) [dB]", fontsize=12, fontweight="bold")
    plt.ylabel("Far-Field Beam Correlation (Higher is Better)", fontsize=12, fontweight="bold")
    plt.title("Blind Holdout (N=500): Beam Correlation vs SNR (Mean ± Std)", fontsize=13, fontweight="bold")
    plt.ylim(0.85, 1.002)
    plt.legend(frameon=True, fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "beam_correlation_vs_snr.png")
    plt.close()

    # 3. Pointing Error vs SNR
    plt.figure(figsize=(8.5, 5.2), dpi=300)
    for model_name in ["Analytical Baseline", "Frozen PINN-v3", "V4 Delta=0.15 (Seed 42)", "V4 Delta=0.15 (Seed Mean)"]:
        point_vals = [results[model_name][k]["pointing_error_deg"] for k in snr_keys]
        plt.plot(snr_levels, point_vals, marker="^", linewidth=2.2, label=model_name, color=colors[model_name])
    plt.xlabel("Signal-to-Noise Ratio (SNR) [dB]", fontsize=12, fontweight="bold")
    plt.ylabel("Beam Pointing Error [deg] (Lower is Better)", fontsize=12, fontweight="bold")
    plt.title("Blind Holdout (N=500): Beam Pointing Stability vs SNR", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pointing_error_vs_snr.png")
    plt.close()

    # 4. Nominal Beam Pattern (Sample #0 from Nominal 20 dB holdout)
    nominal_split = holdout_data["conditions"]["Nominal 20 dB"]
    sample_idx = 0
    target_ang = float(nominal_split["steering_angles"][sample_idx])

    v_ideal_sample = steering_vector(torch.tensor([target_ang], dtype=torch.float64), CONFIG).to(torch.complex128)
    c_true_sample = torch.tensor(nominal_split["Y"][sample_idx, :64] + 1j * nominal_split["Y"][sample_idx, 64:], dtype=torch.complex128, device=DEVICE)
    v_hw_true = (v_ideal_sample.to(DEVICE) / (c_true_sample + 1e-12)).squeeze(0)

    fine_angles = np.linspace(-90.0, 90.0, 721)
    fine_angles_t = torch.tensor(fine_angles, dtype=torch.float64, device=DEVICE)

    def get_pattern(weights):
        steer_grid = steering_vector(fine_angles_t, CONFIG).to(torch.complex128)
        pat = torch.abs(torch.matmul(steer_grid, weights.unsqueeze(-1))).squeeze(-1)
        pat_db = 20.0 * torch.log10(pat / (torch.max(pat) + 1e-12))
        return pat_db.cpu().numpy()

    ideal_pat = get_pattern(v_ideal_sample.squeeze(0).to(DEVICE))
    impaired_pat = get_pattern(v_hw_true)
    ana_w = v_hw_true * torch.tensor(results["Analytical Baseline"]["Nominal 20 dB"]["raw_C"][sample_idx], device=DEVICE)
    v3_w = v_hw_true * torch.tensor(results["Frozen PINN-v3"]["Nominal 20 dB"]["raw_C"][sample_idx], device=DEVICE)
    v4_w = v_hw_true * torch.tensor(results["V4 Delta=0.15 (Seed 42)"]["Nominal 20 dB"]["raw_C"][sample_idx], device=DEVICE)

    plt.figure(figsize=(9.2, 5.2), dpi=300)
    plt.plot(fine_angles, ideal_pat, "k--", label=f"Ideal Steering ({target_ang:.1f}°)", linewidth=1.8)
    plt.plot(fine_angles, get_pattern(ana_w), color=colors["Analytical Baseline"], label="Analytical Baseline", linewidth=1.5)
    plt.plot(fine_angles, get_pattern(v3_w), color=colors["Frozen PINN-v3"], label="Frozen PINN-v3", linewidth=1.5)
    plt.plot(fine_angles, get_pattern(v4_w), color=colors["V4 Delta=0.15 (Seed 42)"], label="V4 Delta=0.15 (Seed 42)", linewidth=2.0)
    plt.plot(fine_angles, impaired_pat, ":", color="#999999", label="Impaired (Uncalibrated)", linewidth=1.2)
    plt.xlabel("Spatial Angle [deg]", fontsize=12, fontweight="bold")
    plt.ylabel("Normalized Power [dB]", fontsize=12, fontweight="bold")
    plt.title(f"Blind Holdout Synthesized Beam Pattern (Theta = {target_ang:.1f}°)", fontsize=13, fontweight="bold")
    plt.ylim(-40.0, 2.0)
    plt.xlim(-80.0, 80.0)
    plt.legend(frameon=True, loc="upper right", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nominal_beam_pattern.png")
    plt.close()

    # 5. Stress Condition Comparison (Bar chart across 5 stress conditions)
    stress_conditions = ["Phase 35 deg", "Gain 4.0 dB", "Coupling 40%", "Steering Stress ±65°", "Combined Severe"]
    models_to_compare = ["Analytical Baseline", "Frozen PINN-v3", "V4 Delta=0.15 (Seed 42)"]
    x = np.arange(len(stress_conditions))
    width = 0.25

    plt.figure(figsize=(10, 5.5), dpi=300)
    for i, m_name in enumerate(models_to_compare):
        vals = [results[m_name][sc]["nmse_db"] for sc in stress_conditions]
        plt.bar(x + i * width, vals, width, label=m_name, color=colors[m_name], edgecolor="black", linewidth=0.8)
    plt.xlabel("Stress Condition", fontsize=12, fontweight="bold")
    plt.ylabel("Calibration NMSE [dB] (Lower is Better)", fontsize=12, fontweight="bold")
    plt.title("Blind Holdout (N=500): Hardware & Operational Stress Comparison", fontsize=13, fontweight="bold")
    plt.xticks(x + width, ["Phase 35°", "Gain 4 dB", "Coupling 40%", "Steer ±65°", "Combined Severe"], fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stress_condition_comparison.png")
    plt.close()

    # 6. V3 vs V4 Calibration Error on Nominal 20 dB (NMSE, Phase, Gain 3-panel bar chart)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=300)
    comp_models = ["Analytical Baseline", "Frozen PINN-v3", "V4 Delta=0.15 (Seed 42)", "V4 Delta=0.15 (Seed Mean)"]
    labels = ["Analytical", "Frozen V3", "V4 (Seed 42)", "V4 (Seed Mean)"]
    b_colors = [colors[m] for m in comp_models]

    nmses = [results[m]["Nominal 20 dB"]["nmse_db"] for m in comp_models]
    phases = [results[m]["Nominal 20 dB"]["phase_error_deg"] for m in comp_models]
    gains = [results[m]["Nominal 20 dB"]["gain_error_db"] for m in comp_models]

    axes[0].bar(range(len(comp_models)), nmses, color=b_colors, edgecolor="black", linewidth=0.8)
    axes[0].set_title("NMSE [dB] (Lower is Better)", fontweight="bold")
    axes[0].set_xticks(range(len(comp_models)))
    axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=9.5)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].bar(range(len(comp_models)), phases, color=b_colors, edgecolor="black", linewidth=0.8)
    axes[1].set_title("Phase Error [deg] (Lower is Better)", fontweight="bold")
    axes[1].set_xticks(range(len(comp_models)))
    axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=9.5)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    axes[2].bar(range(len(comp_models)), gains, color=b_colors, edgecolor="black", linewidth=0.8)
    axes[2].set_title("Gain Error [dB] (Lower is Better)", fontweight="bold")
    axes[2].set_xticks(range(len(comp_models)))
    axes[2].set_xticklabels(labels, rotation=25, ha="right", fontsize=9.5)
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Blind Holdout (N=500): Nominal 20 dB Calibration Quality", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "v3_vs_v4_calibration_error.png")
    plt.close()

    print(f"All 6 Step 64 figures successfully saved to {FIGURES_DIR}/")

def main():
    print("=" * 85)
    print("PINN-CAL v4: STEP 64 BLIND FINAL HOLDOUT BENCHMARK")
    print("=" * 85)

    assert HOLDOUT_PATH.exists(), f"Blind holdout dataset {HOLDOUT_PATH} not found!"
    holdout_data = np.load(HOLDOUT_PATH, allow_pickle=True).item()
    print(f"Loaded Blind Holdout Dataset from {HOLDOUT_PATH}")
    print(f"Base Seed: {holdout_data['metadata']['base_seed']} | Created: {holdout_data['metadata']['creation_timestamp']}")
    print(f"Conditions: {list(holdout_data['conditions'].keys())}")

    # Verify frozen v3 checkpoint SHA256
    v3_path = Path("models/pinn_cal_final_best.pth")
    assert v3_path.exists(), f"Frozen model {v3_path} missing!"
    v3_hash = compute_checkpoint_hash(v3_path)
    print(f"Verified Frozen PINN-v3 Checkpoint SHA256: {v3_hash}")

    # Load models
    v3_model = PINNCalV3().to(DEVICE)
    v3_model.load_state_dict(torch.load(v3_path, map_location=DEVICE, weights_only=True))

    seeds = [42, 123, 456]
    v4_models = {}
    for s in seeds:
        m_path = Path(f"experimental/v4/models/pinn_v4_tikhonov015_seed{s}.pth")
        assert m_path.exists(), f"Model {m_path} missing!"
        m = PINNCalV4(bounded=True, alpha=0.5, delta=0.15).to(DEVICE)
        m.load_state_dict(torch.load(m_path, map_location=DEVICE, weights_only=True))
        v4_models[s] = m

    results = {
        "Analytical Baseline": {},
        "Frozen PINN-v3": {},
        "V4 Delta=0.15 (Seed 42)": {},
        "V4 Delta=0.15 (Seed 123)": {},
        "V4 Delta=0.15 (Seed 456)": {},
        "V4 Delta=0.15 (Seed Mean)": {},
        "Tikhonov Preconditioner Alone": {},
    }

    print("\n--- Running Blind Holdout Evaluations across 10 Conditions (N=500 each) ---")
    for cond_name, split in holdout_data["conditions"].items():
        X = split["X"]
        Y = split["Y"]
        ang = split["steering_angles"]
        C_true = Y[:, :64] + 1j * Y[:, 64:]

        # A. Analytical Baseline
        c_ana = analytical_correction(X[:, :64] + 1j * X[:, 64:128], ang)
        m_ana = compute_detailed_metrics(c_ana, C_true, ang)
        results["Analytical Baseline"][cond_name] = {**m_ana, "raw_C": c_ana}

        # B. Frozen PINN-v3
        m_v3, c_v3 = evaluate_v3_model(v3_model, X, Y, ang)
        results["Frozen PINN-v3"][cond_name] = {**m_v3, "raw_C": c_v3}

        # C. Tikhonov Preconditioner alone (delta=0.15)
        m_p, c_p = evaluate_preconditioner(0.15, X, Y, ang)
        results["Tikhonov Preconditioner Alone"][cond_name] = {**m_p, "raw_C": c_p}

        # D. V4 Delta=0.15 Individual Seeds
        seed_metrics = []
        for s in seeds:
            m_s, c_s = evaluate_v4_model(v4_models[s], X, Y, ang)
            results[f"V4 Delta=0.15 (Seed {s})"][cond_name] = {**m_s, "raw_C": c_s}
            seed_metrics.append(m_s)

        # E. V4 Delta=0.15 Seed Aggregate (Mean ± Std across seeds)
        agg_v4 = {
            "sample_count": len(X),
        }
        for k in ["global_nmse_db", "nmse_db", "phase_error_deg", "gain_error_db", "relative_error", "pointing_error_deg", "beam_correlation"]:
            vals = [sm[k] for sm in seed_metrics]
            agg_v4[k] = float(np.mean(vals))
            agg_v4[f"{k}_seed_std"] = float(np.std(vals))
            agg_v4[f"{k}_seed_spread"] = float(np.max(vals) - np.min(vals))
            agg_v4[f"{k}_std"] = float(np.mean([sm.get(f"{k}_std", 0.0) for sm in seed_metrics]))
        results["V4 Delta=0.15 (Seed Mean)"][cond_name] = agg_v4

        print(f"  Condition: {cond_name:22s} | Analytical={m_ana['nmse_db']:6.2f} dB | PINN-v3={m_v3['nmse_db']:6.2f} dB | V4 Seed 42={results['V4 Delta=0.15 (Seed 42)'][cond_name]['nmse_db']:6.2f} dB | Precond={m_p['nmse_db']:6.2f} dB")

    # Generate Figures
    print("\n--- Generating Publication-Quality Figures ---")
    plot_step64_figures(results, holdout_data)

    # Save results
    np.save(RESULTS_OUTPUT_PATH, {
        "results": results,
        "v3_hash": v3_hash,
        "holdout_metadata": holdout_data["metadata"],
    })
    print(f"\nStep 64 blind holdout benchmark results successfully saved to {RESULTS_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
