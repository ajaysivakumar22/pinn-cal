"""
PINN-Cal v4: STEP 63 Final Exhaustive Benchmark Suite
Evaluates:
1. Analytical Baseline
2. Frozen original PINN-v3 (models/pinn_cal_final_best.pth)
3. V4 Control: delta = 0.00, alpha = 0.50 (Seeds 42, 123, 456)
4. V4 Tikhonov: delta = 0.10, alpha = 0.50 (Seeds 42, 123, 456)
5. V4 Tikhonov: delta = 0.15, alpha = 0.50 (Seeds 42, 123, 456 individual and aggregate)
6. Preconditioners alone (delta = 0.00, 0.10, 0.15)
Across 11 comprehensive evaluation conditions with publication figures.
"""
import sys
import hashlib
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
from calibration_dataset import DatasetConfig, generate_sample
from pinn_cal_final import PINNCalV3
from experimental.v4.pinn_v4_model import PINNCalV4, FixedPhysicsPreconditioner, CONFIG, M
from experimental.v4.train_and_evaluate_v4 import (
    evaluate_full_metrics,
    FROZEN_TEST_PATH,
    DEVICE,
)

FIGURES_DIR = Path("experimental/v4/results/step63_figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = Path("experimental/v4/results/step63_final_benchmark_results.npy")

def compute_checkpoint_hash(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def generate_eval_split(overrides: dict, num_samples: int = 100, base_seed: int = 777000):
    rng = np.random.default_rng(base_seed + int(overrides.get("snr_db", 0) * 10) + int(overrides.get("phase_sigma_deg", 0)))
    X_s, Y_s, ang_s = [], [], []

    min_ang = overrides.get("min_angle_deg", -60.0)
    max_ang = overrides.get("max_angle_deg", 60.0)
    is_split_angle = overrides.get("split_angles", False)

    for i in range(num_samples):
        sample_seed = base_seed + i
        if is_split_angle:
            # Sample off-boresight from [-65, -45] U [45, 65]
            if rng.uniform() > 0.5:
                angle = float(rng.uniform(45.0, 65.0))
            else:
                angle = float(rng.uniform(-65.0, -45.0))
        else:
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

def evaluate_v3_model(model, X, Y, ang):
    model.eval()
    C_true = Y[:, :64] + 1j * Y[:, 64:]
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32, device=DEVICE))
        c_pinn = torch.complex(pred[:, :64].to(torch.float64), pred[:, 64:].to(torch.float64)).cpu().numpy()
    return evaluate_full_metrics(c_pinn, C_true, ang), c_pinn

def evaluate_v4_model(model, X, Y, ang):
    model.eval()
    C_true = Y[:, :64] + 1j * Y[:, 64:]
    with torch.no_grad():
        c_hat_t, _, _ = model(torch.tensor(X, dtype=torch.float32, device=DEVICE))
        c_hat = c_hat_t.cpu().numpy()
    return evaluate_full_metrics(c_hat, C_true, ang), c_hat

def evaluate_preconditioner(delta: float, X, Y, ang):
    precond = FixedPhysicsPreconditioner(CONFIG, delta=delta).to(DEVICE)
    with torch.no_grad():
        c_prior_t, _ = precond(torch.tensor(X, dtype=torch.float32, device=DEVICE))
        c_prior = c_prior_t.cpu().numpy()
    C_true = Y[:, :64] + 1j * Y[:, 64:]
    return evaluate_full_metrics(c_prior, C_true, ang), c_prior

def plot_publication_figures(benchmark_data, datasets):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {
        "Analytical Baseline": "#2b5c8f",
        "Frozen PINN-v3": "#d95f02",
        "V4 Control (Delta=0)": "#7570b3",
        "V4 Tikhonov (Delta=0.10)": "#1b9e77",
        "V4 Tikhonov (Delta=0.15)": "#e7298a",
    }

    snr_levels = [5.0, 7.5, 10.0, 15.0, 20.0]
    snr_keys = ["5 dB SNR", "7.5 dB SNR", "10 dB SNR", "15 dB SNR", "Independent 20 dB"]

    # 1. NMSE vs SNR
    plt.figure(figsize=(8, 5), dpi=300)
    for model_name, col in colors.items():
        nmse_vals = [benchmark_data[model_name][k]["nmse_db"] for k in snr_keys]
        plt.plot(snr_levels, nmse_vals, marker="o", linewidth=2.2, label=model_name, color=col)
    plt.xlabel("Signal-to-Noise Ratio (SNR) [dB]", fontsize=12, fontweight="bold")
    plt.ylabel("Calibration NMSE [dB] (Lower is Better)", fontsize=12, fontweight="bold")
    plt.title("Calibration Accuracy vs SNR Robustness", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nmse_vs_snr.png")
    plt.close()

    # 2. Beam Correlation vs SNR
    plt.figure(figsize=(8, 5), dpi=300)
    for model_name, col in colors.items():
        corr_vals = [benchmark_data[model_name][k]["beam_correlation"] for k in snr_keys]
        plt.plot(snr_levels, corr_vals, marker="s", linewidth=2.2, label=model_name, color=col)
    plt.xlabel("Signal-to-Noise Ratio (SNR) [dB]", fontsize=12, fontweight="bold")
    plt.ylabel("Far-Field Beam Correlation (Higher is Better)", fontsize=12, fontweight="bold")
    plt.title("Far-Field Beam Correlation vs SNR", fontsize=13, fontweight="bold")
    plt.ylim(0.85, 1.002)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "beam_correlation_vs_snr.png")
    plt.close()

    # 3. Pointing Error vs SNR
    plt.figure(figsize=(8, 5), dpi=300)
    for model_name, col in colors.items():
        point_vals = [benchmark_data[model_name][k]["pointing_error_deg"] for k in snr_keys]
        plt.plot(snr_levels, point_vals, marker="^", linewidth=2.2, label=model_name, color=col)
    plt.xlabel("Signal-to-Noise Ratio (SNR) [dB]", fontsize=12, fontweight="bold")
    plt.ylabel("Beam Pointing Error [degrees] (Lower is Better)", fontsize=12, fontweight="bold")
    plt.title("Beam Pointing Stability vs SNR", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pointing_error_vs_snr.png")
    plt.close()

    # 4. Nominal Beam Pattern Comparison (Representative sample from Frozen 20 dB set)
    X_f, Y_f, ang_f = datasets["Frozen 20 dB"]
    sample_idx = 0
    target_ang = float(ang_f[sample_idx])

    # True hardware weights
    v_ideal_sample = steering_vector(torch.tensor([target_ang], dtype=torch.float64), CONFIG).to(torch.complex128)
    c_true_sample = torch.tensor(Y_f[sample_idx, :64] + 1j * Y_f[sample_idx, 64:], dtype=torch.complex128, device=DEVICE)
    v_hw_true = (v_ideal_sample.to(DEVICE) / (c_true_sample + 1e-12)).squeeze(0)

    # Patterns
    fine_angles = np.linspace(-90.0, 90.0, 721)
    fine_angles_t = torch.tensor(fine_angles, dtype=torch.float64, device=DEVICE)

    # Helper for array pattern
    def get_pattern(weights):
        steer_grid = steering_vector(fine_angles_t, CONFIG).to(torch.complex128)
        pat = torch.abs(torch.matmul(steer_grid, weights.unsqueeze(-1))).squeeze(-1)
        pat_db = 20.0 * torch.log10(pat / (torch.max(pat) + 1e-12))
        return pat_db.cpu().numpy()

    ideal_pat = get_pattern(v_ideal_sample.squeeze(0).to(DEVICE))
    impaired_pat = get_pattern(v_hw_true)
    ana_w = v_hw_true * torch.tensor(benchmark_data["Analytical Baseline"]["Frozen 20 dB"]["raw_C"][sample_idx], device=DEVICE)
    v3_w = v_hw_true * torch.tensor(benchmark_data["Frozen PINN-v3"]["Frozen 20 dB"]["raw_C"][sample_idx], device=DEVICE)
    v4_w = v_hw_true * torch.tensor(benchmark_data["V4 Tikhonov (Delta=0.15)"]["Frozen 20 dB"]["raw_C"][sample_idx], device=DEVICE)

    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(fine_angles, ideal_pat, "k--", label=f"Ideal Steering ({target_ang:.1f}°)", linewidth=1.8)
    plt.plot(fine_angles, get_pattern(ana_w), color=colors["Analytical Baseline"], label="Analytical Baseline", linewidth=1.5)
    plt.plot(fine_angles, get_pattern(v3_w), color=colors["Frozen PINN-v3"], label="Frozen PINN-v3", linewidth=1.5)
    plt.plot(fine_angles, get_pattern(v4_w), color=colors["V4 Tikhonov (Delta=0.15)"], label="V4 Tikhonov (Delta=0.15)", linewidth=2.0)
    plt.plot(fine_angles, impaired_pat, ":", color="#999999", label="Impaired (Uncalibrated)", linewidth=1.2)
    plt.xlabel("Spatial Angle [degrees]", fontsize=12, fontweight="bold")
    plt.ylabel("Normalized Power [dB]", fontsize=12, fontweight="bold")
    plt.title(f"Far-Field Synthesized Beam Pattern (Theta = {target_ang:.1f}°)", fontsize=13, fontweight="bold")
    plt.ylim(-40.0, 2.0)
    plt.xlim(-80.0, 80.0)
    plt.legend(frameon=True, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nominal_beam_pattern.png")
    plt.close()

    # 5. Calibration Error Comparison on Nominal 20 dB
    models_bar = ["Analytical Baseline", "Frozen PINN-v3", "V4 Control (Delta=0)", "V4 Tikhonov (Delta=0.10)", "V4 Tikhonov (Delta=0.15)"]
    nmses = [benchmark_data[m]["Frozen 20 dB"]["nmse_db"] for m in models_bar]
    phases = [benchmark_data[m]["Frozen 20 dB"]["phase_error_deg"] for m in models_bar]
    gains = [benchmark_data[m]["Frozen 20 dB"]["gain_error_db"] for m in models_bar]

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)
    bar_colors = [colors[m] for m in models_bar]

    ax[0].bar(range(len(models_bar)), nmses, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax[0].set_title("NMSE [dB] (Lower is Better)", fontweight="bold")
    ax[0].set_xticks(range(len(models_bar)))
    ax[0].set_xticklabels(["Analytical", "PINN-v3", "V4 d=0", "V4 d=0.10", "V4 d=0.15"], rotation=30, ha="right", fontsize=9)
    ax[0].grid(True, linestyle="--", alpha=0.6)

    ax[1].bar(range(len(models_bar)), phases, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax[1].set_title("Phase Error [deg] (Lower is Better)", fontweight="bold")
    ax[1].set_xticks(range(len(models_bar)))
    ax[1].set_xticklabels(["Analytical", "PINN-v3", "V4 d=0", "V4 d=0.10", "V4 d=0.15"], rotation=30, ha="right", fontsize=9)
    ax[1].grid(True, linestyle="--", alpha=0.6)

    ax[2].bar(range(len(models_bar)), gains, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax[2].set_title("Gain Error [dB] (Lower is Better)", fontweight="bold")
    ax[2].set_xticks(range(len(models_bar)))
    ax[2].set_xticklabels(["Analytical", "PINN-v3", "V4 d=0", "V4 d=0.10", "V4 d=0.15"], rotation=30, ha="right", fontsize=9)
    ax[2].grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Nominal 20 dB Calibration Quality Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_error_comparison.png")
    plt.close()

    # 6. Phase Stress Comparison (Phase 35 deg)
    plt.figure(figsize=(7, 4.5), dpi=300)
    phase_stress_nmses = [benchmark_data[m]["Phase 35 deg"]["nmse_db"] for m in models_bar]
    plt.bar(range(len(models_bar)), phase_stress_nmses, color=bar_colors, edgecolor="black", linewidth=0.8)
    plt.ylabel("NMSE [dB] (Lower is Better)", fontsize=11, fontweight="bold")
    plt.title("Severe Phase Error Stress (Phase Sigma = 35°)", fontsize=12, fontweight="bold")
    plt.xticks(range(len(models_bar)), ["Analytical", "PINN-v3", "V4 d=0", "V4 d=0.10", "V4 d=0.15"], rotation=25, ha="right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase_stress_comparison.png")
    plt.close()

    # 7. Coupling Stress Comparison (Coupling 40%)
    plt.figure(figsize=(7, 4.5), dpi=300)
    coupling_stress_nmses = [benchmark_data[m]["Coupling 40%"]["nmse_db"] for m in models_bar]
    plt.bar(range(len(models_bar)), coupling_stress_nmses, color=bar_colors, edgecolor="black", linewidth=0.8)
    plt.ylabel("NMSE [dB] (Lower is Better)", fontsize=11, fontweight="bold")
    plt.title("Severe Coupling Mismatch Stress (Mismatch = 40%)", fontsize=12, fontweight="bold")
    plt.xticks(range(len(models_bar)), ["Analytical", "PINN-v3", "V4 d=0", "V4 d=0.10", "V4 d=0.15"], rotation=25, ha="right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "coupling_stress_comparison.png")
    plt.close()

    print(f"All 7 publication-quality figures successfully saved to {FIGURES_DIR}/")

def main():
    print("=" * 85)
    print("PINN-CAL v4: STEP 63 FINAL EXHAUSTIVE BENCHMARK SUITE")
    print("=" * 85)

    # 1. Verify frozen v3 checkpoint integrity
    v3_path = Path("models/pinn_cal_final_best.pth")
    assert v3_path.exists(), f"Frozen model {v3_path} missing!"
    v3_hash = compute_checkpoint_hash(v3_path)
    print(f"Verified Frozen PINN-v3 Checkpoint SHA256: {v3_hash}")

    # 2. Load all model checkpoints
    seeds = [42, 123, 456]

    # Model A: Frozen v3
    v3_model = PINNCalV3().to(DEVICE)
    v3_model.load_state_dict(torch.load(v3_path, map_location=DEVICE, weights_only=True))

    # Model B: V4 Control (delta=0.00)
    v4_d0_models = {}
    for s in seeds:
        m = PINNCalV4(bounded=True, alpha=0.5, delta=0.00).to(DEVICE)
        m.load_state_dict(torch.load(f"experimental/v4/models/pinn_v4_bounded_seed{s}.pth", map_location=DEVICE, weights_only=True))
        v4_d0_models[s] = m

    # Model C: V4 Tikhonov delta=0.10
    v4_d10_models = {}
    for s in seeds:
        m = PINNCalV4(bounded=True, alpha=0.5, delta=0.10).to(DEVICE)
        m.load_state_dict(torch.load(f"experimental/v4/models/pinn_v4_tikhonov010_seed{s}.pth", map_location=DEVICE, weights_only=True))
        v4_d10_models[s] = m

    # Model D: V4 Tikhonov delta=0.15
    v4_d15_models = {}
    for s in seeds:
        m = PINNCalV4(bounded=True, alpha=0.5, delta=0.15).to(DEVICE)
        m.load_state_dict(torch.load(f"experimental/v4/models/pinn_v4_tikhonov015_seed{s}.pth", map_location=DEVICE, weights_only=True))
        v4_d15_models[s] = m

    # Load history stats for delta=0.15 seeds
    seed_training_info = {}
    for s in seeds:
        hist = np.load(f"experimental/v4/histories/pinn_v4_tikhonov015_seed{s}_history.npy", allow_pickle=True).item()
        best_ep = int(np.argmin(hist["val_loss"])) + 1
        seed_training_info[s] = {
            "best_epoch": best_ep,
            "val_loss": float(np.min(hist["val_loss"])),
            "train_loss": float(hist["train_loss"][best_ep - 1]),
            "gen_gap": float(hist["gen_gap"][best_ep - 1]),
        }

    print("\nCandidate B (Delta=0.15) Seed Consistency on Validation:")
    for s in seeds:
        ti = seed_training_info[s]
        print(f"  Seed {s:3d}: Best Epoch {ti['best_epoch']:2d} | Val Loss: {ti['val_loss']:.6f} | Train Loss: {ti['train_loss']:.6f} | Gap: {ti['gen_gap']:+.6f}")

    # 3. Build Evaluation Conditions
    eval_conditions = [
        ("Frozen 20 dB", None),  # Frozen benchmark
        ("Independent 20 dB", {"snr_db": 20.0}),
        ("15 dB SNR", {"snr_db": 15.0}),
        ("10 dB SNR", {"snr_db": 10.0}),
        ("7.5 dB SNR", {"snr_db": 7.5}),
        ("5 dB SNR", {"snr_db": 5.0}),
        ("Phase 35 deg", {"phase_sigma_deg": 35.0}),
        ("Gain 4.0 dB", {"gain_sigma_db": 4.0}),
        ("Coupling 40%", {"coupling_mismatch_fraction": 0.40}),
        ("Steering Stress", {"min_angle_deg": -65.0, "max_angle_deg": 65.0}),
        ("Combined Severe", {"phase_sigma_deg": 35.0, "gain_sigma_db": 4.0, "coupling_mismatch_fraction": 0.40, "snr_db": 10.0, "split_angles": True}),
    ]

    datasets = {}
    frozen_test = np.load(FROZEN_TEST_PATH, allow_pickle=True).item()
    datasets["Frozen 20 dB"] = (
        frozen_test["X"],
        frozen_test["Y"],
        frozen_test["steering_angles"],
    )

    for cond_name, overrides in eval_conditions[1:]:
        datasets[cond_name] = generate_eval_split(overrides, num_samples=100)

    # 4. Evaluate Benchmark
    results = {
        "Analytical Baseline": {},
        "Frozen PINN-v3": {},
        "V4 Control (Delta=0)": {},
        "V4 Tikhonov (Delta=0.10)": {},
        "V4 Tikhonov (Delta=0.15)": {},
        "V4 Delta=0.15 (Seed 42)": {},
        "V4 Delta=0.15 (Seed 123)": {},
        "V4 Delta=0.15 (Seed 456)": {},
        "Preconditioner Delta=0.00 Alone": {},
        "Preconditioner Delta=0.10 Alone": {},
        "Preconditioner Delta=0.15 Alone": {},
    }

    print("\n--- Running Evaluations across 11 Conditions ---")
    for cond_name, (X, Y, ang) in datasets.items():
        C_true = Y[:, :64] + 1j * Y[:, 64:]

        # A. Analytical Baseline
        c_ana = analytical_correction(X[:, :64] + 1j * X[:, 64:128], ang)
        m_ana = evaluate_full_metrics(c_ana, C_true, ang)
        results["Analytical Baseline"][cond_name] = {**m_ana, "raw_C": c_ana}

        # B. Frozen PINN-v3
        m_v3, c_v3 = evaluate_v3_model(v3_model, X, Y, ang)
        results["Frozen PINN-v3"][cond_name] = {**m_v3, "raw_C": c_v3}

        # C. Preconditioners Alone
        for d in [0.00, 0.10, 0.15]:
            m_p, c_p = evaluate_preconditioner(d, X, Y, ang)
            results[f"Preconditioner Delta={d:.2f} Alone"][cond_name] = {**m_p, "raw_C": c_p}

        # D. V4 Control (delta=0)
        v4_d0_metrics = []
        for s in seeds:
            m, c_hat = evaluate_v4_model(v4_d0_models[s], X, Y, ang)
            v4_d0_metrics.append(m)
        agg_d0 = {}
        for k in ["nmse_db", "phase_error_deg", "gain_error_db", "relative_error", "pointing_error_deg", "beam_correlation"]:
            agg_d0[k] = float(np.mean([x[k] for x in v4_d0_metrics]))
            agg_d0[f"{k}_std"] = float(np.std([x[k] for x in v4_d0_metrics]))
        results["V4 Control (Delta=0)"][cond_name] = agg_d0

        # E. V4 Tikhonov delta=0.10
        v4_d10_metrics = []
        for s in seeds:
            m, c_hat = evaluate_v4_model(v4_d10_models[s], X, Y, ang)
            v4_d10_metrics.append(m)
        agg_d10 = {}
        for k in ["nmse_db", "phase_error_deg", "gain_error_db", "relative_error", "pointing_error_deg", "beam_correlation"]:
            agg_d10[k] = float(np.mean([x[k] for x in v4_d10_metrics]))
            agg_d10[f"{k}_std"] = float(np.std([x[k] for x in v4_d10_metrics]))
        results["V4 Tikhonov (Delta=0.10)"][cond_name] = agg_d10

        # F. V4 Tikhonov delta=0.15 (individual seeds + aggregate)
        v4_d15_metrics = []
        c_hat_last = None
        for s in seeds:
            m, c_hat = evaluate_v4_model(v4_d15_models[s], X, Y, ang)
            results[f"V4 Delta=0.15 (Seed {s})"][cond_name] = m
            v4_d15_metrics.append(m)
            c_hat_last = c_hat
        agg_d15 = {}
        for k in ["nmse_db", "phase_error_deg", "gain_error_db", "relative_error", "pointing_error_deg", "beam_correlation"]:
            agg_d15[k] = float(np.mean([x[k] for x in v4_d15_metrics]))
            agg_d15[f"{k}_std"] = float(np.std([x[k] for x in v4_d15_metrics]))
        agg_d15["raw_C"] = c_hat_last
        results["V4 Tikhonov (Delta=0.15)"][cond_name] = agg_d15

        print(f"  Condition: {cond_name:18s} | Analytical={m_ana['nmse_db']:6.2f} dB | PINN-v3={m_v3['nmse_db']:6.2f} dB | V4(d=0)={agg_d0['nmse_db']:6.2f} dB | V4(d=0.15)={agg_d15['nmse_db']:6.2f} dB")

    # 5. Generate Figures
    print("\n--- Generating Publication-Quality Figures ---")
    plot_publication_figures(results, datasets)

    # 6. Save Complete Benchmark Results
    np.save(RESULTS_PATH, {
        "results": results,
        "v3_hash": v3_hash,
        "seed_training_info": seed_training_info,
        "eval_conditions": [c[0] for c in eval_conditions],
    })
    print(f"\nBenchmark results saved successfully to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
