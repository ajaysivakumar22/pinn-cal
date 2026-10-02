import numpy as np

data = np.load("experimental/v4/results/step63_final_benchmark_results.npy", allow_pickle=True).item()
res = data["results"]
seed_info = data["seed_training_info"]

print("=== Candidate B (Delta=0.15) Seed Comparison ===")
for s in [42, 123, 456]:
    f20 = res[f"V4 Delta=0.15 (Seed {s})"]["Frozen 20 dB"]
    s10 = res[f"V4 Delta=0.15 (Seed {s})"]["10 dB SNR"]
    s5  = res[f"V4 Delta=0.15 (Seed {s})"]["5 dB SNR"]
    ti = seed_info[s]
    print(f"Seed {s:3d}: ValLoss={ti['val_loss']:.6f} | Frozen20: NMSE={f20['nmse_db']:.4f} dB, Phase={f20['phase_error_deg']:.2f}°, Point={f20['pointing_error_deg']:.4f}°, Corr={f20['beam_correlation']:.6f} | 10dB: NMSE={s10['nmse_db']:.4f} dB | 5dB: NMSE={s5['nmse_db']:.4f} dB")

print("\n=== Preconditioner Alone vs Neural Combined (Delta=0.15) ===")
for c in ["Frozen 20 dB", "10 dB SNR", "5 dB SNR", "Coupling 40%", "Combined Severe"]:
    p = res["Preconditioner Delta=0.15 Alone"][c]
    m = res["V4 Tikhonov (Delta=0.15)"][c]
    d_nmse = m["nmse_db"] - p["nmse_db"]
    d_phase = m["phase_error_deg"] - p["phase_error_deg"]
    d_point = m["pointing_error_deg"] - p["pointing_error_deg"]
    d_corr = m["beam_correlation"] - p["beam_correlation"]
    print(f"{c:18s} | Precond NMSE={p['nmse_db']:6.2f} dB -> Combined={m['nmse_db']:6.2f} dB (Added value: {d_nmse:+.2f} dB) | Pointing: {p['pointing_error_deg']:.4f}° -> {m['pointing_error_deg']:.4f}° ({d_point:+.4f}°) | Corr: {p['beam_correlation']:.6f} -> {m['beam_correlation']:.6f} ({d_corr:+.6f})")

print("\n=== All Models Across All 11 Conditions ===")
models = ["Analytical Baseline", "Frozen PINN-v3", "V4 Control (Delta=0)", "V4 Tikhonov (Delta=0.10)", "V4 Tikhonov (Delta=0.15)"]
conditions = data["eval_conditions"]
for c in conditions:
    print(f"\n--- {c} ---")
    for m in models:
        r = res[m][c]
        std_str = f" ± {r['nmse_db_std']:.2f}" if "nmse_db_std" in r else ""
        print(f"  {m:25s}: NMSE={r['nmse_db']:6.2f}{std_str:8s} dB | Phase={r['phase_error_deg']:5.2f}° | Gain={r['gain_error_db']:5.2f} dB | Point={r['pointing_error_deg']:.4f}° | Corr={r['beam_correlation']:.6f}")
