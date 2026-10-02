import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from physics_core import DEFAULT_CONFIG, steering_vector, mutual_impedance_matrix

def main():
    train_data = np.load("experimental/v4/datasets/v4_train_2000.npy", allow_pickle=True).item()
    X = train_data["X"]
    true_snr = train_data["snr_db"]
    angles = train_data["steering_angles"]

    Z = mutual_impedance_matrix(DEFAULT_CONFIG).to(torch.complex128)
    Z_self = Z.diagonal().mean()

    # Reconstruct I_obs and V_eff = I_obs @ Z.T
    I_obs = torch.complex(torch.tensor(X[:, :64], dtype=torch.float64), torch.tensor(X[:, 64:128], dtype=torch.float64))
    angles_t = torch.tensor(angles, dtype=torch.float64)
    V_ideal = steering_vector(angles_t, DEFAULT_CONFIG).to(torch.complex128)

    # 1. Project V_eff onto V_ideal
    V_eff = I_obs @ Z.T
    dot = torch.sum(V_eff * torch.conj(V_ideal), dim=1)
    v_eff_norm = torch.linalg.norm(V_eff, dim=1)
    v_ideal_norm = torch.linalg.norm(V_ideal, dim=1)
    rho = torch.abs(dot) / (v_eff_norm * v_ideal_norm + 1e-12)

    # Coherent signal power vs orthogonal disturbance power
    p_sig = torch.abs(dot)**2 / (v_ideal_norm**2 + 1e-12)
    p_total = v_eff_norm**2
    p_orth = torch.clamp(p_total - p_sig, min=1e-8)
    proxy_snr_db = 10.0 * torch.log10(p_sig / p_orth)

    corr = float(np.corrcoef(true_snr, proxy_snr_db.numpy())[0, 1])
    corr_rho = float(np.corrcoef(true_snr, rho.numpy())[0, 1])
    print(f"Correlation of proxy_snr_db with true SNR: {corr:.4f}")
    print(f"Correlation of rho with true SNR: {corr_rho:.4f}")
    print(f"True SNR range: [{true_snr.min():.2f}, {true_snr.max():.2f}] dB")
    print(f"Proxy SNR range: [{proxy_snr_db.numpy().min():.2f}, {proxy_snr_db.numpy().max():.2f}] dB")

    # Let's inspect mean proxy across true SNR bins
    rho_np = rho.numpy()
    print("rho percentiles:")
    print(f"  1%: {np.percentile(rho_np, 1):.4f}, 5%: {np.percentile(rho_np, 5):.4f}, 50%: {np.percentile(rho_np, 50):.4f}, 95%: {np.percentile(rho_np, 95):.4f}, 99%: {np.percentile(rho_np, 99):.4f}")

    # Test candidate smooth mapping:
    # Let rho_min = 0.80, rho_max = 0.92
    # w = torch.clamp((rho - 0.80) / (0.92 - 0.80), 0.0, 1.0)
    # Using smooth sigmoid or smoothstep:
    # smooth_w = w * w * (3.0 - 2.0 * w)  # Hermite smoothstep
    w = torch.clamp((rho - 0.80) / 0.12, 0.0, 1.0)
    smooth_w = w * w * (3.0 - 2.0 * w)
    
    # Candidate 2: [0.30, 0.80]
    alpha_c2 = 0.30 + (0.80 - 0.30) * smooth_w
    # Candidate 3: [0.25, 1.00]
    alpha_c3 = 0.25 + (1.00 - 0.25) * smooth_w

    print("\nAlpha Candidate 2 (Mild [0.30, 0.80]):")
    print(f"  Overall range: [{alpha_c2.min():.4f}, {alpha_c2.max():.4f}], mean={alpha_c2.mean():.4f}")
    print("\nAlpha Candidate 3 (Strong [0.25, 1.00]):")
    print(f"  Overall range: [{alpha_c3.min():.4f}, {alpha_c3.max():.4f}], mean={alpha_c3.mean():.4f}")

    for b_low, b_high in [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]:
        mask = (true_snr >= b_low) & (true_snr < b_high)
        print(f"  True SNR [{b_low:2d}, {b_high:2d}) dB: C2 mean alpha = {alpha_c2[mask].mean():.4f}, C3 mean alpha = {alpha_c3[mask].mean():.4f}")


if __name__ == "__main__":
    main()
