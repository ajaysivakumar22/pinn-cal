import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from physics_core import DEFAULT_CONFIG, steering_vector, mutual_impedance_matrix

def analyze_noise_propagation():
    config = DEFAULT_CONFIG
    Z = mutual_impedance_matrix(config).to(torch.complex128)
    Z_self = Z.diagonal().mean()

    print("=== Nominal Impedance Statistics ===")
    print(f"Z_self: {Z_self.item():.4f}")
    # Row norms of Z
    row_norms = torch.linalg.norm(Z, dim=1)
    print(f"Mean row norm of Z: {row_norms.mean().item():.4f}, min={row_norms.min().item():.4f}, max={row_norms.max().item():.4f}")

    # Now let us inspect V_hw distribution on the multi-SNR training set and frozen test set
    train_data = np.load("experimental/v4/datasets/v4_train_2000.npy", allow_pickle=True).item()
    X = train_data["X"]
    snrs = train_data["snr_db"]
    angles = train_data["steering_angles"]

    I_obs = torch.complex(torch.tensor(X[:, :64], dtype=torch.float64), torch.tensor(X[:, 64:128], dtype=torch.float64))
    V_eff = I_obs @ Z.T
    V_hw = V_eff / Z_self
    v_hw_mag = torch.abs(V_hw)

    print("\n=== V_hw Magnitude Distribution Across Train Dataset ===")
    v_np = v_hw_mag.numpy().flatten()
    print(f"min={v_np.min():.6e}, 0.1%={np.percentile(v_np, 0.1):.4f}, 1%={np.percentile(v_np, 1.0):.4f}, median={np.median(v_np):.4f}, mean={v_np.mean():.4f}, max={v_np.max():.4f}")

    # Check V_hw magnitude by SNR bin
    print("\nV_hw min & 1st percentile by SNR:")
    for b_low, b_high in [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]:
        mask = (snrs >= b_low) & (snrs < b_high)
        sub_v = v_hw_mag[mask].numpy().flatten()
        print(f"  SNR [{b_low:2d}, {b_high:2d}) dB: min={sub_v.min():.6e}, 0.1%={np.percentile(sub_v, 0.1):.4f}, 1%={np.percentile(sub_v, 1.0):.4f}, median={np.median(sub_v):.4f}")

    # Physical delta_noise derivation:
    # At nominal excitation, |V_ideal| = 1.0 on all elements.
    # When hardware errors are applied (phase sigma 20 deg, gain sigma 2 dB), E[|V_hw|^2] ~ 1.0.
    # At SNR_dB, signal power P_sig ~ 1.0, noise power sigma_n^2 = 10^(-SNR_dB / 10).
    # At 20 dB SNR: sigma_n^2 = 0.01 -> sigma_n = 0.10.
    # At 10 dB SNR: sigma_n^2 = 0.10 -> sigma_n = 0.316.
    # At 5 dB SNR: sigma_n^2 = 0.316 -> sigma_n = 0.562.
    # For a nominal fixed Tikhonov regularizer designed for typical nominal array operation (20 dB):
    # delta_noise_20db = 10^(-20/20) = 0.10 -> delta^2 = 0.01.
    # Or for mid-range (15 dB): delta_noise_15db = 10^(-15/20) ~ 0.177 -> delta^2 ~ 0.0316.
    # Let us compute the effect of delta on |C_prior|!
    V_ideal = steering_vector(torch.tensor(angles, dtype=torch.float64), config).to(torch.complex128)
    Y = train_data["Y"]
    C_true = torch.complex(torch.tensor(Y[:, :64], dtype=torch.float64), torch.tensor(Y[:, 64:], dtype=torch.float64))

    print("\n=== Testing Preconditioner with Various delta ===")
    for delta in [0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30]:
        denom = torch.abs(V_hw)**2 + delta**2
        if delta == 0.0:
            C_prior = V_ideal / (V_hw + 1e-12)
        else:
            C_prior = (V_ideal * torch.conj(V_hw)) / denom

        c_mag = torch.abs(C_prior).numpy().flatten()
        err = (torch.abs(C_prior - C_true)**2).numpy()
        sample_mse = np.mean(err, axis=1) / 2.0  # real/imag MSE

        print(f"delta = {delta:5.2f} | max |C| = {c_mag.max():8.2f} | 99.9% = {np.percentile(c_mag, 99.9):6.2f} | 99% = {np.percentile(c_mag, 99):5.2f} | mean MSE = {np.mean(sample_mse):8.4f} | median MSE = {np.median(sample_mse):6.4f}")

if __name__ == "__main__":
    analyze_noise_propagation()
