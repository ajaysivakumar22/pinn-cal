"""
PINN-Cal v4 Unit and Pipeline Test Suite
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from physics_core import DEFAULT_CONFIG
from analytical_baseline import nmse_db
from experimental.v4.pinn_v4_model import FixedPhysicsPreconditioner, BoundedResidualDenoiser, PINNCalV4, CONFIG, M

def test_fixed_physics_preconditioner_no_params():
    precond = FixedPhysicsPreconditioner(CONFIG)
    num_params = sum(p.numel() for p in precond.parameters())
    assert num_params == 0, f"Preconditioner must have 0 trainable parameters, found {num_params}"
    assert hasattr(precond, "Z_nom"), "Preconditioner must have Z_nom buffer"
    assert hasattr(precond, "Z_self"), "Preconditioner must have Z_self buffer"

def test_fixed_physics_preconditioner_forward():
    precond = FixedPhysicsPreconditioner(CONFIG)
    batch_size = 8
    # 130 inputs: 64 Re(I), 64 Im(I), sin(theta), cos(theta)
    x = torch.randn(batch_size, 130, dtype=torch.float32)
    # Ensure sin^2 + cos^2 = 1
    ang = torch.tensor([0.0] * batch_size)
    x[:, 128] = torch.sin(ang)
    x[:, 129] = torch.cos(ang)

    c_prior, x_prior = precond(x)
    assert c_prior.shape == (batch_size, 64)
    assert c_prior.dtype == torch.complex128
    assert x_prior.shape == (batch_size, 128)
    assert x_prior.dtype == torch.float32
    assert torch.all(torch.isfinite(c_prior))
    assert torch.all(torch.isfinite(x_prior))

def test_bounded_residual_denoiser_strict_bound():
    alpha = 0.5
    denoiser = BoundedResidualDenoiser(in_features=128, bounded=True, alpha=alpha)
    # Test on extreme inputs
    extreme_x = torch.randn(10, 128) * 1000.0
    r = denoiser(extreme_x)
    assert r.shape == (10, 128)
    assert torch.all(r >= -alpha), f"Residual breached lower bound {-alpha}: min={r.min().item()}"
    assert torch.all(r <= alpha), f"Residual breached upper bound {alpha}: max={r.max().item()}"

def test_v4_model_reconstruction_integrity():
    model = PINNCalV4(bounded=True, alpha=0.5)
    model.eval()
    x = torch.randn(4, 130, dtype=torch.float32)
    x[:, 128] = 0.0
    x[:, 129] = 1.0

    c_hat, delta_c_real, x_prior = model(x)
    c_prior = torch.complex(x_prior[:, :64].to(torch.float64), x_prior[:, 64:].to(torch.float64))
    delta_c = torch.complex(delta_c_real[:, :64].to(torch.float64), delta_c_real[:, 64:].to(torch.float64))

    # c_hat must be exactly c_prior + delta_c
    reconstructed = c_prior + delta_c
    diff = torch.abs(c_hat - reconstructed).max().item()
    assert diff < 1e-6, f"Reconstruction mismatch: {diff}"

def test_trained_models_exist_and_nmse():
    frozen_path = Path("data/physics_test_200.npy")
    assert frozen_path.exists(), "Frozen test set must exist"
    frozen = np.load(frozen_path, allow_pickle=True).item()
    X_frozen = frozen["X"]
    Y_frozen = frozen["Y"]
    C_true = Y_frozen[:, :64] + 1j * Y_frozen[:, 64:]

    for seed in [42, 123, 456]:
        ckpt = Path(f"experimental/v4/models/pinn_v4_bounded_seed{seed}.pth")
        assert ckpt.exists(), f"Checkpoint {ckpt} missing"

        model = PINNCalV4(bounded=True, alpha=0.5)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()

        with torch.no_grad():
            c_hat_t, _, _ = model(torch.tensor(X_frozen, dtype=torch.float32))
            c_hat = c_hat_t.numpy()

        nmse = nmse_db(c_hat, C_true)
        # All seeds must be around -14.2 dB, strictly better than -14.0 dB
        assert nmse <= -14.0, f"Seed {seed} NMSE ({nmse:.2f} dB) exceeds -14.0 dB threshold"

def test_adaptive_alpha_models_exist_and_nmse():
    frozen_path = Path("data/physics_test_200.npy")
    frozen = np.load(frozen_path, allow_pickle=True).item()
    X_frozen = frozen["X"]
    Y_frozen = frozen["Y"]
    C_true = Y_frozen[:, :64] + 1j * Y_frozen[:, 64:]

    for mode, prefix in [("mild", "pinn_v4_mild_adaptive"), ("strong", "pinn_v4_strong_adaptive")]:
        for seed in [42, 123, 456]:
            ckpt = Path(f"experimental/v4/models/{prefix}_seed{seed}.pth")
            assert ckpt.exists(), f"Checkpoint {ckpt} missing"

            model = PINNCalV4(bounded=True, alpha_mode=mode)
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            model.eval()

            with torch.no_grad():
                c_hat_t, delta_c_t, _, eff_alpha = model(
                    torch.tensor(X_frozen, dtype=torch.float32),
                    return_alpha=True,
                )
                c_hat = c_hat_t.numpy()

            nmse = nmse_db(c_hat, C_true)
            assert nmse <= -14.0, f"{prefix} seed {seed} NMSE ({nmse:.2f} dB) exceeds -14.0 dB threshold"
            assert eff_alpha.min() >= (0.25 if mode == "strong" else 0.30)
            assert eff_alpha.max() <= (1.00 if mode == "strong" else 0.80)

def test_tikhonov_combined_models_exist_and_nmse():
    frozen_path = Path("data/physics_test_200.npy")
    frozen = np.load(frozen_path, allow_pickle=True).item()
    X_frozen = frozen["X"]
    Y_frozen = frozen["Y"]
    C_true = Y_frozen[:, :64] + 1j * Y_frozen[:, 64:]

    for delta, prefix, threshold in [(0.10, "pinn_v4_tikhonov010", -14.50), (0.15, "pinn_v4_tikhonov015", -14.80)]:
        for seed in [42, 123, 456]:
            ckpt = Path(f"experimental/v4/models/{prefix}_seed{seed}.pth")
            assert ckpt.exists(), f"Checkpoint {ckpt} missing"

            model = PINNCalV4(bounded=True, alpha=0.5, delta=delta)
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            model.eval()

            with torch.no_grad():
                c_hat_t, _, _ = model(torch.tensor(X_frozen, dtype=torch.float32))
                c_hat = c_hat_t.numpy()

            nmse = nmse_db(c_hat, C_true)
            assert nmse <= threshold, f"{prefix} seed {seed} NMSE ({nmse:.2f} dB) exceeds {threshold} dB threshold"


