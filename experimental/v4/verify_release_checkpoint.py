"""
Part A: Verify V4 Release Checkpoint.
"""
import sys
import hashlib
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experimental.v4.pinn_v4_model import PINNCalV4, CONFIG

def main():
    print("=" * 80)
    print("PART A: VERIFY V4 RELEASE CHECKPOINT")
    print("=" * 80)

    ckpt_path = Path("experimental/v4/models/pinn_v4_tikhonov015_seed42.pth")
    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} missing!"
    print(f"1. Checkpoint Path: {ckpt_path} (EXISTS)")

    # 2. SHA256 Checksum
    h = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    print(f"2. Checkpoint SHA256: {h}")
    expected_hash = "f5da358645e24536ce4f3c6144bc28f47f9a13bfc6c6744169f6ee055f2f9efe"
    assert h == expected_hash, f"Hash mismatch: {h} != {expected_hash}"
    print("   -> Exact match confirmed.")

    # 3. Model Architecture & Config
    delta = 0.15
    alpha = 0.50
    model = PINNCalV4(bounded=True, alpha=alpha, delta=delta)
    print(f"3. Model Architecture: {model.__class__.__name__}")
    print(f"4. Tikhonov delta: {model.physics_preconditioner.delta:.2f}")
    assert abs(model.physics_preconditioner.delta - delta) < 1e-9, "Delta mismatch"
    print(f"5. Residual bound alpha: {model.residual_denoiser.alpha:.2f}")
    assert abs(model.residual_denoiser.alpha - alpha) < 1e-9, "Alpha mismatch"
    print("6. Reference Seed: 42")

    # 7. Parameter counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Total parameters    : {total_params:,}")

    # 8. Load weights
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("8. Checkpoint load: SUCCESS")

    # Forward pass verification
    dummy = torch.randn(4, 130, dtype=torch.float32)
    with torch.no_grad():
        c_hat, delta_c, x_prior = model(dummy)
    print(f"   Input shape : (4, 130)")
    print(f"   c_hat shape : {c_hat.shape} (dtype: {c_hat.dtype})")
    print(f"   delta_c     : {delta_c.shape}")
    print(f"   x_prior     : {x_prior.shape}")
    assert c_hat.shape == (4, 64)
    assert delta_c.shape == (4, 128)
    assert x_prior.shape == (4, 128)
    assert torch.all(delta_c >= -alpha) and torch.all(delta_c <= alpha), "Residual bounds violated"
    print("   -> Forward pass & bounded residual integrity: 100% VERIFIED.")

if __name__ == "__main__":
    main()
