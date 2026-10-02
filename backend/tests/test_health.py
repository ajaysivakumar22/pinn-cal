"""
PINN-Cal Backend Tests — Health and System Status
"""

import sys
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/api/health")
    assert response.status_code == 200


def test_health_has_required_fields():
    response = client.get("/api/health")
    data = response.json()
    assert "status" in data
    assert "uptime_s" in data
    assert "pinn_model_loaded" in data
    assert "scientific_core_available" in data


def test_system_status_has_pytorch():
    response = client.get("/api/system-status")
    assert response.status_code == 200
    data = response.json()
    assert "pytorch_version" in data
    assert "cuda_available" in data
    assert "python_version" in data


def test_model_status():
    response = client.get("/api/model-status")
    assert response.status_code == 200
    data = response.json()
    assert data["total_parameters"] == 33088
    assert data["best_epoch"] == 29
    assert data["train_samples"] == 2000
    assert "validated_results" in data


def test_config_endpoint():
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert data["pinn_compatible"]["num_elements"] == 64
    assert data["pinn_compatible"]["frequency_hz"] == 28e9


# ===========================================================================
# Focused V4 Integration Tests (Step 70)
# ===========================================================================

import numpy as np
import torch
import pytest
from backend.app.scientific.adapter import PINNCalV4, DEFAULT_CONFIG
from backend.app.services.model_service import ModelService
from backend.app.core.config import (
    MODEL_PARAMS,
    TIKHONOV_DELTA,
    RESIDUAL_ALPHA,
)


def test_v4_checkpoint_loads_successfully():
    """1. V4 checkpoint loads successfully without errors."""
    svc = ModelService.get()
    assert svc.loaded is True
    assert svc.load_error is None
    assert svc.model is not None


def test_v4_model_class_is_pinncalv4():
    """2. Model class is PINNCalV4 from the validated V4 implementation."""
    svc = ModelService.get()
    assert isinstance(svc.model, PINNCalV4)


def test_v4_parameter_count_is_33088():
    """3. Trainable and total parameter count equals exactly 33,088."""
    svc = ModelService.get()
    total_params = sum(p.numel() for p in svc.model.parameters())
    trainable_params = sum(p.numel() for p in svc.model.parameters() if p.requires_grad)
    assert total_params == 33088
    assert trainable_params == 33088
    assert svc.total_parameters == 33088
    assert MODEL_PARAMS == 33088


def test_v4_inference_accepts_shape_n_130():
    """4. Inference accepts input tensor with shape (N, 130)."""
    svc = ModelService.get()
    N = 4
    x_np = np.random.randn(N, 130).astype(np.float32)
    # Test through model_service.predict
    pred = svc.predict(x_np)
    assert pred.shape == (N, 128)

    # Test through raw model forward
    device = next(svc.model.parameters()).device
    x_t = torch.from_numpy(x_np).to(device)
    with torch.no_grad():
        c_hat, delta_c, x_prior = svc.model(x_t)
    assert c_hat.shape == (N, 64)


def test_v4_c_hat_shape_is_n_64():
    """5. Final correction coefficients c_hat has shape (N, 64)."""
    svc = ModelService.get()
    N = 3
    x_t = torch.randn(N, 130, device=next(svc.model.parameters()).device)
    with torch.no_grad():
        c_hat, _, _ = svc.model(x_t)
    assert c_hat.shape == (N, 64)


def test_v4_c_hat_is_complex_valued():
    """6. c_hat is complex-valued (torch.complex64 or torch.complex128)."""
    svc = ModelService.get()
    x_t = torch.randn(2, 130, device=next(svc.model.parameters()).device)
    with torch.no_grad():
        c_hat, _, _ = svc.model(x_t)
    assert torch.is_complex(c_hat)
    assert c_hat.dtype in (torch.complex64, torch.complex128, torch.cfloat, torch.cdouble)


def test_v4_residual_is_bounded_by_pm_050():
    """7. Neural residual representation is strictly bounded within [-0.50, +0.50]."""
    svc = ModelService.get()
    # High-magnitude inputs to push tanh towards limits
    x_extreme = torch.randn(10, 130, device=next(svc.model.parameters()).device) * 50.0
    with torch.no_grad():
        _, delta_c_real, _ = svc.model(x_extreme)
    assert delta_c_real.shape == (10, 128)
    assert torch.all(delta_c_real >= -0.50 - 1e-6)
    assert torch.all(delta_c_real <= 0.50 + 1e-6)


def test_v4_model_runs_on_cpu():
    """8. Model runs successfully on CPU."""
    svc = ModelService.get()
    cpu_model = PINNCalV4(
        config=DEFAULT_CONFIG,
        bounded=True,
        alpha=RESIDUAL_ALPHA,
        delta=TIKHONOV_DELTA,
    ).to("cpu")
    cpu_model.load_state_dict(svc.model.state_dict())
    cpu_model.eval()

    x = torch.randn(2, 130, device="cpu")
    with torch.no_grad():
        c_hat, delta_c, x_prior = cpu_model(x)
    assert c_hat.device.type == "cpu"
    assert c_hat.shape == (2, 64)


def test_v4_model_runs_on_cuda_when_available():
    """9. Model runs on CUDA when available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available in current test environment")
    
    svc = ModelService.get()
    cuda_model = PINNCalV4(
        config=DEFAULT_CONFIG,
        bounded=True,
        alpha=RESIDUAL_ALPHA,
        delta=TIKHONOV_DELTA,
    ).to("cuda")
    cuda_model.load_state_dict(svc.model.state_dict())
    cuda_model.eval()

    x = torch.randn(2, 130, device="cuda")
    with torch.no_grad():
        c_hat, delta_c, x_prior = cuda_model(x)
    assert c_hat.device.type == "cuda"
    assert c_hat.shape == (2, 64)


def test_v4_repeated_inference_is_deterministic_in_eval_mode():
    """10. Repeated inference in eval mode is strictly deterministic."""
    svc = ModelService.get()
    device = next(svc.model.parameters()).device
    x = torch.randn(4, 130, device=device)
    with torch.no_grad():
        c_hat1, delta_c1, prior1 = svc.model(x)
        c_hat2, delta_c2, prior2 = svc.model(x)
    assert torch.equal(c_hat1, c_hat2)
    assert torch.equal(delta_c1, delta_c2)
    assert torch.equal(prior1, prior2)


def test_v4_health_endpoint_reports_v4():
    """11. System endpoints report V4 model metadata."""
    # Health endpoint
    h_resp = client.get("/api/health")
    assert h_resp.status_code == 200
    h_data = h_resp.json()
    assert h_data["pinn_model_loaded"] is True
    assert h_data["pinn_model_error"] is None

    # Model status endpoint
    m_resp = client.get("/api/model-status")
    assert m_resp.status_code == 200
    m_data = m_resp.json()
    assert "V4" in m_data["model_name"]
    assert "4.0.0" in m_data["version"]
    assert m_data["total_parameters"] == 33088
    assert m_data["trainable_parameters"] == 33088
    assert m_data["architecture"] == "PINNCalV4 (PhysicsPreconditioner + BoundedResidualDenoiser)"
    assert m_data["tikhonov_delta"] == 0.15
    assert m_data["residual_alpha"] == 0.50
    assert "pinn_v4_tikhonov015_seed42.pth" in m_data["checkpoint_path"]


def test_v4_calibration_endpoint_successfully_completes():
    """12. Calibration endpoint successfully completes inference using V4."""
    req = {
        "num_elements": 64,
        "frequency_hz": 28e9,
        "element_spacing_lambda": 0.5,
        "target_angle_deg": 15.0,
        "phase_sigma_deg": 15.0,
        "gain_sigma_db": 1.5,
        "snr_db": 20.0,
        "coupling_mismatch_fraction": 0.20,
        "seed": 42,
    }
    resp = client.post("/api/simulation", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert data["pinn_available"] is True
    assert data["calibration"]["pinn"] is not None
    pinn_res = data["calibration"]["pinn"]
    assert np.isfinite(pinn_res["nmse_db"])
    assert np.isfinite(pinn_res["mean_gain_error_db"])
    assert np.isfinite(pinn_res["mean_phase_error_deg"])
    assert len(data["elements"]) == 64
    for elem in data["elements"]:
        assert "pinn_correction_real" in elem
        assert "pinn_correction_imag" in elem
        assert np.isfinite(elem["pinn_correction_real"])
        assert np.isfinite(elem["pinn_correction_imag"])

