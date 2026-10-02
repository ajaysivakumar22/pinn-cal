"""
PINN-Cal Backend Tests — Simulation Endpoint
"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

STANDARD_REQUEST = {
    "num_elements": 64,
    "frequency_hz": 28e9,
    "element_spacing_lambda": 0.5,
    "target_angle_deg": 20.0,
    "phase_sigma_deg": 20.0,
    "gain_sigma_db": 2.0,
    "snr_db": 20.0,
    "coupling_mismatch_fraction": 0.20,
    "seed": 42,
}


def test_simulation_returns_200():
    response = client.post("/api/simulation", json=STANDARD_REQUEST)
    assert response.status_code == 200, response.text


def test_simulation_has_64_elements():
    response = client.post("/api/simulation", json=STANDARD_REQUEST)
    data = response.json()
    assert len(data["elements"]) == 64


def test_simulation_element_has_required_fields():
    response = client.post("/api/simulation", json=STANDARD_REQUEST)
    data = response.json()
    elem = data["elements"][0]
    assert "element_index" in elem
    assert "ideal_magnitude" in elem
    assert "gain_error_db" in elem
    assert "phase_error_deg" in elem
    assert "observed_real" in elem
    assert "analytical_correction_real" in elem


def test_simulation_pinn_available_for_64_elements():
    response = client.post("/api/simulation", json=STANDARD_REQUEST)
    data = response.json()
    assert data["pinn_available"] is True


def test_simulation_pinn_unavailable_for_other_config():
    req = {**STANDARD_REQUEST, "num_elements": 32}
    response = client.post("/api/simulation", json=req)
    data = response.json()
    assert data["pinn_available"] is False
    assert "PINN" in data["pinn_compatibility_message"]


def test_simulation_beam_has_four_traces():
    response = client.post("/api/simulation", json=STANDARD_REQUEST)
    data = response.json()
    beam = data["beam"]
    assert "ideal" in beam
    assert "uncalibrated" in beam
    assert "analytical" in beam
    assert "pinn" in beam


def test_simulation_beam_angles_correct_length():
    response = client.post("/api/simulation", json=STANDARD_REQUEST)
    data = response.json()
    from backend.app.core.config import BEAM_ANGLE_POINTS
    assert len(data["beam"]["ideal"]["angles_deg"]) == BEAM_ANGLE_POINTS


def test_simulation_calibration_metrics_finite():
    response = client.post("/api/simulation", json=STANDARD_REQUEST)
    data = response.json()
    ana = data["calibration"]["analytical"]
    assert math.isfinite(ana["nmse_db"])
    assert math.isfinite(ana["mean_phase_error_deg"])
    assert math.isfinite(ana["mean_gain_error_db"])
    assert math.isfinite(ana["mean_relative_error_pct"])


def test_simulation_pinn_metrics_finite():
    response = client.post("/api/simulation", json=STANDARD_REQUEST)
    data = response.json()
    if data["pinn_available"] and data["calibration"]["pinn"]:
        pinn = data["calibration"]["pinn"]
        assert math.isfinite(pinn["nmse_db"])
        assert math.isfinite(pinn["mean_phase_error_deg"])


def test_simulation_rejects_invalid_angle():
    req = {**STANDARD_REQUEST, "target_angle_deg": 91.0}
    response = client.post("/api/simulation", json=req)
    assert response.status_code == 422


def test_simulation_rejects_negative_snr_out_of_range():
    req = {**STANDARD_REQUEST, "snr_db": -20.0}
    response = client.post("/api/simulation", json=req)
    assert response.status_code == 422
