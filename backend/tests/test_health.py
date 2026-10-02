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
    assert data["total_parameters"] == 83904
    assert data["best_epoch"] == 40
    assert data["train_samples"] == 800
    assert "validated_results" in data


def test_config_endpoint():
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert data["pinn_compatible"]["num_elements"] == 64
    assert data["pinn_compatible"]["frequency_hz"] == 28e9
