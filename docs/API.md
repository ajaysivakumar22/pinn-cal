# PINN-Cal REST API Specification

PINN-Cal provides a RESTful HTTP API documented interactively via OpenAPI / Swagger UI at `/docs`.

Base URL: `http://localhost:8000/api`

---

## 1. System Endpoints

### `GET /api/health`
Health and liveness probe. Checks scientific core availability and model load state.

**Response (200 OK):**
```json
{
  "status": "ok",
  "timestamp": 1790184000.0,
  "uptime_s": 42.5,
  "scientific_core_available": true,
  "scientific_core_error": null,
  "pinn_model_loaded": true,
  "pinn_model_error": null
}
```

---

### `GET /api/system-status`
Live telemetry of runtime environment and hardware acceleration.

**Response (200 OK):**
```json
{
  "python_version": "3.11.15",
  "pytorch_version": "2.5.1+cu121",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce GTX 1650",
  "cuda_device_count": 1,
  "platform": "Windows-10-...",
  "scientific_core_available": true,
  "scientific_core_error": null,
  "backend_uptime_s": 54.2
}
```

---

### `GET /api/model-status`
Comprehensive metadata for the frozen PINN checkpoint (`models/pinn_cal_final_best.pth`).

**Response (200 OK):**
```json
{
  "model_name": "PINNCalFinalNet",
  "architecture": "130 -> 256 -> 128 -> 64 -> 128",
  "input_dim": 130,
  "output_dim": 128,
  "hidden_dims": [256, 128, 64],
  "total_parameters": 83904,
  "checkpoint": "models/pinn_cal_final_best.pth",
  "checkpoint_available": true,
  "best_epoch": 40,
  "train_samples": 800,
  "val_samples": 200,
  "test_samples": 200,
  "device": "cuda:0",
  "loaded": true,
  "load_error": null,
  "load_time_s": 0.35,
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce GTX 1650",
  "pinn_trained_for": {
    "num_elements": 64,
    "frequency_hz": 28000000000.0,
    "frequency_label": "28 GHz",
    "note": "Current checkpoint trained specifically for 64-element ULA at 28 GHz."
  },
  "validated_results": {
    "analytical": {
      "nmse_db": -14.4545,
      "mean_phase_error_deg": 3.9201,
      "mean_gain_error_db": 1.0404,
      "mean_relative_error_pct": 15.1264,
      "current_residual_pct": 15.1264
    },
    "pinn": {
      "nmse_db": -7.7489,
      "mean_phase_error_deg": 16.199,
      "mean_gain_error_db": 1.7084,
      "mean_relative_error_pct": 36.7887,
      "current_residual_pct": 36.7887
    }
  },
  "honesty_note": "Analytical baseline achieves lower error on frozen test set."
}
```

---

## 2. Core Simulation Endpoints

### `POST /api/simulation`
Unified dashboard endpoint performing array excitation, hardware impairment, mutual coupling, calibration inference, beam synthesis, and summary telemetry in a single round-trip.

**Request Body:**
```json
{
  "num_elements": 64,
  "frequency_hz": 28000000000.0,
  "element_spacing_lambda": 0.5,
  "target_angle_deg": 0.0,
  "phase_sigma_deg": 20.0,
  "gain_sigma_db": 2.0,
  "snr_db": 20.0,
  "coupling_mismatch_fraction": 0.2,
  "seed": 42
}
```

**Response (200 OK):**
Returns `SimulationResponse` containing:
- `pinn_available`: `true` if $N=64$ and $f=28\text{ GHz}$; `false` otherwise.
- `elements`: Array of 64 `ElementState` objects (magnitudes, phases, observed currents, analytical corrections, PINN corrections).
- `beam`: Angular traces (`angles_deg` and `power_db`) for Ideal, Uncalibrated, Analytical, and PINN beams.
- `calibration`: Comparison metrics (`nmse_db`, `mean_phase_error_deg`, `mean_gain_error_db`, `mean_relative_error_pct`).
- `metrics`: Peak pointing angle, pointing error, correlation, 3dB beamwidth, and PSLL.

---

### `POST /api/robustness`
Performs parameter sweeps over held-out Monte Carlo samples to validate calibration stability.

**Request Body:**
```json
{
  "sweep_type": "snr",
  "num_samples": 50,
  "seed": 4200
}
```

Supported `sweep_type`:
- `"snr"`: Evaluates across $[40, 30, 20, 15, 10]\text{ dB}$.
- `"coupling"`: Evaluates across $[0\%, 10\%, 20\%, 30\%, 50\%]$ mismatch.
- `"steering"`: Evaluates across 5 angular sectors ($[-60^\circ, +60^\circ]$).
