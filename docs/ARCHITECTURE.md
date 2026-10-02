# PINN-Cal Application Architecture

## 1. System Overview

**PINN-Cal** (Physics-Informed Neural Calibration for 5G Massive MIMO Antenna Arrays) is an end-to-end engineering instrument built around a frozen, scientifically validated research core (`scientific-core-v1.0`).

The system couples electromagnetic antenna array physics with real-time neural network inference to calibrate mutual coupling impairments and hardware gain/phase mismatches in 64-element uniform linear arrays (ULA) operating at 28 GHz mmWave.

```
┌─────────────────────────────────────────────────────────────┐
│                 React + TypeScript Frontend                 │
│  (Vite · Zustand · Plotly · SVG 64-Element Live Array)      │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / JSON
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                         │
│   (Pydantic v2 Schemas · Dynamic PyTorch / CUDA Runtime)    │
└──────────────┬───────────────────────────────┬──────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────────┐ ┌────────────────────────────┐
│      Scientific Adapter      │ │    Model Service (Cache)   │
│ (Mutual coupling & Z matrix) │ │ (pinn_cal_final_best.pth)  │
└──────────────┬───────────────┘ └─────────────┬──────────────┘
               │                               │
               └───────────────┬───────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Frozen Scientific Core (v1.0)                  │
│   (physics_core.py, coupling_model.py, pinn_cal_final.py)   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Architectural Layers

### 2.1 Scientific Core Layer (`D:\pinn-cal\*.py`)
- **Status:** Protected and Frozen (`scientific-core-v1.0`).
- **Files:** `physics_core.py`, `coupling_model.py`, `split_physics_dataset.py`, `analytical_baseline.py`, `pinn_cal_final.py`, `PHYSICAL_PROBLEM_SPEC.md`.
- **Functionality:** Provides analytical compact reciprocal network impedance matrices ($Z_{\text{nom}}$, $Z_{\text{true}}$), mutual coupling matrix synthesis, steering vectors ($a_n(\theta) = \exp(j k x_n \sin\theta)$), hardware impairments ($\sigma_\phi = 20^\circ, \sigma_g = 2\text{ dB}$), and the PyTorch feedforward PINN model definition (`PINNCalFinalNet`).

### 2.2 Scientific Adapter Layer (`backend/app/scientific/adapter.py`)
- Isolates backend API logic from research scripts without modifying core files.
- Converts between domain entities and standardized Pydantic schemas.
- Computes radiation beam patterns ($P(\theta)$) over 361 angular samples spanning $[-90^\circ, +90^\circ]$.
- Computes metrics: Pointing Error, Beam Correlation, Half-Power Beamwidth (HPBW), Peak Sidelobe Level (PSLL), Normalized Mean Squared Error (NMSE), and Current Residuals.

### 2.3 Model Management Service (`backend/app/services/model_service.py`)
- Singleton pattern: loads `models/pinn_cal_final_best.pth` once at application startup.
- Uses `torch.load(..., weights_only=True)` to ensure secure deserialization.
- Automatically selects GPU (`cuda:0`, NVIDIA GeForce GTX 1650) when available, falling back cleanly to CPU if CUDA is unavailable.
- Caches weights in memory for microsecond inference latency.

### 2.4 API & Routing Layer (`backend/app/api/`)
- Built on **FastAPI** with auto-generated OpenAPI / Swagger documentation (`/docs`).
- Endpoints:
  - `GET /api/health` — Liveness and scientific core integrity.
  - `GET /api/config` — Nominal constants and parameter defaults.
  - `GET /api/system-status` — Dynamic Python, PyTorch, CUDA, and hardware telemetry.
  - `GET /api/model-status` — PINN architecture, parameters (83,904), training splits, and validation records.
  - `POST /api/simulation` — Unified single-request simulation returning array excitation, impairments, calibration corrections, beam traces, and metrics.
  - `POST /api/calibration` — Isolated calibration comparison.
  - `POST /api/beam-pattern` — Beam synthesis endpoint.
  - `POST /api/robustness` — Live robustness sweeps (SNR, coupling mismatch, steering angle).

### 2.5 Presentation Layer (`frontend/src/`)
- Built with **React 19**, **TypeScript**, and **Vite**.
- **State Management:** Centralized **Zustand** store (`appStore.ts`) managing simulation parameters, telemetry, and view routing.
- **Visual Design:** Sophisticated antenna laboratory instrument aesthetic (clean light theme, technical typography, precise tabular data).
- **Core Visualizations:**
  - *SVG 64-Element Antenna Array:* Dynamic mapping of phase to color hue ($0^\circ–360^\circ$) and magnitude to element radius, supporting instant element inspection on hover.
  - *Plotly Beam Pattern Chart:* Dual-trace normalized power patterns ($[-90^\circ, +90^\circ]$) comparing Ideal, Impaired, Analytical, and PINN beam shapes.
  - *Element Error Analysis:* Per-element bar and line distributions exposing individual element phase and gain degradation alongside calibration residuals.
  - *Robustness Lab:* Real-time parameter sweep curves and tabulated statistics across varying SNR, coupling mismatch, and steering regions.
