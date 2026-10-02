# PINN-Cal
## Physics-Informed Neural Calibration for 5G Massive MIMO Antenna Arrays

**Final Year Project — Dr. Mahalingam College of Engineering and Technology**  
Department of Electronics and Communication Engineering

### Team

- Mowsikan H
- Ajay S
- Shanjesh P S

---

# 1. Project Overview

PINN-Cal investigates physics-informed neural calibration of a 64-element
massive-MIMO antenna array.

The current validated project models:

- a 64-element uniform linear array (ULA);
- 28 GHz operating frequency;
- 0.5 λ element spacing;
- independent element gain and phase impairments;
- a reciprocal mutual-impedance network;
- hidden coupling uncertainty between the physical array and the nominal model;
- complex measurement noise;
- a neural calibration model that predicts per-element complex correction
  coefficients.

The project is intentionally built as a physically grounded simulation and
calibration framework. The mutual-impedance model is a compact reciprocal
network model and is **not a full-wave electromagnetic solver**.

---

# 2. Frozen Physical Problem

| Parameter | Value |
|---|---:|
| Array elements | 64 |
| Frequency | 28 GHz |
| Element spacing | 0.5 λ |
| Hardware phase-error σ | 20° |
| Hardware gain-error σ | 2 dB |
| Measurement SNR | 20 dB |
| Hidden coupling mismatch | ±20% |
| Input dimension | 130 |
| Output dimension | 128 |
| Steering-angle range | −60° to +60° |

The model input contains:

- 64 real observed-current values;
- 64 imaginary observed-current values;
- sin(θ);
- cos(θ).

Total input dimension:

```text
64 + 64 + 2 = 130
```

---

# 3. Live Engineering Application

A full-stack, production-grade interactive application wraps the frozen scientific core (`scientific-core-v1.0`).

### Quick Start

#### Prerequisites
- **Python:** 3.11 with PyTorch 2.5.1+cu121 (CUDA GPU supported)
- **Node.js:** v24+ with npm 11+

#### 1. Backend Service
```bash
# Start backend on http://localhost:8000
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

#### 2. Frontend Interface
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

# 4. Documentation

- [System Architecture](file:///D:/pinn-cal/docs/ARCHITECTURE.md)
- [REST API Specification](file:///D:/pinn-cal/docs/API.md)
- [Scientific Core Integration](file:///D:/pinn-cal/docs/SCIENTIFIC-INTEGRATION.md)
- [Backend Service README](file:///D:/pinn-cal/backend/README.md)
- [Frontend Interface README](file:///D:/pinn-cal/frontend/README.md)
- [Physical Problem Specification](file:///D:/pinn-cal/PHYSICAL_PROBLEM_SPEC.md)

---

# 5. Scientific Validation Results (Held-Out Test Set)

| Method | NMSE (dB) | Phase Error (°) | Gain Error (dB) | Relative Error (%) |
|---|---|---|---|---|
| **Analytical Baseline** | **−14.4545** | **3.9201** | **1.0404** | **15.1264** |
| **Final PINN (`pinn_cal_final_best.pth`)** | **−7.7489** | **16.1990** | **1.7084** | **36.7887** |

> **Neutral Reporting:** The frozen baseline analytical method currently achieves lower calibration error than the feedforward PINN on the controlled test split. The application faithfully displays both methods side-by-side.

---

# 6. PINN-Cal V4 Reference Model (Frozen Release)

The advanced **PINN-Cal V4** reference model incorporates a two-stage hybrid architecture combining a scalar complex Tikhonov-regularized physics preconditioner with a downstream bounded neural residual denoiser:

- **Physics Preconditioner Regularization:** Tikhonov parameter $\delta = 0.15$
- **Neural Residual Denoiser Bounding:** Strict hyperbolic tangent bound $\alpha = 0.50$
- **Reference Checkpoint:** [`experimental/v4/models/pinn_v4_tikhonov015_seed42.pth`](file:///D:/pinn-cal/experimental/v4/models/pinn_v4_tikhonov015_seed42.pth)
- **Checkpoint SHA256:** `f5da358645e24536ce4f3c6144bc28f47f9a13bfc6c6744169f6ee055f2f9efe`
- **Model Parameters:** **33,088** trainable parameters ($128 \to 128 \to 64 \to 128$ MLP)
- **Blind Holdout Evaluation:** Validated on **5,000 independent physical samples** ($10$ conditions $\times$ $500$ samples per condition) generated under base seed **`918273`**.
- **Holdout Status:** The blind holdout is **FINAL evaluation evidence only** and must never be used for model selection, architecture tuning, or retraining.
- **Full Release Specification:** Refer to [`experimental/v4/RELEASE_MANIFEST.md`](file:///D:/pinn-cal/experimental/v4/RELEASE_MANIFEST.md) for full benchmark tables, seed consistency analysis, and architecture details.