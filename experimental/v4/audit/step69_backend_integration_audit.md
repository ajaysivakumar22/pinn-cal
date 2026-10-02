# PINN-Cal Step 69: V4 Production Integration Branch & Backend Architecture Audit

**Audit Date**: 2026-10-02  
**Integration Branch**: `v4-integration`  
**Base Release Commit**: `a812584df63f137e3f478a68e103d9c5ae9caefc`  
**Base Release Tag**: `v4.0.0-frozen-ref`  
**Audit Purpose**: Thorough forensic inspection of backend services, model loading, live inference pipeline, and test coverage to prepare for clean V4 production integration without violating scientific freeze boundaries.

---

## 1. Integration Branch Verification

The dedicated integration branch was created cleanly from the frozen release tag without modifying any files:
```bash
git checkout -b v4-integration v4.0.0-frozen-ref
```
- **Current Branch**: `v4-integration`
- **HEAD Commit**: `a812584df63f137e3f478a68e103d9c5ae9caefc`
- **Tag Target Commit (`v4.0.0-frozen-ref^{commit}`)**: `a812584df63f137e3f478a68e103d9c5ae9caefc`
- **Status**: The branch points precisely to the immutable frozen release commit.

---

## 2. Current Backend Architecture & File Responsibilities

The backend is built with FastAPI, modularized into clear API routes, schemas, core configurations, and domain services:

| Component Responsibility | Implementing File Path | Implementation Details |
| :--- | :--- | :--- |
| **FastAPI Startup & Lifespan** | `backend/app/main.py` | Defines `lifespan(app: FastAPI)` which initializes the singleton `ModelService.get()` once at startup. Configures CORS middleware and registers `/api` routers. |
| **API Routers** | `backend/app/api/system.py`<br>`backend/app/api/simulation.py`<br>`backend/app/api/robustness.py` | `/api/health`, `/api/system-status`, `/api/model-status`, `/api/config`<br>`POST /api/simulation`<br>`POST /api/robustness` |
| **Calibration Request Handling** | `backend/app/api/simulation.py`<br>`backend/app/services/simulation_service.py` | `simulation(req: SimulationRequest)` validates input schemas and delegates to `run_simulation()`. |
| **Model Loading** | `backend/app/services/model_service.py` | `ModelService._load()` loads model architecture and checkpoint into memory once on application boot. |
| **Model Inference** | `backend/app/services/model_service.py` | `ModelService.predict(x: np.ndarray)` executes `@torch.no_grad()` forward pass on `DEVICE`. |
| **Physics Preconditioner** | `backend/app/services/simulation_service.py` | Currently implements standard nominal impedance inversion $C_{\text{analytical}} = V_{\text{ideal}} / (V_{\text{hw,est}} + \epsilon)$ for the analytical baseline. Physics preconditioner for the neural model is absent in V3. |
| **Beam-Pattern Generation** | `backend/app/services/beam_service.py` | `compute_all_beam_patterns()` computes 1,801-point angular power patterns ($-90^\circ$ to $+90^\circ$ at $0.1^\circ$ resolution) for ideal, uncalibrated, analytical, and PINN excitations. |
| **Metrics Calculation** | `backend/app/services/simulation_service.py` | Implements `_nmse_db`, `_phase_error_deg`, `_gain_error_db`, `_relative_error`, and beam metrics (`pointing_error_deg`, `beam_correlation`). |
| **Model / Checkpoint Config** | `backend/app/core/config.py` | Sets `PINN_BEST_CHECKPOINT`, `MODEL_PARAMS`, `MODEL_BEST_EPOCH`, and physics constants (`SELF_IMPEDANCE`, `NEAREST_NEIGHBOR_COUPLING`). |
| **Scientific Module Adapter** | `backend/app/scientific/adapter.py` | Dynamically resolves repository root on `sys.path` and wraps imports to protect frozen scientific code from runtime modification. |
| **Error Handling** | `backend/app/api/simulation.py` | Translates domain exceptions (`RuntimeError`, `ValueError`) into structured HTTP 503 / 500 / 422 JSON responses. |

---

## 3. Current Model Audit (V3 Baseline)

The backend is currently wired exclusively to the legacy V3 model:

- **Model Class Loaded**: `PINNCalV3` (imported from `pinn_cal_final.py` via `adapter.py`).
- **Checkpoint Loaded**: `models/pinn_cal_final_best.pth` (configured via `backend/app/core/config.py`).
- **Model Version**: **V3 (Legacy)**. The backend does **not** currently use V4.
- **Loading Cadence**: Loaded **once** as a persistent singleton (`ModelService._load()`) during application startup via FastAPI lifespan.
- **Hardware Acceleration / Device**: Dynamically detects CUDA:
  `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- **Inference Input Format**: `(N, 130)` `float32` array:
  $[64 \text{ Re}(I_{\text{obs}}), 64 \text{ Im}(I_{\text{obs}}), \sin\theta, \cos\theta]$.
- **Inference Output Format**: `(N, 128)` `float32` array:
  $[64 \text{ Re}(C), 64 \text{ Im}(C)]$, converted via `pred[:, :64] + 1j * pred[:, 64:]`.
- **Reported Parameter Count**: **83,904** parameters (V3 MLP $130 \to 256 \to 128 \to 64 \to 128$).

---

## 4. Comparison Against Frozen V4 Specification

| Architectural Property | Current Backend (V3) | Frozen V4 Specification | Integration Gap Analysis |
| :--- | :--- | :--- | :--- |
| **Model Class** | `PINNCalV3` | `PINNCalV4` (`experimental.v4.pinn_v4_model`) | Must update `adapter.py` and `model_service.py` to import and instantiate `PINNCalV4`. |
| **Checkpoint Path** | `models/pinn_cal_final_best.pth` | `experimental/v4/models/pinn_v4_tikhonov015_seed42.pth` | Must update `config.py` to point to the frozen V4 reference checkpoint. |
| **Preconditioner** | None in neural path | `FixedPhysicsPreconditioner` ($\delta = 0.15$) | Built into `PINNCalV4`; non-learned, zero parameter impedance inversion. |
| **Residual Bounding** | None (direct regression) | `BoundedResidualDenoiser` ($\alpha = 0.50$) | Built into `PINNCalV4`; guarantees $\Delta C \in [-0.50, +0.50]$. |
| **Parameter Count** | 83,904 trainable parameters | **33,088** trainable parameters | 60.6% parameter reduction; metadata in `config.py` and tests must be updated. |
| **Forward Pass Signature** | `output = model(x)` $\to (N, 128)$ | `c_hat, delta_c, x_prior = model(x)` | `PINNCalV4` returns a 3-tuple. `c_hat` is already `(N, 64)` `complex128`. `model_service.predict()` must be adapted to handle this output without crashing. |
| **Explainability Data** | None | Exposes physics prior $C_{\text{prior}}$ and neural residual $\Delta C$ | Opportunity to surface $C_{\text{prior}}$ and $\Delta C$ to frontend for visual explainability. |

---

## 5. Scientific Integration Safety

The backend architecture preserves absolute scientific integrity through the adapter pattern (`backend/app/scientific/adapter.py`):
1. **Zero Math Duplication**: The backend does **not** reimplement the mutual impedance formulas, steering equations, or array factors. All physical computations delegate directly to `physics_core.py` and `coupling_model.py`.
2. **Read-Only Model Import**: `PINNCalV4` will be imported directly from `experimental.v4.pinn_v4_model`. No duplicate neural network classes will be created inside `backend/`.
3. **Protected Core Invariance**: The following files represent ground-truth physical simulation and must remain byte-identical:
   - `physics_core.py`
   - `coupling_model.py`
   - `calibration_dataset.py`
   - `pinn_cal_final.py`
   - `analytical_baseline.py`
   - `models/pinn_cal_final_best.pth`
   - `data/physics_test_200.npy`
   - `experimental/v4/models/pinn_v4_tikhonov015_seed42.pth`
   - `experimental/v4/results/blind_final_holdout_500.npy`

---

## 6. Live Application Flow & Trace

```text
[Frontend ControlPanel.tsx] (Target Angle, Impairments, SNR, Coupling Mismatch, Seed)
        │
        ▼ HTTP POST /api/simulation (JSON payload)
[FastAPI Router backend/app/api/simulation.py] (Validates SimulationRequest schema)
        │
        ▼ Calls run_simulation(req)
[Simulation Service backend/app/services/simulation_service.py]
        │
        ├─► physics_core.steering_vector(theta) ────────► V_ideal
        ├─► physics_core.generate_hardware_errors() ───► V_hardware
        ├─► coupling_model.generate_true_coupling() ────► Z_true
        ├─► torch.linalg.solve(Z_true, V_hw) ───────────► I_true
        └─► Add complex Gaussian noise (SNR_dB) ────────► I_observed (130-dim vector)
        │
        ▼ ModelService.predict(x_input)
[Frozen V4 Model experimental/v4/pinn_v4_model.py]
        │
        ├─► Stage 1: FixedPhysicsPreconditioner(delta=0.15) ──► C_prior
        ├─► Stage 2: BoundedResidualDenoiser(alpha=0.50) ────► Delta_C
        └─► Reconstruction: c_hat = C_prior + Delta_C ───────► Calibrated Correction
        │
        ▼
[Beam Service backend/app/services/beam_service.py]
        │
        ├─► Ideal Beam Pattern (1,801 points)
        ├─► Uncalibrated Impaired Beam Pattern (1,801 points)
        ├─► Analytical Baseline Calibrated Pattern (1,801 points)
        └─► PINN-Cal V4 Calibrated Pattern (1,801 points)
        │
        ▼ Metrics Computation (NMSE, Phase Error, Gain Error, Pointing Error, Beam Correlation)
[FastAPI HTTP Response 200 OK] (SimulationResponse JSON)
        │
        ▼
[Frontend Visualizations]
        ├─► BeforeAfterView.tsx (Side-by-side array correction & pattern)
        ├─► BeamPatternChart.tsx (4-trace polar & Cartesian beam comparison)
        ├─► MetricsPanel.tsx (Headline NMSE, phase error, pointing error)
        └─► ElementInspector.tsx (Per-element complex gain & phase correction)
```

### Missing Connection Identified:
In `simulation_service.py` (lines 226–227):
```python
pred = model_svc.predict(x_input) # Currently expects shape (1, 128)
c_pinn = pred[0, :64] + 1j * pred[0, 64:]
```
Because `PINNCalV4` outputs `c_hat` (shape `(B, 64)`, `complex128`), `model_service.predict()` currently crashes if given `PINNCalV4` unless it is adapted to return `c_hat` directly or unpack `(c_hat, delta_c_real, x_prior)`.

---

## 7. Test Coverage Audit

Inspecting `backend/tests/`:
- `test_health.py`:
  - Tests `/api/health`, `/api/system-status`, `/api/model-status`, `/api/config`.
  - **Issue**: Line 44 hardcodes `assert data["total_parameters"] == 83904` and `assert data["train_samples"] == 800`. When V4 is integrated, this test will fail unless updated to assert V4 parameter count (33,088).
- `test_simulation.py`:
  - Tests simulation pipeline, 64-element outputs, PINN availability, 4 beam traces, 1,801-point grid, finite metrics, and invalid input rejection (422).
  - All 11 simulation tests pass cleanly against the existing service.
- **Coverage Gaps**:
  - No existing test verifies `pinn_v4_tikhonov015_seed42.pth` loading in backend.
  - No test verifies `delta = 0.15` and `alpha = 0.50` properties in backend model status.
  - No test checks that `model_service.predict()` handles V4 complex output format.

---

## 8. Exact Files That WILL Need Modification in Step 70

Only 4 backend integration files will require minimal modifications:
1. `backend/app/core/config.py`:
   - Point `PINN_BEST_CHECKPOINT` to `experimental/v4/models/pinn_v4_tikhonov015_seed42.pth`.
   - Update `MODEL_PARAMS` to `33_088`.
   - Update `VALIDATED_RESULTS` with frozen V4 headline holdout benchmarks (-14.81 dB at 20 dB, -2.30 dB at 5 dB).
2. `backend/app/scientific/adapter.py`:
   - Import `PINNCalV4` and `FixedPhysicsPreconditioner` from `experimental.v4.pinn_v4_model`.
3. `backend/app/services/model_service.py`:
   - Instantiate `PINNCalV4(bounded=True, alpha=0.50, delta=0.15)`.
   - Update `predict()` to handle `PINNCalV4` forward output (`c_hat`, `delta_c`, `x_prior`).
   - Update `status_dict()` to report V4 architecture, $\delta=0.15$, $\alpha=0.50$, and 33,088 parameters.
4. `backend/tests/test_health.py`:
   - Update parameter assertion from 83,904 to 33,088.

---

## 9. Exact Files That MUST Remain Untouched

- `physics_core.py`
- `coupling_model.py`
- `calibration_dataset.py`
- `pinn_cal_final.py`
- `analytical_baseline.py`
- `models/pinn_cal_final_best.pth`
- `data/physics_test_200.npy`
- `experimental/v4/pinn_v4_model.py`
- `experimental/v4/models/pinn_v4_tikhonov015_seed42.pth`
- `experimental/v4/models/pinn_v4_tikhonov015_seed123.pth`
- `experimental/v4/models/pinn_v4_tikhonov015_seed456.pth`
- `experimental/v4/results/blind_final_holdout_500.npy`
- `experimental/v4/RELEASE_MANIFEST.md`

---

## 10. Audit Conclusion

The backend is well-structured and decoupled. Integrating the frozen V4 reference model requires zero changes to the scientific core or V4 model definitions. The transition from V3 to V4 requires adapting the singleton `ModelService` and updating configuration metadata in Step 70.
