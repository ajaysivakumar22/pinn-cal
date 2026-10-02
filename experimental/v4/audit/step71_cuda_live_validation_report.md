# Step 71 Audit Report: CUDA Inference + Live End-to-End Validation

**Date**: 2026-10-02  
**Branch**: `v4-integration`  
**Base Release Tag**: `v4.0.0-frozen-ref`  
**Base Commit**: `a812584df63f137e3f478a68e103d9c5ae9caefc`  
**Status**: VERIFICATION COMPLETE — ALL PARTS PASSED  

---

## 1. CUDA Environment

- **Host OS**: Microsoft Windows 11
- **WSL Linux Distribution**: Ubuntu 26.04 LTS (Kernel: 6.6.114.1-microsoft-standard-WSL2 x86_64)
- **Host Python Environment**: Python 3.12.10 (PyTorch 2.10.0+cpu — CPU only)
- **WSL Python Environment**: Python 3.11.15 (`/mnt/d/pinn-cal/venv/bin/python`)
- **PyTorch Version in WSL**: `2.5.1+cu121`
- **NVIDIA Driver**: 566.07 (Host WDDM) / 565.65 (WSL2 Driver)
- **CUDA Runtime Version**: CUDA 12.7

---

## 2. GPU Name & Hardware Specification

- **GPU**: NVIDIA GeForce GTX 1650
- **Device Index**: `cuda:0`
- **Device Count**: 1
- **VRAM Total**: 4,096 MiB (4.0 GB)
- **Accessibility**: Directly accessible from both Windows host via `nvidia-smi` and WSL environment via `torch.cuda` and `nvidia-smi`.

---

## 3. CUDA Availability

- `torch.cuda.is_available()`: **`True`** (inside WSL PyTorch 2.5.1+cu121 environment)
- `torch.cuda.get_device_name(0)`: **`NVIDIA GeForce GTX 1650`**
- `torch.cuda.device_count()`: **`1`**

---

## 4. V4 Checkpoint Verification

- **Checkpoint Path**: `experimental/v4/models/pinn_v4_tikhonov015_seed42.pth`
- **SHA-256**:
  `f5da358645e24536ce4f3c6144bc28f47f9a13bfc6c6744169f6ee055f2f9efe`
- **Verification Status**: Exact bit-for-bit match with the frozen release manifest.
- **Model Instantiation**:
  ```python
  PINNCalV4(
      bounded=True,
      alpha=0.50,
      delta=0.15,
  )
  ```
- **Loaded Keys**: Full strict state_dict match.
- **Total Parameters**: 33,088
- **Trainable Parameters**: 33,088
- **Evaluation Mode**: `model.training == False` (Active `eval()` mode).
- **Gradients**: All inference executed strictly under `torch.no_grad()`.

---

## 5. CPU vs CUDA Numerical Consistency

A controlled deterministic tensor of shape `(1, 130)` was passed through the identical checkpoint on both CPU and CUDA (`cuda:0`).

### Maximum and Mean Absolute Discrepancies:
| Component | Max Absolute Difference | Mean Absolute Difference | Tolerance Threshold | Status |
| :--- | :--- | :--- | :--- | :--- |
| **$c_{\text{hat}}$ (Real part)** | $5.96 \times 10^{-8}$ | $1.78 \times 10^{-8}$ | $1.00 \times 10^{-4}$ | **PASS** |
| **$c_{\text{hat}}$ (Imag part)** | $5.96 \times 10^{-8}$ | $1.76 \times 10^{-8}$ | $1.00 \times 10^{-4}$ | **PASS** |
| **$c_{\text{hat}}$ (Complex norm)** | $7.92 \times 10^{-8}$ | $2.86 \times 10^{-8}$ | $1.00 \times 10^{-4}$ | **PASS** |
| **Residual $\Delta c$** | $5.96 \times 10^{-8}$ | $1.77 \times 10^{-8}$ | $1.00 \times 10^{-4}$ | **PASS** |

**Conclusion**: CPU and CUDA evaluations agree within single-precision floating point limits ($\sim 10^{-8}$), far exceeding the required engineering tolerance ($10^{-4}$).

---

## 6. Real API Simulation Result

Controlled baseline request sent to `POST /api/simulation`:
- **Array Configuration**: 64 elements, 28 GHz, 0.5 $\lambda$ spacing
- **Steering Angle**: $20.0^\circ$
- **Distortions**: Phase sigma $20.0^\circ$, Gain sigma $2.0\text{ dB}$, SNR $20\text{ dB}$, Mutual coupling mismatch fraction $20\%$
- **Seed**: 42

### Outputs Recorded:
- **Observed Current Count**: 64 complex values (all finite, zero NaN/Inf)
- **Correction Count**: 64 complex correction coefficients (all finite, zero NaN/Inf)
- **Beam Trace Count**: 4 complete traces (`ideal`, `uncalibrated`, `analytical`, `pinn`)
- **Angle Grid Resolution**: 1,801 angular points ($-90.0^\circ$ to $+90.0^\circ$, step $0.1^\circ$)
- **Peak Angles**:
  - Ideal: $20.00^\circ$
  - Analytical: $20.00^\circ$
  - PINN V4: $20.00^\circ$
- **Analytical Metrics**:
  - NMSE: $-15.76\text{ dB}$
  - Mean Phase Error: $3.17^\circ$
  - Mean Gain Error: $1.03\text{ dB}$
  - Relative Error: $14.01\%$
- **PINN V4 Metrics**:
  - NMSE: $-14.57\text{ dB}$
  - Mean Phase Error: $4.49^\circ$
  - Mean Gain Error: $1.10\text{ dB}$
  - Relative Error: $16.14\%$
  - Current Residual: $9.08\%$

---

## 7. Dynamic Parameter Verification

Four varied physical conditions were evaluated to confirm that the backend dynamically calculates responses and does not serve cached or static values:

| Variation Tested | Observed Current Max Diff | V4 Correction Max Diff | Beam Peak Angle | PINN NMSE | NMSE Change from Base | Dynamic Response Confirmed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Steering $20^\circ \to -30^\circ$** | 2.2890 | 0.9221 | **$-30.00^\circ$** (shifted) | $-13.97\text{ dB}$ | $+0.60\text{ dB}$ | **YES** |
| **SNR $20 \to 10\text{ dB}$** | 0.3820 | 1.0072 | $20.00^\circ$ | **$-8.02\text{ dB}$** | **$-6.55\text{ dB}$** | **YES** |
| **Phase $\sigma$: $20^\circ \to 35^\circ$** | 0.5954 | 0.8685 | $20.00^\circ$ | **$-12.47\text{ dB}$** | **$-2.09\text{ dB}$** | **YES** |
| **Gain $\sigma$: $2 \to 4\text{ dB}$** | 1.0678 | 0.8791 | $20.00^\circ$ | **$-10.75\text{ dB}$** | **$-3.82\text{ dB}$** | **YES** |

All outputs varied deterministically in accordance with physical principles: lower SNR degraded calibration performance, increased noise sigma increased residual error, and beam steering shifted peak power to exactly $-30.00^\circ$.

---

## 8. Frontend / Backend Connection

Live end-to-end execution verified in browser subagent:
- **Backend Service**: `http://127.0.0.1:8000` (FastAPI / Uvicorn)
- **Frontend Service**: `http://localhost:5173` (Vite / React 19)
- **Health Status Display**:
  - `BACKEND ONLINE` (Green status indicator)
  - `33,088 weights`
  - Checkpoint displayed: `D:\pinn-cal\experimental\v4\models\pinn_v4_tikhonov015_seed42.pth`
- **Interactive Action**: Triggered `RUN CALIBRATION BENCH`.
- **Render Results**:
  - Status banner: `★ CALIBRATION COMPLETE` (`Pipeline Complete · Array Calibrated`)
  - 64-element per-channel gain/phase array and bar charts rendered cleanly
  - 4-trace comparative beam pattern plotted with mainlobe aligned at $+20.0^\circ$
  - Metric cards rendered: Pointing residual $0.00^\circ$, Correlation $0.9999$, Analytical NMSE $-15.758\text{ dB}$, PINN NMSE $-14.567\text{ dB}$
- **Browser Console Errors**: 0 console errors, 0 network failures.

---

## 9. Backend Tests

Command (WSL with CUDA support):
```bash
pytest -v experimental/v4/test_v4_pipeline.py backend/tests
```

**Results**:
- **Total Tests Collected**: 35
- **Passed**: 35
- **Skipped**: 0 (`test_v4_model_runs_on_cuda_when_available` PASSED on CUDA)
- **Failed**: 0
- **Duration**: 40.61s

---

## 10. Frontend Tests

Command:
```bash
cd frontend && npm run test
```

**Results**:
- **Test Files**: 1 passed (1)
- **Tests**: 11 passed (11)
- **Duration**: 5.88s

---

## 11. Production Build Status

Command:
```bash
cd frontend && npm run build
```

**Results**:
- TypeScript type-check (`tsc -b`): Clean, zero errors.
- Vite build: Succeeded (`dist/index.html`, `dist/assets/index-DeXshm1C.css`, `dist/assets/index-QPg8gvnY.js`, `dist/assets/dist-BkSUXM8Q.js`).
- Build time: 4.81s.

---

## 12. Git Diff Scope

### `git status`
```
On branch v4-integration
Changes not staged for commit:
	modified:   backend/app/core/config.py
	modified:   backend/app/scientific/adapter.py
	modified:   backend/app/services/model_service.py
	modified:   backend/tests/test_health.py

Untracked files:
	experimental/v4/audit/step69_backend_integration_audit.md
	experimental/v4/audit/step70_backend_v4_integration_report.md
	experimental/v4/audit/step71_cuda_live_validation_report.md
```

### `git diff --stat`
```
 backend/app/core/config.py            |  86 +++++++++-----
 backend/app/scientific/adapter.py     |   9 ++
 backend/app/services/model_service.py |  78 +++++++++++--
 backend/tests/test_health.py          | 204 +++++++++++++++++++++++++++++++++-
 4 files changed, 336 insertions(+), 41 deletions(-)
```

### `git diff --name-only`
```
backend/app/core/config.py
backend/app/scientific/adapter.py
backend/app/services/model_service.py
backend/tests/test_health.py
```

Production code changes remain strictly confined to the 4 permitted files.

---

## 13. Issues Discovered

- None. Both CPU and CUDA execution pipelines operate nominally. CUDA evaluation yields virtually identical results to CPU ($\Delta \le 7.92 \times 10^{-8}$) while confirming full hardware acceleration capabilities on the NVIDIA GeForce GTX 1650.
