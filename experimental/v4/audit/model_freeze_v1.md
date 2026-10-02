# PINN-Cal V4 Model Freeze Record (v1.0)

**Freeze Date**: 2026-10-02  
**Status**: **PERMANENTLY FROZEN**

```text
PINN-Cal V4 Reference Model
delta = 0.15
alpha = 0.50
seed = 42
```

---

## 1. Frozen Model Checkpoint

| Attribute | Specification |
| :--- | :--- |
| **Model Name** | PINN-Cal V4 Reference Model |
| **Checkpoint Path** | `experimental/v4/models/pinn_v4_tikhonov015_seed42.pth` |
| **SHA256 Checksum** | `f5da358645e24536ce4f3c6144bc28f47f9a13bfc6c6744169f6ee055f2f9efe` |
| **Architecture** | Two-Stage `PINNCalV4` (Tikhonov Preconditioner + Bounded Residual Denoiser) |
| **Tikhonov Regularization ($\delta$)** | `0.15` |
| **Residual Bound ($\alpha$)** | `0.50` |
| **Reference Seed** | `42` |
| **Trainable Parameter Count** | `33,088` |
| **Total Parameter Count** | `33,088` |

---

## 2. Frozen Scientific Core Hashes

The following scientific core files represent the physical ground truth and simulation pipeline. All hashes have been cryptographically verified:

| File Path | SHA256 Checksum | Verification Status |
| :--- | :--- | :--- |
| `physics_core.py` | `91e47b01230bf92216cec1595b58283b92d3f0fcee851a3b2614bd2b1f464083` | **UNMODIFIED / PROTECTED** |
| `coupling_model.py` | `98861acbd5fe98fecfa19d68e08a907b041cd63fa595a042d85c32c9d9c79d12` | **UNMODIFIED / PROTECTED** |
| `calibration_dataset.py` | `7656af098369f3a9a862f644c44ddf718ed67c75dc77c97901af9f019fc8351a` | **UNMODIFIED / PROTECTED** |
| `pinn_cal_final.py` | `f3c5a8b09f47f9279b6a99b1265342d372cf027ab911e0bbef83c6f4e4d3edb2` | **UNMODIFIED / PROTECTED** |
| `analytical_baseline.py` | `2d9e91ea88a8115fa11e7ffb620f58e0837343a0e9e00fbd6d8036e58aa8c9e6` | **UNMODIFIED / PROTECTED** |

---

## 3. Benchmark & Reference Hashes

| Artifact Description | File Path | SHA256 Checksum |
| :--- | :--- | :--- |
| **Frozen PINN-v3 Checkpoint** | `models/pinn_cal_final_best.pth` | `5edc7d62cd7db7a655c580a42b4d70b649a0868cac62e503abd7d744d7936eb8` |
| **Frozen Benchmark Test Split** | `data/physics_test_200.npy` | `00670ac08b808606903d763234fac0d7c6e0d1958771d9af66f723d41de7e432` |
| **Blind Final Holdout (N=5000)** | `experimental/v4/results/blind_final_holdout_500.npy` | `edb5b02ba8825da5633db844b0186b030cc2d39b66a4751bf9c52108d3781c48` |
| **V4 Reference Checkpoint** | `experimental/v4/models/pinn_v4_tikhonov015_seed42.pth` | `f5da358645e24536ce4f3c6144bc28f47f9a13bfc6c6744169f6ee055f2f9efe` |

---

## 4. Test Suite Execution Status

The complete test suite was executed against the release checkpoint and backend services:

```text
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0
rootdir: /mnt/d/pinn-cal
collected 23 items

experimental/v4/test_v4_pipeline.py .......                              [ 30%]
backend/tests/test_health.py .....                                       [ 52%]
backend/tests/test_simulation.py ...........                             [100%]

======================= 23 passed, 2 warnings in 39.93s ========================
```

- Total Tests: **23**
- Passed: **23 (100%)**
- Failed: **0**
- Test Coverage: Unit tests, preconditioner invariance, bounded residual guarantees, pipeline reconstruction, checkpoint validation, FastAPI health, and simulation endpoints.

---

## 5. Model Freeze Declaration

The model `pinn_v4_tikhonov015_seed42.pth` is formally frozen. No further training, tuning, or hyperparameter adjustment will be performed. This checkpoint constitutes the baseline for future production backend deployment.
