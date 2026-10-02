# PINN-Cal V4 Release Manifest

**Model Release Date**: 2026-10-02  
**Status**: **FROZEN REFERENCE MODEL**  
**Release Tag**: `v4.0.0-frozen-ref`

---

## 1. Model Identification & Checkpoint Specification

- **Model Name**: PINN-Cal V4 Reference Model (Tikhonov Preconditioned + Bounded Residual Denoiser)
- **Model Version**: `v4.0.0-frozen`
- **Class Implementation**: `experimental.v4.pinn_v4_model.PINNCalV4`
- **Reference Checkpoint Path**: `experimental/v4/models/pinn_v4_tikhonov015_seed42.pth`
- **Checkpoint SHA256**: `f5da358645e24536ce4f3c6144bc28f47f9a13bfc6c6744169f6ee055f2f9efe`
- **Reference Training Seed**: `42`
- **Parameter Count**:
  - Trainable parameters: **33,088**
  - Non-trainable parameters: **0**
  - Total parameters: **33,088**
- **Input Dimension**: $130$ float32 features ($64 \text{ Re}(I_{\text{obs}}), 64 \text{ Im}(I_{\text{obs}}), \sin\theta, \cos\theta$)
- **Output Dimension**: $64$ complex128 element-wise correction coefficients ($\hat{C} \in \mathbb{C}^{64}$)

---

## 2. Architecture & Physics Preconditioning

The V4 reference model employs a two-stage hybrid physics-neural architecture:

### Stage 1: Upstream Physics Preconditioner (`FixedPhysicsPreconditioner`)
- **Regularization Strategy**: Scalar complex Tikhonov regularization applied directly to the array's mutual impedance inverse:
  $$C_{\text{prior}} = \frac{V_{\text{ideal}} V_{\text{hw}}^*}{|V_{\text{hw}}|^2 + \delta^2}$$
  where $V_{\text{hw}} = (Z_{\text{nom}} I_{\text{obs}}) / Z_{\text{self,nom}}$.
- **Regularization Parameter**: $\delta = 0.15$
- **Mathematical Bound**:
  $$\max |C_{\text{prior}}| \le \frac{1}{2\delta} = \frac{1}{0.30} \approx 3.333$$
- **Phase Invariance**: The denominator $|V_{\text{hw}}|^2 + \delta^2$ is strictly positive real, introducing **$0.0000^\circ$ phase distortion** to the physics estimate.

### Stage 2: Downstream Bounded Residual Denoiser (`BoundedResidualDenoiser`)
- **Architecture**: Multi-layer perceptron ($128 \to 128 \to 64 \to 128$) with ReLU activations.
- **Bounding Mechanism**: Strict hyperbolic tangent scaling:
  $$\Delta C = \alpha \tanh(\text{MLP}(X_{\text{prior}})), \quad \alpha = 0.50$$
- **Residual Range**: $\Delta C \in [-0.50, +0.50]$ along real and imaginary components.
- **Final Reconstruction**:
  $$\hat{C} = C_{\text{prior}} + \Delta C$$

---

## 3. Dataset Provenance & Training Protocol

### 3.1 Training Dataset
- **File**: `experimental/v4/datasets/v4_train_2000.npy` (Size: 2.10 MB, SHA256: `7b5040fd61e7627341270258169fe5a21be14a79b292e0df294fae9b114dcae4`)
- **Provenance**: Generated via the frozen physical simulation core (`calibration_dataset.generate_sample` with `DEFAULT_CONFIG`).
- **Sample Count**: $2,000$ training realizations.
- **SNR Diversity**: Uniformly sampled over $[5.0, 30.0]\text{ dB}$.
- **Hardware Impairments**: Phase error $\sigma_\phi = 20.0^\circ$, gain error $\sigma_g = 2.0\text{ dB}$, mutual coupling mismatch $20\%$, steering angle range $[-60^\circ, +60^\circ]$.
- **Base Seed**: `202604`

### 3.2 Validation Protocol
- **File**: `experimental/v4/datasets/v4_val_500.npy` (Size: 525 KB, SHA256: `cf6a48b65493d583c4838b8aa8cf2436f6d2da53ea2e2ef9ceea4a8aa527b140`)
- **Sample Count**: $500$ independent validation realizations.
- **Optimizer**: AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$).
- **Batch Size**: 32 | **Epochs**: 100.
- **Best Validation Epoch**: Epoch 29 (Validation Loss: $0.134375$, Train Loss: $0.133285$, Generalization Gap: $+0.001090$).

---

## 4. Blind Final Holdout Specification & Provenance

> [!CAUTION]
> **BLIND HOLDOUT STATUS: FINAL — DO NOT USE FOR MODEL SELECTION**  
> The blind holdout dataset was generated strictly for final release validation after all model hyperparameters were frozen. It must never be incorporated into future training, validation, or checkpoint selection loops.

- **File Path**: `experimental/v4/results/blind_final_holdout_500.npy` (Size: 4.96 MB)
- **SHA256 Checksum**: `edb5b02ba8825da5633db844b0186b030cc2d39b66a4751bf9c52108d3781c48`
- **Creation Timestamp**: 2026-10-02 06:31:37 UTC
- **Base RNG Seed**: `918273` (verified 100% fresh, never used in prior development)
- **Sample Count**: $500$ independent physical realizations per condition across 10 evaluation conditions ($5,000$ total samples).
- **Zero-Overlap Audit**: Exact zero sample overlap verified against **9,800 historical samples** across 10 datasets with a minimum $L_\infty$ distance of $0.765478$.

---

## 5. Baselines for Comparison

### 5.1 Frozen Original PINN-v3
- **Checkpoint**: `models/pinn_cal_final_best.pth`
- **SHA256**: `5edc7d62cd7db7a655c580a42b4d70b649a0868cac62e503abd7d744d7936eb8`
- **Architecture**: Single-stage direct MLP with physics loss regularizer.

### 5.2 Analytical Baseline
- **Implementation**: `analytical_baseline.analytical_correction`
- **Methodology**: Inversion using nominal mutual impedance matrix without Tikhonov regularizer ($C_{\text{analytical}} = V_{\text{ideal}} / (V_{\text{hw}} + \epsilon)$).

---

## 6. Blind Holdout Headline Metrics ($N=500$ Samples per Condition)

All metrics are recorded as $\text{Mean} \pm \text{Std}$ over the $N=500$ independent blind holdout samples:

| Condition | Analytical Baseline | Frozen PINN-v3 | V4 Reference (Seed 42) | V4 Advantage vs. Analytical | V4 Advantage vs. PINN-v3 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Nominal 20 dB** | $-14.23\text{ dB}$ | $-7.81\text{ dB}$ | **$-14.81\text{ dB}$** | **$+0.58\text{ dB}$** | **$+7.00\text{ dB}$** |
| Phase Error (°) | $3.96 \pm 0.63^\circ$ | $16.16 \pm 1.53^\circ$ | **$4.47 \pm 0.72^\circ$** | $+0.51^\circ$ | **$-11.69^\circ$** |
| Gain Error (dB) | $1.07 \pm 0.17\text{ dB}$ | $1.68 \pm 0.21\text{ dB}$ | **$1.00 \pm 0.15\text{ dB}$** | **$-0.07\text{ dB}$** | **$-0.68\text{ dB}$** |
| Pointing Error (°) | $0.0259 \pm 0.0170^\circ$ | $0.0482 \pm 0.0379^\circ$ | **$0.0253 \pm 0.0163^\circ$** | **$-0.0006^\circ$** | **$-0.0229^\circ$** |
| Beam Correlation | $0.99979$ | $0.99716$ | **$0.99979$** | Equal | **$+0.00263$** |
| **15 dB SNR** | $-10.01\text{ dB}$ | $-7.84\text{ dB}$ | **$-11.47\text{ dB}$** | **$+1.46\text{ dB}$** | **$+3.63\text{ dB}$** |
| **10 dB SNR** | $-2.39\text{ dB}$ | $-7.78\text{ dB}$ | **$-6.88\text{ dB}$** | **$+4.49\text{ dB}$** | $-0.90\text{ dB}$ |
| **7.5 dB SNR** | $+1.47\text{ dB}$ | $-7.76\text{ dB}$ | **$-4.27\text{ dB}$** | **$+5.74\text{ dB}$** | $-3.49\text{ dB}$ |
| **5 dB SNR** | $+3.16\text{ dB}$ | $-7.80\text{ dB}$ | **$-2.30\text{ dB}$** | **$+5.46\text{ dB}$** | $-5.50\text{ dB}$ |
| Pointing Error at 5 dB | $0.4275 \pm 5.1559^\circ$ | $0.0510 \pm 0.0396^\circ$ | **$0.0765 \pm 0.0706^\circ$** | **$-0.3510^\circ$ (5.6x tighter)** | $+0.0255^\circ$ |
| Beam Corr at 5 dB | $0.90895$ | $0.99692$ | **$0.98133$** | **$+0.07238$** | $-0.01559$ |
| **Phase Stress (35°)** | $-14.33\text{ dB}$ | $-4.46\text{ dB}$ | **$-13.96\text{ dB}$** | $-0.37\text{ dB}$ | **$+9.50\text{ dB}$** |
| **Gain Stress (4.0 dB)** | $-9.56\text{ dB}$ | $-5.32\text{ dB}$ | **$-11.77\text{ dB}$** | **$+2.21\text{ dB}$** | **$+6.45\text{ dB}$** |
| **Coupling Stress (40%)**| $-9.92\text{ dB}$ | $-7.77\text{ dB}$ | **$-11.02\text{ dB}$** | **$+1.10\text{ dB}$** | **$+3.25\text{ dB}$** |
| **Steering Stress (±65°)**| $-14.25\text{ dB}$ | $-7.84\text{ dB}$ | **$-14.82\text{ dB}$** | **$+0.57\text{ dB}$** | **$+6.98\text{ dB}$** |
| **Combined Severe** | $+8.41\text{ dB}$ | $-3.45\text{ dB}$ | **$-2.35\text{ dB}$** | **$+10.76\text{ dB}$** | **$+1.10\text{ dB}$** |
| Pointing (Combined) | $0.3919^\circ$ | $0.1255^\circ$ | **$0.0928^\circ$** | **$-0.2991^\circ$ (4.2x tighter)** | **$-0.0327^\circ$** |
| Beam Corr (Combined) | $0.92748$ | $0.98041$ | **$0.98992$** | **$+0.06244$** | **$+0.00951$** |

---

## 7. Software & Execution Environment

- **Operating System**: Linux (Ubuntu 24.04 on WSL2 / Windows 11 host)
- **Python**: 3.11.15
- **PyTorch**: 2.5.1+cu121
- **CUDA Device**: NVIDIA GeForce GTX 1650 (Driver Version 566.07, CUDA 12.7)
- **NumPy**: 2.2.3
- **Test Suite Status**: **23 / 23 Tests Passed (100%)** via `pytest experimental/v4/test_v4_pipeline.py backend/tests`.
