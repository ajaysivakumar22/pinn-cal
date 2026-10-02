# PINN-Cal V4 Frozen Model Release Artifact

This directory contains metadata and formal specifications for the frozen **PINN-Cal V4 Reference Model**.

## Checkpoint Reference
- **Model Checkpoint**: `experimental/v4/models/pinn_v4_tikhonov015_seed42.pth`
- **SHA256**: `f5da358645e24536ce4f3c6144bc28f47f9a13bfc6c6744169f6ee055f2f9efe`
- **Status**: **FROZEN — FINAL PRODUCTION CANDIDATE**

## Model Architecture
- **Type**: Two-Stage Physics Preconditioner + Bounded Neural Residual Denoiser (`PINNCalV4`)
- **Stage 1 (Upstream Physics)**: Scalar complex Tikhonov-regularized analytical preconditioner ($\delta=0.15$)
  $$C_{\text{prior}} = \frac{V_{\text{ideal}} V_{\text{hw}}^*}{|V_{\text{hw}}|^2 + \delta^2}, \quad |C_{\text{prior}}| \le \frac{1}{2\delta} = 3.333$$
- **Stage 2 (Downstream Neural)**: Multi-layer perceptron with fixed hyperbolic tangent residual bounding ($\alpha=0.50$)
  $$\Delta C = \alpha \tanh(\text{MLP}(X_{\text{prior}})), \quad \Delta C \in [-0.50, +0.50]$$
- **Total Parameters**: 33,088 trainable parameters (MLP: $128 \to 128 \to 64 \to 128$)
- **Input Dimension**: 130 features ($64 \text{ Re}(I_{\text{obs}}), 64 \text{ Im}(I_{\text{obs}}), \sin\theta, \cos\theta$)
- **Output Dimension**: 64 complex element-wise corrections $\hat{C} = C_{\text{prior}} + \Delta C$

## Verification
- Fully verified via Step 64 blind holdout ($N=5,000$ independent realizations).
- Blind holdout data file: `experimental/v4/results/blind_final_holdout_500.npy` (**FINAL — DO NOT USE FOR MODEL SELECTION**).
- Zero overlap verified across 9,800 historical samples.
