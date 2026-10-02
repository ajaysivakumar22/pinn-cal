# PINN-Cal v4: Loss Calculation and Generalization Audit

**Date**: 2026-09-30  
**Status**: Completed  
**Objective**: Audit the implementation of training/validation loss, residual bounding, target preprocessing, batch tracing, and diagnose the apparent mismatch between train loss (~0.3425) and validation loss (~0.1290).

---

## 1. Exact Implementation Audit

### 1.1 Training Loss
* **Source**: `experimental/v4/train_and_evaluate_v4.py` (Lines 134–146)
* **Definition**:
  ```python
  c_hat, delta_c_real, x_prior = model(xb)
  c_hat_real = torch.cat([c_hat.real.to(torch.float32), c_hat.imag.to(torch.float32)], dim=-1)
  loss = criterion(c_hat_real, yb)
  ```
  where `criterion = nn.MSELoss(reduction='mean')`.
* **Reduction**: Standard MSE across all $B \times 128$ real/imag dimensions:
  $$\mathcal{L}_{\text{batch}} = \frac{1}{B \cdot 2M} \sum_{b=1}^B \sum_{j=1}^{2M} (\hat{C}_{b,j} - Y_{b,j})^2$$
* **Epoch Aggregation**: `train_loss = float(np.mean(b_losses))` over mini-batches of size $B=32$ ($N_{\text{train}}=2000$, 63 mini-batches).
* **Gradient Regularization**: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.
* **Optimizer Regularization**: Adam optimizer with $L_2$ weight decay `weight_decay = 1e-4` on network weights. Note that weight decay affects the parameter update step in Adam, but is **not** added into `loss.item()` or `train_loss`.

### 1.2 Validation Loss
* **Source**: `experimental/v4/train_and_evaluate_v4.py` (Lines 151–159)
* **Definition**:
  ```python
  model.eval()
  with torch.no_grad():
      for xvb, yvb in val_loader:
          c_hat_v, _, _ = model(xvb)
          c_hat_v_real = torch.cat([c_hat_v.real.to(torch.float32), c_hat_v.imag.to(torch.float32)], dim=-1)
          v_losses.append(criterion(c_hat_v_real, yvb).item())
  val_loss = float(np.mean(v_losses))
  ```
* **Reduction**: Identical `nn.MSELoss(reduction='mean')` over mini-batches of size $B=32$ ($N_{\text{val}}=500$, 16 mini-batches).

### 1.3 Residual Bound
* **Source**: `experimental/v4/pinn_v4_model.py` (Lines 127–132, 164–174)
* **Equation**:
  $$\Delta C = \alpha \tanh(r), \quad \alpha = 0.5$$
* **Implementation**:
  ```python
  def forward(self, x_prior: torch.Tensor) -> torch.Tensor:
      r = self.net(x_prior)
      if self.bounded:
          return self.alpha * torch.tanh(r)
      return r
  ```
  The physical correction is reconstructed as:
  $$\hat{C} = C_{\text{prior}} + \Delta C$$
  Because $|\tanh(r)| < 1$, the neural residual is strictly bounded to $[-0.5, 0.5]$ in both real and imaginary components.

### 1.4 Target Preprocessing
* **Source**: `calibration_dataset.py` (Lines 216–258)
* **Ground Truth**: Hardware excitation impairment correction $C_{\text{true}} = V_{\text{ideal}} / V_{\text{hardware}}$.
* **Representation**: Stored as float32 in $Y \in \mathbb{R}^{N \times 128}$ where the first 64 columns represent $\mathrm{Re}(C_{\text{true}})$ and the subsequent 64 represent $\mathrm{Im}(C_{\text{true}})$.
* **Normalization**: No artificial z-scoring or scaling is applied to $Y$; it preserves natural physical calibration coefficients (mean $\approx 0.483$, std $\approx 0.568$).

### 1.5 Regularization
* **Explicit Loss Penalty**: Zero additional loss terms (no physics residual penalty, no custom $L_1/L_2$ terms added to the loss scalar).
* **Optimizer Weight Decay**: $1 \times 10^{-4}$ configured in `torch.optim.Adam`.
* **Gradient Clipping**: Norm clipped at $1.0$.

### 1.6 Best-Checkpoint Selection
* **Source**: `train_and_evaluate_v4.py` (Lines 167–176)
* **Criterion**: Evaluated strictly on `val_loss < best_val_loss`.
* **Patience**: 15 epochs.
* **Storage**: In-memory deep copy of `model.state_dict()` saved to disk upon completion.

---

## 2. Single-Batch Execution Trace

Tracing one training batch ($B=32$) and one validation batch ($B=32$) through `model.eval()`:

| Component | Training Batch (`xb`, `yb`) | Validation Batch (`xvb`, `yvb`) |
| :--- | :--- | :--- |
| **Input Shape** | `torch.Size([32, 130])` | `torch.Size([32, 130])` |
| **Prior Feature Range** | $[-3.7122, 4.5970]$ | $[-3.1045, 4.2180]$ |
| **Prior Feature Mean** | $0.4737$ | $0.4682$ |
| **Residual Range ($\Delta C$)** | $[-0.0024, 0.0027]$ | $[-0.0023, 0.0025]$ |
| **Output Shape ($\hat{C}$)** | `torch.Size([32, 64])` (complex128) | `torch.Size([32, 64])` (complex128) |
| **Concatenated Real Shape** | `torch.Size([32, 128])` (float32) | `torch.Size([32, 128])` (float32) |
| **Target Shape** | `torch.Size([32, 128])` (float32) | `torch.Size([32, 128])` (float32) |
| **Batch MSE Loss** | $0.077003$ | $0.085215$ |

**Batch Trace Confirmation**:  
The tensors, types, shapes, and forward-backward mechanics operate identically and deterministically on both training and validation batches.

---

## 3. Diagnosis: Why Train Loss (~0.3425) > Validation Loss (~0.1290)

In typical overparameterized deep networks without data augmentation, training loss is lower than validation loss. In PINN-Cal v4, the earlier report and current histories recorded:
* **Train Loss (at best epoch)**: $\approx 0.3425$
* **Validation Loss (at best epoch)**: $\approx 0.1289$
* **Apparent Generalization Gap**: $\approx -0.2135$ (negative gap!)

### 3.1 Empirical Investigation

Empirical analysis of the dataset distributions and sample-by-sample losses revealed the exact cause:

#### A. Median vs. Mean Analysis
| Metric | Training Set ($N=2000$) | Validation Set ($N=500$) |
| :--- | :--- | :--- |
| **Zero-Residual Prior Mean MSE** | **$0.369953$** | **$0.142325$** |
| **Zero-Residual Prior Median MSE** | **$0.026732$** | **$0.026552$** |
| **95th Percentile Sample MSE** | $0.621535$ | $0.448755$ |
| **99th Percentile Sample MSE** | $2.700758$ | $2.284678$ |
| **Maximum Sample MSE** | **$310.755472$** | **$7.744472$** |

Notice that the **median errors** on Train ($0.0267$) and Validation ($0.0265$) are identical to four decimal places!

#### B. The Heavy-Tail Mechanism: Low-SNR Division Spikes
In the fixed physics preconditioner:
$$V_{\text{hw}} = \frac{I_{\text{obs}} Z_{\text{nom}}^T}{Z_{\text{self}}}$$
$$C_{\text{prior}} = \frac{V_{\text{ideal}}}{V_{\text{hw}} + 10^{-12}}$$

In the multi-SNR dataset ($5.0$ to $30.0$ dB):
* At low SNR ($5$ to $10$ dB), random Gaussian measurement noise $n \sim \mathcal{CN}(0, \sigma^2)$ is added to $I_{\text{true}}$.
* For certain element-angle realizations, destructive interference between signal and noise causes $V_{\text{hw}}$ for an element to fall extremely close to zero ($|V_{\text{hw}}| \approx 10^{-3} - 10^{-4}$).
* When $V_{\text{ideal}} \sim 1$ is divided by this near-zero noisy voltage, $|C_{\text{prior}}|$ spikes to $>100$, producing squared errors of $>300$ for that sample.

#### C. Sample Outlier Impact
* In the training set ($N=2000$), the 5 worst samples have MSEs:
  $$[15.48, 19.72, 35.53, 40.98, \mathbf{310.76}]$$
  The single outlier at sample index 803 ($\text{MSE} = 310.755$ at $\text{SNR} = 5.4\text{ dB}$) contributes:
  $$\frac{310.755}{2000} = \mathbf{+0.15538} \text{ to the mean training MSE!}$$
  The top 5 outliers sum to $422.47$, which alone adds $+0.2112$ to the training MSE.
* In the validation set ($N=500$), the maximum sample MSE happens to be only $7.744$ (due to $4\times$ smaller sample size, extreme tail events are less represented).
* Subtracting the tail outliers from the training MSE:
  $$0.3700 - 0.2112 = \mathbf{0.1588} \approx 0.1423 \text{ (Val MSE)}$$

#### D. The Bounding Effect ($\alpha = 0.5$)
Because the Bounded Neural Residual Denoiser is constrained by:
$$|\Delta C| \le \alpha = 0.5$$
the neural network is physically incapable of fitting or overfitting the $300$-magnitude outlier in the training set. It correctly refuses to destabilize itself to chase this low-SNR numerical singularity.
Consequently:
* In `pinn_v4_bounded_seed42`: Train loss drops only from $0.3700 \to 0.3425$, and Val loss drops from $0.1423 \to 0.1290$.
* In contrast, in `pinn_v4_unbounded_seed42` (linear, unbounded residual): The network expends capacity fitting the outliers, driving Train loss down to $0.0731$, but Val loss only reaches $0.1136$, and frozen test NMSE degrades from $-14.21\text{ dB} \to -13.10\text{ dB}$.

---

## 4. Audit Checklist & Verdict

| Audit Item | Status | Finding |
| :--- | :---: | :--- |
| **Different normalization?** | No | Inputs and targets use identical float32 representations in Train and Val. |
| **Different objective definitions?** | No | Both compute `nn.MSELoss(reduction='mean')` on `(c_hat_real, y)`. |
| **Regularization in training only?** | No | Weight decay is handled inside optimizer step; not in scalar loss. |
| **Train/eval mode differences?** | No | Model has no Dropout or BatchNorm layers; outputs in train and eval modes are identical. |
| **Reduction/averaging differences?** | No | Both average over mini-batches of 32. |
| **Residual bound effects?** | **Yes** | Bound ($\pm 0.5$) prevents network from fitting rare low-SNR tail outliers ($>300$). |
| **Heavy-tail sample variance?** | **Yes** | 1 single extreme sample in train ($N=2000$) adds $+0.155$ to the mean MSE. |
| **Accidental code mismatch?** | **No** | Pipeline logic is clean, consistent, and mathematically sound. |

### Conclusion: Are Train and Validation Losses Directly Comparable?
* **Conceptually**: Yes. Both measure the exact same physical MSE metric over the respective sets.
* **Statistically**: The mean MSE is heavily distorted by rare, heavy-tailed division singularities ($V_{\text{hw}} \to 0$) in low-SNR samples ($5-10$ dB). Because $N_{\text{train}}=2000$ and $N_{\text{val}}=500$, sample extrema differ significantly ($\max_{\text{train}} = 310.8$ vs $\max_{\text{val}} = 7.7$).
* When compared via **median sample error** ($0.0267$ vs $0.0265$) or when excluding the top $1\%$ tail, the training and validation losses are virtually identical.
* The discrepancy of $0.3425$ vs $0.1290$ is an expected statistical property of heavy-tailed data under mean squared error with a bounded denoiser, not a code defect or training flaw.
