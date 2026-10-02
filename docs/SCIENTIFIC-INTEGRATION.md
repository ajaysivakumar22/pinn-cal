# Scientific Core Integration Guide

## 1. Frozen Scientific Baseline

PINN-Cal integrates a strictly audited, frozen scientific baseline tagged at:
`scientific-core-v1.0`

The scientific core files are treated as **read-only**:
- `physics_core.py`: Physical constants, steering vectors ($a_n(\theta)$), nominal impedance matrices ($Z_{\text{nom}}$), reciprocity definitions.
- `coupling_model.py`: Mutual coupling matrix generation, compact reciprocal network equations, impedance perturbation ($Z_{\text{true}}$).
- `calibration_dataset.py` & `split_physics_dataset.py`: 1200 controlled simulation instances (800 train, 200 val, 200 test, seed 2026).
- `analytical_baseline.py`: Closed-form mutual coupling compensation using nominal impedance inversion ($C_{\text{ana}} = Z_{\text{nom}}^{-1} Z_{\text{true}}$).
- `pinn_cal_final.py`: PyTorch network definition (`PINNCalFinalNet`, 83,904 parameters).
- `PHYSICAL_PROBLEM_SPEC.md`: Formal mathematical problem formulation.

---

## 2. Mathematical Problem Formulation

### 2.1 Array Geometry and Ideal Signal
For an $N$-element uniform linear array (ULA) with inter-element spacing $d = 0.5\lambda$:
$$x_n = \left(n - \frac{N-1}{2}\right) d, \quad n = 0, \dots, N-1$$
The ideal steering vector toward angle $\theta$ is:
$$a_n(\theta) = \exp\left(j \frac{2\pi}{\lambda} x_n \sin\theta\right)$$
The ideal excitation voltage vector is:
$$V_{\text{ideal}} = a(\theta)$$

### 2.2 Mutual Coupling & Hardware Impairment
Mutual coupling is modeled via a compact reciprocal impedance network:
$$Z_{m,n} = \begin{cases}
Z_{\text{self}} & m = n \\
Z_{\text{self}} \cdot c_1 \cdot \exp(-\alpha |m-n|) \cdot \exp(-j \beta |m-n|) & m \neq n
\end{cases}$$
where:
- $Z_{\text{self}} = 73.1 + 42.5j\ \Omega$
- $c_1 = 0.12$ (nearest-neighbor coupling)
- $\alpha = 0.55$ (spatial decay)
- $\beta = 0.35\text{ rad}$ (phase delay)

Hardware impairments introduce random diagonal perturbation $H = \text{diag}(h_0, \dots, h_{N-1})$:
$$h_n = (1 + \delta g_n) \exp(j \delta \phi_n)$$
with $\delta \phi_n \sim \mathcal{N}(0, \sigma_\phi^2)$ ($\sigma_\phi = 20^\circ$) and $\delta g_n \sim \mathcal{N}(0, \sigma_g^2)$ ($\sigma_g = 2\text{ dB}$).

Coupled currents with additive thermal noise ($w \sim \mathcal{CN}(0, \sigma_w^2)$ at $\text{SNR} = 20\text{ dB}$):
$$I_{\text{obs}} = Z_{\text{true}}^{-1} (H V_{\text{ideal}}) + w$$

### 2.3 Calibration Objective
Calibration determines a complex correction vector $C = [c_0, \dots, c_{N-1}]^T$ such that:
$$V_{\text{corrected}} = C \odot (H V_{\text{ideal}}) \approx V_{\text{ideal}}$$
Ground truth calibration:
$$C_{\text{true}} = \frac{V_{\text{ideal}}}{H V_{\text{ideal}}} = H^{-1}$$

---

## 3. PINN Architecture & Constraints

- **Input Dimension:** 130 features ($64 \text{ Re}(I_{\text{obs}}) + 64 \text{ Im}(I_{\text{obs}}) + \sin\theta + \cos\theta$).
- **Hidden Layers:** $130 \to 256 \to 128 \to 64 \to 128$. Each linear layer is followed by `BatchNorm1d`, `ReLU`, and `Dropout(0.2)`.
- **Output Dimension:** 128 features ($64 \text{ Re}(C_{\text{pred}}) + 64 \text{ Im}(C_{\text{pred}})$).
- **Total Trainable Parameters:** 83,904.
- **Physical Loss:**
  $$V_{\text{obs}} = Z_{\text{nom}} I_{\text{obs}}$$
  $$V_{\text{corr}} = C_{\text{pred}} \odot V_{\text{obs}}$$
  $$I_{\text{corr}} = Z_{\text{nom}}^{-1} V_{\text{corr}}$$
  $$\mathcal{L}_{\text{phys}} = \|I_{\text{corr}} - I_{\text{ideal}}\|_2^2$$

---

## 4. Scientific Honesty & Validation Benchmarks

On the isolated, held-out test dataset (200 samples, seed 2026):

| Metric | Analytical Baseline | Final PINN (`pinn_cal_final_best.pth`) |
| :--- | :--- | :--- |
| **NMSE** | **−14.4545 dB** | −7.7489 dB |
| **Phase Error** | **3.9201°** | 16.1990° |
| **Gain Error** | **1.0404 dB** | 1.7084 dB |
| **Relative Error** | **15.1264%** | 36.7887% |

**System Obligation:**
The application strictly reflects these empirical results. The user interface does not claim PINN superiority over analytical methods. Both methods are evaluated neutrally using identical physical metrics.
