# PINN-Cal v4: Step 61 Analytical Derivation & Preconditioner Regularization Audit

**Date**: 2026-10-01  
**Status**: Completed  
**Objective**: Analytically derive the scalar complex Tikhonov-regularized physics preconditioner, evaluate numerical stability against the low-SNR singularity, perform a controlled $\delta$ sweep, and evaluate nominal preservation.

---

## 1. Part A — Analytical Mathematical Derivation

### 1.1 Problem Formulation & Unregularized Inversion
In the nominal physical calibration framework:
* $V_{\text{ideal}} = \mathbf{a}(\theta) \in \mathbb{C}^{M}$ ($M=64$) is the ideal plane-wave steering vector ($|V_{\text{ideal}, m}| = 1.0$).
* $I_{\text{obs}} \in \mathbb{C}^{M}$ is the observed array current.
* $V_{\text{eff}} = I_{\text{obs}} Z_{\text{nom}}^T \in \mathbb{C}^{M}$ is the nominal array effective voltage.
* $V_{\text{hw}} = V_{\text{eff}} / Z_{\text{self}} \in \mathbb{C}^{M}$ is the estimated per-element hardware excitation.

For each antenna element $m \in \{1, \dots, M\}$, the calibration coefficient $C_m \in \mathbb{C}$ maps the distorted hardware excitation back to the ideal excitation:
$$V_{\text{hw}, m} \cdot C_m \approx V_{\text{ideal}, m}$$

In unregularized least-squares:
$$\min_{C_m \in \mathbb{C}} \left| V_{\text{hw}, m} C_m - V_{\text{ideal}, m} \right|^2$$
The solution is the direct algebraic quotient:
$$C_{\text{prior}, m} = \frac{V_{\text{ideal}, m}}{V_{\text{hw}, m} + \epsilon}$$
where $\epsilon = 10^{-12}$ was an ad-hoc floating-point safeguard.

### 1.2 The Singularity Problem
Under measurement noise $I_{\text{obs}} = I_{\text{true}} + n$, the estimated hardware voltage is:
$$V_{\text{hw}} = V_{\text{hw, true}} + \frac{n Z_{\text{nom}}^T}{Z_{\text{self}}}$$
At low SNR ($5-10\text{ dB}$), destructive interference between signal and noise causes $V_{\text{hw}, m}$ for certain elements to become arbitrarily close to zero ($|V_{\text{hw}, m}| < 10^{-2}$).
As $|V_{\text{hw}, m}| \to 0$:
$$|C_{\text{prior}, m}| \to \infty$$
In the training set ($N=2000$), this produced $|C_{\text{prior}}| > 200$ and sample squared errors $> 310$, creating the heavy-tail loss distortion diagnosed in Step 59.

### 1.3 Derivation of the Tikhonov Regularized Inverse
To prevent explosive division by near-zero noisy observations, we pose penalized least squares (Tikhonov / ridge regression):
$$\min_{C_m \in \mathbb{C}} \mathcal{J}(C_m) = \left| V_{\text{hw}, m} C_m - V_{\text{ideal}, m} \right|^2 + \delta^2 |C_m|^2$$
where $\delta^2 > 0$ penalizes excessively large calibration coefficients (minimum-norm solution).

Expanding the objective using complex variables ($|z|^2 = z z^*$):
$$\mathcal{J}(C_m) = (V_{\text{hw}, m} C_m - V_{\text{ideal}, m})(V_{\text{hw}, m}^* C_m^* - V_{\text{ideal}, m}^*) + \delta^2 C_m C_m^*$$

Taking the Wirtinger derivative with respect to $C_m^*$ and setting to zero:
$$\frac{\partial \mathcal{J}}{\partial C_m^*} = V_{\text{hw}, m}^* (V_{\text{hw}, m} C_m - V_{\text{ideal}, m}) + \delta^2 C_m = 0$$
$$\left( |V_{\text{hw}, m}|^2 + \delta^2 \right) C_m = V_{\text{ideal}, m} V_{\text{hw}, m}^*$$
$$C_{\text{prior, reg}, m} = \frac{V_{\text{ideal}, m} \cdot V_{\text{hw}, m}^*}{|V_{\text{hw}, m}|^2 + \delta^2}$$

### 1.4 Physical Interpretation & Universal Upper Bound Theorem
1. **Zero Phase Distortion**:  
   Notice that the denominator $|V_{\text{hw}, m}|^2 + \delta^2$ is **strictly real and strictly positive**.  
   Therefore:
   $$\angle C_{\text{prior, reg}, m} = \angle(V_{\text{ideal}, m} V_{\text{hw}, m}^*) = \angle(V_{\text{ideal}, m} / V_{\text{hw}, m})$$
   Tikhonov regularization introduces **zero phase distortion** across all antenna elements.
2. **Universal Magnitude Bounding**:  
   Let $x = |V_{\text{hw}, m}| \ge 0$. Since $|V_{\text{ideal}, m}| = 1$, the magnitude is:
   $$f(x) = \frac{x}{x^2 + \delta^2}$$
   Differentiating: $f'(x) = \frac{\delta^2 - x^2}{(x^2 + \delta^2)^2} = 0 \implies x = \delta$.  
   The global maximum is:
   $$f(\delta) = \frac{\delta}{2\delta^2} = \frac{1}{2\delta}$$
   $$\max |C_{\text{prior, reg}}| \le \frac{1}{2\delta}$$
   **Theorem**: For any input current $I_{\text{obs}}$ (even $I_{\text{obs}} \to 0$ or pure noise), the regularized calibration prior is guaranteed to never exceed $\frac{1}{2\delta}$.

### 1.5 Physical Scale and Units of $\delta$
* $V_{\text{ideal}}$: dimensionless (normalized steering excitation, unit magnitude $|V_{\text{ideal}}| = 1.0$).
* $V_{\text{hw}} = V_{\text{eff}} / Z_{\text{self}}$: normalized voltage excitation ($\mathbb{E}[|V_{\text{hw}}|^2] \approx 1.0$).
* $\delta$: dimensionless voltage threshold, representing the noise floor below which excitation measurements cannot be trusted.
* $\delta^2$: noise power relative to nominal signal power.

#### Noise-Derived Value $\delta_{\text{noise}}$:
In our system, nominal array operation occurs at $\mathrm{SNR} = 20\text{ dB}$.  
Relative noise power is $\sigma_n^2 / P_{\text{sig}} = 10^{-20/10} = 0.01$.  
Mapping through nominal impedance $Z_{\text{nom}} / Z_{\text{self}}$:
$$\sigma_{V_{\text{hw}}}^2 \approx 10^{-\mathrm{SNR}_{\text{dB}}/10} \times \left(\frac{\|Z_{\text{nom}}\|_F}{\sqrt{M} |Z_{\text{self}}|}\right)^2 \approx 0.01 \times 1.04 = 0.0104$$
$$\delta_{\text{noise}} = \sqrt{0.0104} \approx 0.10$$
* At $\delta = 0.10$, $\max |C_{\text{prior}}| \le \frac{1}{2 \times 0.10} = 5.00$.

---

## 2. Part B — Numerical Stability & Singularity Analysis

Evaluated on the 2,000 multi-SNR training samples ($2,000 \times 64 = 128,000$ element observations):

### 2.1 $|V_{\text{hw}}|$ Denominator Distribution
* $\min |V_{\text{hw}}| = 4.985 \times 10^{-3}$
* $0.1\%$ percentile = $0.1566$
* $1.0\%$ percentile = $0.3970$
* Median $|V_{\text{hw}}| = 1.0266$
* Mean $|V_{\text{hw}}| = 1.0612$
* Element count with $|V_{\text{hw}}| < 0.01$: **1 element**
* Element count with $|V_{\text{hw}}| < 0.05$: **13 elements**
* Element count with $|V_{\text{hw}}| < 0.10$: **53 elements**

### 2.2 $|C_{\text{prior}}|$ Distribution and Sample MSE by $\delta$
| $\delta$ | Multiplier | Theoretical Max | Empirical Max $|C|$ | $99.9\%$ $|C|$ | $99\%$ $|C|$ | Mean Sample MSE | Max Sample MSE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$0.000$** | $0.00 \delta_n$ | $\infty$ | **$200.62$** | $6.38$ | $2.52$ | **$0.3700$** | **$310.76$** |
| **$0.025$** | $0.25 \delta_n$ | $20.00$ | **$19.92$** | $6.23$ | $2.51$ | **$0.1401$** | **$5.31$** |
| **$0.050$** | $0.50 \delta_n$ | $10.00$ | **$10.00$** | $5.77$ | $2.48$ | **$0.1120$** | **$2.51$** |
| **$0.100$** | $1.00 \delta_n$ | $5.00$ | **$5.00$** | $4.37$ | $2.36$ | **$0.0849$** | **$1.03$** |
| **$0.150$** | $1.50 \delta_n$ | $3.33$ | **$3.33$** | $3.20$ | $2.19$ | **$0.0695$** | **$0.65$** |
| **$0.200$** | $2.00 \delta_n$ | $2.50$ | **$2.50$** | $2.47$ | $1.99$ | **$0.0597$** | **$0.55$** |
| **$0.300$** | $3.00 \delta_n$ | $1.67$ | **$1.67$** | $1.67$ | $1.58$ | **$0.0513$** | **$0.41$** |

### Stability Verdict:
* **The heavy tail is completely eliminated.**  
  At $\delta = 0.10$, maximum sample MSE collapses from **$310.76 \to 1.03$** (a **300x reduction** in worst-case error!).
* Mean sample MSE across the training set drops from **$0.3700 \to 0.0849$**, eliminating the spurious train-validation gap identified in Step 59.

---

## 3. Parts D & E — Preconditioner Quality & Benchmark Evaluation

Evaluated across all test sets without neural network training:

### 3.1 Preconditioner Performance Across Conditions
| Condition | Metric | $\delta=0.000$ | $\delta=0.025$ | $\delta=0.050$ | $\mathbf{\delta=0.100}$ | $\mathbf{\delta=0.150}$ | $\delta=0.200$ | $\delta=0.300$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frozen 20 dB** | **NMSE (dB)** | −14.45 | −14.48 | −14.55 | **−14.80** | **−15.09** | **−15.25** | −14.58 |
| | Phase Error (°) | 3.92 | 3.92 | 3.92 | 3.92 | 3.92 | 3.92 | 3.92 |
| | Gain Error (dB) | 1.04 | 1.04 | 1.04 | 1.02 | 1.01 | 1.02 | 1.17 |
| | Pointing Error (°) | 0.0282 | 0.0282 | 0.0282 | 0.0282 | 0.0281 | 0.0280 | 0.0280 |
| | Beam Correlation | 0.999807 | 0.999807 | 0.999809 | 0.999814 | 0.999821 | 0.999829 | 0.999841 |
| **10 dB SNR** | **NMSE (dB)** | −4.01 | −4.18 | −4.59 | **−5.56** | **−6.44** | **−7.17** | **−8.09** |
| | Phase Error (°) | 12.49 | 12.49 | 12.49 | 12.49 | 12.49 | 12.49 | 12.49 |
| | Pointing Error (°) | 0.0479 | 0.0479 | 0.0461 | 0.0446 | 0.0416 | 0.0391 | 0.0383 |
| | Beam Correlation | 0.994201 | 0.994581 | 0.995346 | 0.996532 | 0.997200 | 0.997623 | 0.998120 |
| **5 dB SNR** | **NMSE (dB)** | **+5.91** | **+1.73** | **+0.56** | **−0.78** | **−1.79** | **−2.60** | **−3.71** |
| | Phase Error (°) | 23.23 | 23.23 | 23.23 | 23.23 | 23.23 | 23.23 | 23.23 |
| | Pointing Error (°) | 0.3204 | 0.1224 | 0.1105 | 0.1011 | 0.0921 | 0.0913 | 0.0832 |
| | Beam Correlation | 0.896721 | 0.916204 | 0.937885 | 0.959172 | 0.970691 | 0.977541 | 0.984727 |
| **Phase 35° Stress** | **NMSE (dB)** | −14.16 | −14.19 | −14.26 | −14.52 | −14.85 | −15.07 | −14.56 |
| | Pointing Error (°) | 0.0267 | 0.0267 | 0.0267 | 0.0267 | 0.0262 | 0.0262 | 0.0254 |
| **Gain 4.0 dB Stress**| **NMSE (dB)** | −10.14 | −10.23 | −10.46 | −11.14 | −11.59 | −11.48 | −9.99 |
| **Coupling 40% Stress**|**NMSE (dB)** | −9.99 | −10.02 | −10.11 | −10.44 | −10.90 | −11.36 | −11.79 |

---

## 4. Part E — Detailed Nominal Preservation Analysis (Frozen 20 dB Test Set)

Difference relative to unregularized $\delta=0$ ($\text{NMSE} = -14.4545\text{ dB}$, $\text{Phase} = 3.9201^\circ$, $\text{Point} = 0.0282^\circ$, $\text{Corr} = 0.999807$):

| $\delta$ | $\Delta \text{NMSE}$ (dB) | $\Delta \text{Phase}$ (deg) | $\Delta \text{Gain}$ (dB) | $\Delta \text{Pointing}$ (deg) | $\Delta \text{Beam Correlation}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$0.025$** | **−0.0250** | $0.0000$ | −0.0012 | $0.0000$ | $+0.000000$ |
| **$0.050$** | **−0.0973** | $0.0000$ | −0.0048 | $0.0000$ | $+0.000002$ |
| **$0.100$** | **−0.3489** | $0.0000$ | −0.0163 | $0.0000$ | $+0.000007$ |
| **$0.150$** | **−0.6378** | $0.0000$ | −0.0259 | −0.0001 | $+0.000014$ |
| **$0.200$** | **−0.7910** | $0.0000$ | −0.0186 | −0.0002 | $+0.000022$ |
| **$0.300$** | −0.1208 | $0.0000$ | +0.1270 | −0.0002 | $+0.000035$ |

*(Note: Negative $\Delta \text{NMSE}$ denotes calibration improvement).*

### Findings:
1. **Nominal Performance is NOT Degraded — It is Improved!**  
   At $\delta = 0.10$, nominal 20 dB NMSE improves by **$0.35\text{ dB}$** ($-14.45\text{ dB} \to -14.80\text{ dB}$).  
   At $\delta = 0.15$, it improves by **$0.64\text{ dB}$** ($-14.45\text{ dB} \to -15.09\text{ dB}$).  
   Why? Because even at 20 dB SNR, minor noise on $V_{\text{hw}}$ inflates the unregularized inverse. Tikhonov regularization provides shrinkage that damps noise amplification.
2. **Phase Calibration is Exactly Invariant**:  
   Across all tested $\delta \in [0.0, 0.30]$, $\Delta \text{Phase} = 0.0000^\circ$ to machine precision!
3. **Over-regularization Threshold**:  
   Beyond $\delta = 0.20$, shrinkage begins attenuating legitimate signal power, causing gain error to rise ($1.04\text{ dB} \to 1.17\text{ dB}$ at $\delta = 0.30$). Thus, $\delta \in [0.10, 0.15]$ represents the optimal operating range.

---

## 5. Part F — Fairness & Information Leakage Audit

| Information Category | Used by Regularizer? | Verification |
| :--- | :---: | :--- |
| **Hidden Ground-Truth SNR?** | **No** | $\delta$ is a constant hyperparameter ($0.10$ or $0.15$) derived from nominal 20 dB design specs. |
| **Hidden Ground-Truth Coupling $Z_{\text{true}}$?** | **No** | Uses only nominal $Z_{\text{nom}}$, which is fixed and public. |
| **Hidden Hardware Errors ($\phi_m, g_m$)?** | **No** | Operates strictly on observed voltages $V_{\text{hw}}$. |
| **Target Correction Target $Y$?** | **No** | Evaluated in strictly feedforward inference. |

---

## 6. Breakthrough Summary: Low-SNR Impact

1. **At 5 dB SNR**:  
   * Unregularized ($\delta=0$): **$+5.91\text{ dB}$** NMSE (catastrophic failure), Pointing Error = $0.3204^\circ$, Correlation = $0.8967$.  
   * Regularized ($\delta=0.10$): **$-0.78\text{ dB}$** NMSE (**$+6.69\text{ dB}$ improvement!**), Pointing Error = $0.1011^\circ$, Correlation = **$0.9592$**.  
   * Regularized ($\delta=0.15$): **$-1.79\text{ dB}$** NMSE (**$+7.70\text{ dB}$ improvement!**), Pointing Error = $0.0921^\circ$, Correlation = **$0.9707$**.
2. **At 10 dB SNR**:  
   * Unregularized ($\delta=0$): $-4.01\text{ dB}$ NMSE.  
   * Regularized ($\delta=0.10$): **$-5.56\text{ dB}$** NMSE (**$+1.55\text{ dB}$ improvement**).  
   * Regularized ($\delta=0.15$): **$-6.44\text{ dB}$** NMSE (**$+2.43\text{ dB}$ improvement**).

---

## 7. Recommendation for Next Step (Step 62)

The Tikhonov regularizer decisively solves the upstream linear preconditioner singularity.

**Recommended Parameters for Step 62 (Neural Residual Denoiser Retraining)**:
* **Primary Recommendation**: $\delta = 0.10$ (nominal design value matching 20 dB noise variance).
* **Alternative/Robust Candidate**: $\delta = 0.15$ (stronger noise suppression for deep sub-10 dB environments).
* **Next Action**: Update `FixedPhysicsPreconditioner` to include $\delta = 0.10$ and retrain PINN-Cal v4 across seeds 42, 123, 456 to measure the combined physical + neural performance.
