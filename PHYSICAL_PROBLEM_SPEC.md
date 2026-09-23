# PINN-Cal — Physical Problem Specification

## 1. Purpose

PINN-Cal is a physics-informed calibration system for a simulated
64-element uniform linear antenna array operating at 28 GHz.

The objective is to estimate per-element hardware correction factors
from an imperfect array observation.

The rebuilt formulation must represent a genuinely non-trivial
estimation problem rather than an analytically exact inversion.

---

## 2. Array Configuration

### Array

- Number of elements: 64
- Geometry: Uniform Linear Array (ULA)
- Operating frequency: 28 GHz
- Element spacing: 0.5 wavelength
- Steering range: -60° to +60°

### Wavelength

λ = c / f

### Element spacing

d = 0.5 λ

---

## 3. Ideal Excitation

For a requested steering angle θ:

v_ideal = a(θ)

where a(θ) is the complex array steering vector.

The ideal excitation represents the desired element excitation before
hardware imperfections are applied.

---

## 4. Hardware Impairment

Each antenna element has an independent complex gain/phase impairment.

For element n:

g_n = 10^(ΔG_n / 20)

φ_n = Δφ_n

γ_n = g_n exp(j φ_n)

The impaired excitation is:

v_hw = γ ⊙ v_ideal

where ⊙ denotes element-wise multiplication.

### Primary operating distribution

Phase error:

Δφ ~ N(0, 20°)

Gain error:

ΔG ~ N(0, 2 dB)

These values define the primary simulation condition.

They are simulation parameters and are not intended to imply a
universal real-world hardware-error distribution.

---

## 5. Mutual Coupling

The physical system contains a reciprocal complex impedance matrix.

### True coupling matrix

Z_true is the coupling matrix used to generate the physical
observation.

It satisfies:

Z_true = Z_true^T

The real/Hermitian part is required to remain numerically positive
and the matrix must remain well-conditioned.

### Nominal coupling matrix

Z_nom is the matrix available to the calibration model.

Z_nom is not necessarily identical to Z_true.

The primary operating mismatch is:

±20%

for the coupling parameters.

Additional mismatch levels will be used during robustness testing.

---

## 6. Physical Observation

The true coupled current is generated from:

Z_true I_true = V_hw Z_self,true

where Z_self,true is the mean self impedance represented by the
true network.

Therefore:

I_true = Z_true^-1 (V_hw Z_self,true)

The measurement system does not provide I_true directly.

---

## 7. Measurement Noise

The observed current contains complex additive white Gaussian noise:

I_obs = I_true + n

The primary operating condition is:

SNR = 20 dB

The noise realization is not known to PINN-Cal.

Additional SNR conditions will be used for robustness testing.

---

## 8. What PINN-Cal Knows

The model may know:

- Number of antenna elements
- Operating frequency
- Array geometry
- Element spacing
- Target steering angle
- Nominal coupling matrix Z_nom

The model receives the noisy observed array state.

---

## 9. What PINN-Cal Does NOT Know

The model does not directly know:

- Individual element phase errors
- Individual element gain errors
- True coupling matrix Z_true
- Individual measurement-noise realization

The model must infer the required correction from the available
observation.

---

## 10. PINN Input

The model input contains:

- Real part of the 64-element observed complex current
- Imaginary part of the 64-element observed complex current
- sin(θ)
- cos(θ)

Total input dimension:

64 + 64 + 2 = 130

---

## 11. PINN Output

The model predicts one complex correction coefficient per antenna
element:

C_n = A_n exp(j ψ_n)

The network output is represented as:

- 64 real values
- 64 imaginary values

Total output dimension:

128

---

## 12. Correction Objective

The correction is applied element-wise to the observed hardware
excitation after the physical relationship between current and
voltage is reconstructed.

The objective is to recover an array state that is consistent with
the desired ideal excitation under the physical coupling model.

---

## 13. Physics Constraint

The primary physics residual operates in the complex coupled-current
domain.

The predicted correction is passed through the physical impedance
network.

The resulting corrected current is compared with the desired ideal
coupled current.

The principal residual is therefore:

L_physics =
|| I_corrected - I_ideal ||²
--------------------------------
|| I_ideal ||²

averaged over the batch.

The physics loss must remain differentiable with respect to the
network output.

---

## 14. Data Loss

The supervised target remains the known inverse hardware correction
for simulation training:

L_data = MSE(C_pred, C_true)

The data target is available during training because this is a
controlled simulation.

---

## 15. Total Training Objective

The training objective is:

L_total = L_data + λ L_physics

The value and scheduling of λ are experimental parameters and must
be validated rather than assumed to improve performance.

---

## 16. Important Baseline

A direct analytical inversion using the nominal model must be treated
as a formal baseline.

This baseline is important because the current observation model can
be partially inverted analytically.

PINN-Cal must therefore demonstrate value under conditions where the
nominal analytical inversion is imperfect, such as:

- measurement noise
- coupling-model mismatch
- combinations of both

The PINN must never be presented as superior merely because an
analytical baseline was omitted.

---

## 17. Evaluation Domains

PINN-Cal will be evaluated in four domains.

### 17.1 Calibration domain

- Complex correction NMSE
- Phase error
- Gain error
- Element-wise relative correction error

### 17.2 Physical-state domain

- Relative coupled-current error

### 17.3 Beam domain

- Beam pointing error
- 3-dB beamwidth
- Peak sidelobe level
- Pattern correlation

### 17.4 Robustness domain

Evaluate across:

- SNR
- Coupling mismatch
- Steering angle
- Random seeds

---

## 18. Primary Experimental Condition

The initial primary condition for the rebuilt system is:

| Parameter | Value |
|---|---:|
| Elements | 64 |
| Frequency | 28 GHz |
| Spacing | 0.5 λ |
| Phase error σ | 20° |
| Gain error σ | 2 dB |
| Measurement SNR | 20 dB |
| Coupling mismatch | ±20% |
| Steering range | -60° to +60° |

---

## 19. Scientific Constraints

The project must not:

- claim that the simplified impedance model is a full-wave EM solver
- claim real-world hardware performance without supporting measurements
- use fabricated or interpolated experimental results
- treat normalized beam-pattern similarity as proof of unique calibration
- omit the analytical baseline
- use hard-coded scientific metrics in the application

---

## 20. Application Requirement

The eventual application must be computation-driven.

The interface must not be a static visualization.

Changing a physical parameter must trigger a real update in the
underlying simulation/model state.

Examples:

- steering angle changes beam calculation
- phase-error level changes hardware impairment
- gain-error level changes hardware impairment
- SNR changes observation noise
- coupling mismatch changes the true physical network
- calibration triggers actual model inference
- beam plots are generated from computed complex array states
- metrics are calculated from the current experiment

The frontend must not contain hard-coded scientific results.

---

## 21. Architecture Principle

The physics engine is the single source of truth.

Planned structure:

Physical Engine
    ↓
Dataset / Experiment Engine
    ↓
PINN / Baseline Models
    ↓
Evaluation Engine
    ↓
API / Backend
    ↓
Live Interactive Application

The web application must consume the same physical/model logic used
for experiments rather than implementing a separate simplified
simulation.

---

## 22. Current Development Rule

Scientific assumptions must be validated before large training runs.

The project should proceed in this order:

1. Physical model
2. Observation model
3. Analytical baseline
4. Dataset
5. PINN
6. Evaluation
7. Robustness
8. Live application

The application layer must not drive the scientific design.
