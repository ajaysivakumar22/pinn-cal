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