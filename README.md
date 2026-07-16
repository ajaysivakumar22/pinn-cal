# PINN-Cal: Physics-Informed Neural Calibration for 5G Massive MIMO

**Final Year Project — Dr. Mahalingam College of Engineering and Technology**
Department of Electronics and Communication Engineering

## Team
- Mowsikan H (727623BEA030)
- Ajay S (727623BEA039)
- Shanjesh P S (727623BEA050)

**Guide:** Ms. K.C. Rajarajeshwari, AP/ECE

---

## Project Summary

PINN-Cal is a software framework that calibrates 5G antenna arrays by embedding
electromagnetic laws (reciprocity, mutual coupling, energy conservation) directly
into the neural network training process.

**Key Result:** PINN-Cal achieves near-identical calibration accuracy to a standard
neural network while using **10x fewer training samples** (100 vs 1000).

| Method        | Samples | Phase Error | Gain Error | Data Efficiency |
|---------------|---------|-------------|------------|-----------------|
| Baseline MLP  | 1000    | 4.00°       | 0.43 dB    | 1x              |
| PINN-Cal      | 100     | 4.12°       | 0.61 dB    | **10x better**  |

---

## Setup

### Requirements
- Python 3.11
- NVIDIA GPU (CUDA 12.1+) or CPU
- WSL2 (if on Windows)

### Installation
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy matplotlib pandas scikit-learn deepxde tensorboard
```

---

## Running the Project

Run these scripts in order:

```bash
# Stage 1: Generate dataset
python array_simulator.py

# Stage 2: Train baseline (1000 samples)
python baseline_model.py

# Stage 3: Train PINN-Cal (100 samples)
python pinn_cal_v2.py

# Stage 4: Generate comparison plots
python results_comparison.py

# Stage 5: Ablation study
python ablation_study.py
```

---

## Project Structure
---

## Physics Constraints

Three electromagnetic laws embedded in the loss function:

1. **Reciprocity** — H_ij = H_ji (channel matrix symmetry)
2. **Mutual Coupling** — Impedance matrix consistent with half-wave dipole theory
3. **Energy Conservation** — Total radiated power ≤ input power

Loss function:
---

## Hardware Used
- CPU: AMD Ryzen 5 4600H
- GPU: NVIDIA GeForce GTX 1650 (4GB)
- OS: Ubuntu 26.04 (WSL2)
- Training time: ~5 minutes per model
