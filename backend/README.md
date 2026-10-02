# PINN-Cal Backend

FastAPI service powering real-time simulation, mutual coupling calculation, analytical calibration, and PINN neural inference for 5G massive MIMO arrays.

## Features
- **Dynamic Hardware Detection:** Automatically selects CUDA device (`cuda:0`, NVIDIA GeForce GTX 1650) with graceful CPU fallback.
- **Model Caching:** Loads `models/pinn_cal_final_best.pth` once at startup using secure `weights_only=True`.
- **Unified Simulation Endpoint:** Computes array excitation, impairments, calibration corrections, beam synthesis, and telemetry in a single POST request (`/api/simulation`).
- **Robustness Engine:** On-demand Monte Carlo sweeps across SNR, coupling uncertainty, and steering angles.
- **Frozen Core Protection:** Non-intrusive scientific adapter layer preserving `scientific-core-v1.0`.

## Installation & Setup

1. Activate your Python environment (Python 3.11 with PyTorch CUDA):
   ```bash
   python --version
   ```

2. Install dependencies:
   ```bash
   pip install fastapi uvicorn pydantic numpy scipy torch pytest httpx
   ```

3. Run the backend development server:
   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. Interactive API Documentation:
   Visit [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

## Testing
Run the backend test suite:
```bash
python -m pytest backend/tests/ -v
```
