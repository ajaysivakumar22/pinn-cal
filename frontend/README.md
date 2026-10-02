# PINN-Cal Frontend

React + TypeScript + Vite engineering interface for real-time visualization and control of antenna calibration.

## Technology Stack
- **Framework:** React 19 + TypeScript
- **Bundler:** Vite 8
- **State Management:** Zustand
- **Charting & Plots:** Plotly.js / react-plotly.js
- **Array Visualization:** Custom responsive SVG (phase $\to$ hue, magnitude $\to$ radius)

## Features
- **Laboratory Instrument Aesthetic:** Clean, high-density light theme with technical typography.
- **Interactive 64-Element Array:** Inspect element excitation, observed currents, corrections, and residual states on hover/click.
- **Dual-Trace Radiation Patterns:** Real-time normalized beam power comparisons with interactive crosshair telemetry.
- **Neutral Comparison Panel:** Unbiased side-by-side presentation of Analytical baseline vs. PINN calibration.
- **Dedicated Robustness Lab:** Parameter sweeps with live backend re-computation.
- **PINN Compatibility Guard:** Clear notification when simulating array configurations outside the trained 64-element @ 28 GHz envelope.

## Setup & Running

1. Install Node dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Start the Vite development server:
   ```bash
   npm run dev
   ```

3. Build production bundle:
   ```bash
   npm run build
   ```

4. Run tests:
   ```bash
   npm test
   ```
