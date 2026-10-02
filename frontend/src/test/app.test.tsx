import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { App } from '../App';
import { TopStatusBar } from '../components/layout/TopStatusBar';
import { SideNavigation } from '../components/layout/SideNavigation';
import { ControlPanel } from '../components/controls/ControlPanel';
import { MetricsPanel } from '../components/metrics/MetricsPanel';
import { CalibrationPanel } from '../components/charts/CalibrationPanel';
import { AntennaArray } from '../components/array/AntennaArray';
import { ElementInspector } from '../components/array/ElementInspector';
import { PinnDetectorFlow } from '../components/array/PinnDetectorFlow';
import { useAppStore } from '../store/appStore';
import * as apiClient from '../api/client';
import type { SimulationResponse } from '../types/api';

const mockSimulationResponse: SimulationResponse = {
  request: {
    num_elements: 64,
    frequency_hz: 28e9,
    element_spacing_lambda: 0.5,
    target_angle_deg: 0,
    phase_sigma_deg: 20,
    gain_sigma_db: 2,
    snr_db: 20,
    coupling_mismatch_fraction: 0.2,
    seed: 42,
  },
  pinn_available: true,
  pinn_compatibility_message: 'PINN model active: 64 elements @ 28 GHz.',
  elements: Array.from({ length: 64 }, (_, i) => ({
    element_index: i,
    ideal_real: 1.0,
    ideal_imag: 0.0,
    ideal_magnitude: 1.0,
    ideal_phase_deg: 0.0,
    hardware_real: 0.9,
    hardware_imag: 0.1,
    hardware_magnitude: 0.905,
    hardware_phase_deg: 6.34,
    gain_error_db: -0.86,
    phase_error_deg: 6.34,
    observed_real: 0.012,
    observed_imag: -0.004,
    analytical_correction_real: 1.05,
    analytical_correction_imag: -0.11,
    analytical_correction_magnitude: 1.055,
    analytical_correction_phase_deg: -5.98,
    pinn_correction_real: 1.02,
    pinn_correction_imag: -0.09,
    pinn_correction_magnitude: 1.023,
    pinn_correction_phase_deg: -5.04,
    true_phase_error_deg: 6.34,
    true_gain_error_db: -0.86,
    analytical_estimated_phase_error_deg: 5.98,
    analytical_estimated_gain_error_db: -0.46,
    analytical_phase_residual_deg: 0.36,
    analytical_gain_residual_db: -0.40,
    pinn_estimated_phase_error_deg: 5.04,
    pinn_estimated_gain_error_db: -0.20,
    phase_error_residual_deg: 1.30,
    gain_error_residual_db: -0.66,
    compensation_phase_deg: -5.04,
    compensation_gain_db: 0.20,
  })),
  beam: {
    ideal: { angles_deg: [-30, 0, 30], power_db: [-25, 0, -25] },
    uncalibrated: { angles_deg: [-30, 0, 30], power_db: [-20, -1.2, -22] },
    analytical: { angles_deg: [-30, 0, 30], power_db: [-24.8, -0.05, -24.8] },
    pinn: { angles_deg: [-30, 0, 30], power_db: [-23.5, -0.3, -23.5] },
  },
  calibration: {
    analytical: {
      nmse_db: -14.45,
      mean_phase_error_deg: 3.92,
      mean_gain_error_db: 1.04,
      mean_relative_error_pct: 15.12,
      current_residual_pct: 15.12,
    },
    pinn: {
      nmse_db: -7.75,
      mean_phase_error_deg: 16.20,
      mean_gain_error_db: 1.71,
      mean_relative_error_pct: 36.79,
      current_residual_pct: 36.79,
    },
    pinn_available: true,
  },
  metrics: {
    target_angle_deg: 0,
    beam_peak_angle_deg: 0.05,
    pointing_error_deg: 0.05,
    beam_correlation: 0.998,
    beamwidth_3db_deg: 1.78,
    psll_db: -13.2,
    analytical_nmse_db: -14.45,
    pinn_nmse_db: -7.75,
  },
};

describe('PINN-Cal Frontend Component Tests', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    useAppStore.setState({
      activeView: 'simulation',
      simulationResult: null,
      simulateState: 'idle',
      simulateError: null,
      health: {
        status: 'ok',
        timestamp: Date.now(),
        uptime_s: 100,
        scientific_core_available: true,
        scientific_core_error: null,
        pinn_model_loaded: true,
        pinn_model_error: null,
      },
      systemStatus: {
        python_version: '3.11.15',
        pytorch_version: '2.5.1+cu121',
        cuda_available: true,
        gpu_name: 'NVIDIA GeForce GTX 1650',
        cuda_device_count: 1,
        platform: 'Windows',
        scientific_core_available: true,
        scientific_core_error: null,
        backend_uptime_s: 100,
      },
      modelStatus: null,
      selectedElement: null,
    });
  });

  // 1. Render tests
  it('renders TopStatusBar telemetry and SideNavigation brand correctly', () => {
    render(
      <>
        <SideNavigation />
        <TopStatusBar onOpenMobileNav={vi.fn()} />
      </>
    );
    expect(screen.getAllByText('PINN-Cal').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Physics-Informed Neural Calibration')).toBeInTheDocument();
    expect(screen.getByText(/PATCHES/i)).toBeInTheDocument();
    expect(screen.getByText(/mmWAVE/i)).toBeInTheDocument();
    expect(screen.getByText(/CUDA ACCEL/i)).toBeInTheDocument();
  });

  it('renders ControlPanel with array controls', () => {
    render(<ControlPanel />);
    expect(screen.getByLabelText(/Elements/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Frequency/i)).toBeInTheDocument();
    expect(screen.getByText(/RUN CALIBRATION/i)).toBeInTheDocument();
  });

  // 2. Loading state tests
  it('displays computing state during active simulation', () => {
    useAppStore.setState({ simulateState: 'computing' });
    render(<AntennaArray />);
    expect(screen.getByText(/Computing physics/i)).toBeInTheDocument();
  });

  // 3. Error state tests
  it('displays simulation error message when simulation fails', () => {
    useAppStore.setState({
      simulateState: 'error',
      simulateError: 'Backend computation failed: singular matrix',
    });
    render(<App />);
    expect(
      screen.getByText(/Backend computation failed: singular matrix/i)
    ).toBeInTheDocument();
  });

  // 4. Result rendering tests
  it('renders 64 antenna elements and exposes deep element inspection', () => {
    useAppStore.setState({
      simulationResult: mockSimulationResponse,
      simulateState: 'success',
      selectedElement: 0,
    });
    render(<AntennaArray />);
    expect(screen.getByText(/LIVE 64-ELEMENT MICROSTRIP PATCH ARRAY/i)).toBeInTheDocument();

    render(<ElementInspector />);
    expect(screen.getByText(/ELEMENT #1/i)).toBeInTheDocument();
    expect(screen.getByText(/PHASE IMPAIRMENT/i)).toBeInTheDocument();
    expect(screen.getByText(/GAIN IMPAIRMENT/i)).toBeInTheDocument();
    expect(screen.getAllByText(/COMPENSATION WEIGHTS/i)[0]).toBeInTheDocument();
  });

  it('renders live metrics in MetricsPanel', () => {
    useAppStore.setState({
      simulationResult: mockSimulationResponse,
      simulateState: 'success',
    });
    render(<MetricsPanel />);
    expect(screen.getByText('Beam Correlation')).toBeInTheDocument();
    expect(screen.getByText('0.9980')).toBeInTheDocument();
    expect(screen.getByText('Analytical NMSE')).toBeInTheDocument();
    expect(screen.getByText('-14.45 dB')).toBeInTheDocument();
  });

  it('renders neutral side-by-side CalibrationPanel', () => {
    useAppStore.setState({
      simulationResult: mockSimulationResponse,
      simulateState: 'success',
    });
    render(<CalibrationPanel />);
    expect(screen.getByText(/Calibration Comparison/i)).toBeInTheDocument();
    expect(screen.getByText(/Both methods are shown without ranking/i)).toBeInTheDocument();
    expect(screen.getByText('-14.450')).toBeInTheDocument(); // Analytical NMSE
    expect(screen.getByText('-7.750')).toBeInTheDocument(); // PINN NMSE
  });

  // 5. API connectivity integration
  it('connects to backend simulation API when triggered', async () => {
    const postSpy = vi.spyOn(apiClient, 'postSimulation').mockResolvedValue(mockSimulationResponse);

    render(<ControlPanel />);
    const runBtn = screen.getByText(/RUN CALIBRATION/i);
    fireEvent.click(runBtn);

    await waitFor(() => {
      expect(postSpy).toHaveBeenCalled();
    });
  });

  // 6. Super Mode & Calibration Flow tests
  it('toggles Super Mode via SideNavigation control station', () => {
    render(<SideNavigation />);
    const superBtn = screen.getByLabelText(/Toggle Super Presentation Mode/i);
    fireEvent.click(superBtn);
    expect(useAppStore.getState().superMode).toBe(true);
  });

  // 7. Left Side Rail Navigation tests
  it('renders Left Side Rail navigation with all numbered modules', () => {
    render(<SideNavigation />);
    expect(screen.getByText('01')).toBeInTheDocument();
    expect(screen.getByText('Simulation')).toBeInTheDocument();
    expect(screen.getByText('02')).toBeInTheDocument();
    expect(screen.getByText('Calibration Explorer')).toBeInTheDocument();
    expect(screen.getByText('03')).toBeInTheDocument();
    expect(screen.getByText('Beam Lab')).toBeInTheDocument();
    expect(screen.getByText('04')).toBeInTheDocument();
    expect(screen.getByText('Robustness Lab')).toBeInTheDocument();
    expect(screen.getByText('05')).toBeInTheDocument();
    expect(screen.getByText('Research Status')).toBeInTheDocument();
  });

  // 8. PINN Neural Detector Flow tests
  it('renders PinnDetectorFlow architecture pipeline', () => {
    render(<PinnDetectorFlow />);
    expect(screen.getByText(/THE PINN AS AN RF DETECTOR/i)).toBeInTheDocument();
    expect(screen.getByText(/83,904 PARAMETERS/i)).toBeInTheDocument();
    expect(screen.getByText(/Observed Port Currents/i)).toBeInTheDocument();
  });
});


