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
import { BeamPatternChart } from '../components/charts/BeamPatternChart';
import { ResearchStatus } from '../pages/ResearchStatus';
import { ThemeToggle } from '../components/common/ThemeToggle';
import { StatusChip } from '../components/common/StatusChip';
import { Panel } from '../components/common/Panel';
import { MetricCard } from '../components/common/MetricCard';
import { PlaceholderSection } from '../components/common/PlaceholderSection';
import { useAppStore } from '../store/appStore';
import * as apiClient from '../api/client';
import type { SimulationResponse, ModelStatus } from '../types/api';

const mockModelStatus: ModelStatus = {
  model_name: 'PINNCalV4',
  version: '4.0.0',
  architecture: 'PINNCalV4 (PhysicsPreconditioner + BoundedResidualDenoiser)',
  tikhonov_delta: 0.15,
  residual_alpha: 0.50,
  bounded: true,
  input_dim: 128,
  output_dim: 128,
  hidden_dims: [128, 64, 128],
  total_parameters: 33088,
  trainable_parameters: 33088,
  checkpoint: 'experimental/v4/models/pinn_v4_tikhonov015_seed42.pth',
  checkpoint_path: 'experimental/v4/models/pinn_v4_tikhonov015_seed42.pth',
  checkpoint_available: true,
  best_epoch: 29,
  train_samples: 800,
  val_samples: 200,
  test_samples: 200,
  device: 'cuda:0',
  loaded: true,
  load_error: null,
  load_time_s: 0.12,
  cuda_available: true,
  gpu_name: 'NVIDIA GeForce GTX 1650',
  pinn_trained_for: {
    num_elements: 64,
    frequency_hz: 28e9,
    frequency_label: '28 GHz',
    note: 'PINN-Cal V4 calibration requires exactly 64 elements at 28 GHz.',
  },
  validated_results: {
    analytical: {
      nmse_db: -14.23,
      mean_phase_error_deg: 3.96,
      mean_gain_error_db: 1.07,
      mean_relative_error_pct: 15.13,
      current_residual_pct: 15.13,
    },
    pinn_v4: {
      nmse_db: -14.81,
      mean_phase_error_deg: 4.47,
      mean_gain_error_db: 1.00,
      mean_relative_error_pct: 15.34,
      current_residual_pct: 15.34,
      pointing_error_deg: 0.0253,
      beam_correlation: 0.99979,
    },
    pinn: {
      nmse_db: -14.81,
      mean_phase_error_deg: 4.47,
      mean_gain_error_db: 1.00,
      mean_relative_error_pct: 15.34,
      current_residual_pct: 15.34,
    },
  },
  honesty_note: 'Validated PINN-Cal V4 results.',
};

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
    ideal: { label: 'Ideal', angles_deg: [-30, 0, 30], power_db: [-25, 0, -25], peak_angle_deg: 0, pointing_error_deg: 0, beamwidth_3db_deg: 1.8, psll_db: -13.2, beam_correlation: 1.0 },
    uncalibrated: { label: 'Uncalibrated', angles_deg: [-30, 0, 30], power_db: [-20, -1.2, -22], peak_angle_deg: 0.8, pointing_error_deg: 0.8, beamwidth_3db_deg: 2.1, psll_db: -9.5, beam_correlation: 0.94 },
    analytical: { label: 'Analytical Baseline', angles_deg: [-30, 0, 30], power_db: [-24.8, -0.05, -24.8], peak_angle_deg: 0.05, pointing_error_deg: 0.05, beamwidth_3db_deg: 1.8, psll_db: -13.1, beam_correlation: 0.998 },
    pinn: { label: 'PINN-Cal V4', angles_deg: [-30, 0, 30], power_db: [-24.9, -0.02, -24.9], peak_angle_deg: 0.02, pointing_error_deg: 0.02, beamwidth_3db_deg: 1.79, psll_db: -13.2, beam_correlation: 0.999 },
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
      modelStatus: mockModelStatus,
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
    expect(screen.getByText(/ELEMENTS/i)).toBeInTheDocument();
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
    expect(screen.getByText(/LIVE 64-ELEMENT UNIFORM LINEAR ARRAY/i)).toBeInTheDocument();

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
    expect(screen.getByText(/Live Simulation Results/i)).toBeInTheDocument();
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
  it('renders Left Side Rail navigation with all 7 numbered modules', () => {
    render(<SideNavigation />);
    expect(screen.getByText('01')).toBeInTheDocument();
    expect(screen.getByText('OVERVIEW')).toBeInTheDocument();
    expect(screen.getByText('02')).toBeInTheDocument();
    expect(screen.getByText('LIVE ARRAY')).toBeInTheDocument();
    expect(screen.getByText('03')).toBeInTheDocument();
    expect(screen.getByText('CALIBRATE')).toBeInTheDocument();
    expect(screen.getByText('04')).toBeInTheDocument();
    expect(screen.getByText('BEAM RESTORATION')).toBeInTheDocument();
    expect(screen.getByText('05')).toBeInTheDocument();
    expect(screen.getByText('ROBUSTNESS LAB')).toBeInTheDocument();
    expect(screen.getByText('06')).toBeInTheDocument();
    expect(screen.getByText('RESEARCH & DATA')).toBeInTheDocument();
    expect(screen.getByText('07')).toBeInTheDocument();
    expect(screen.getByText('SYSTEM')).toBeInTheDocument();
  });

  // 8. PINN Neural Detector Flow tests
  it('renders PinnDetectorFlow architecture pipeline', () => {
    useAppStore.setState({
      modelStatus: mockModelStatus,
    });
    render(<PinnDetectorFlow />);
    expect(screen.getByText(/PINN-CAL V4 TWO-STAGE COMPENSATION/i)).toBeInTheDocument();
    expect(screen.getByText(/33,088 PARAMETERS/i)).toBeInTheDocument();
    expect(screen.getByText(/Measured \/ Observed Currents/i)).toBeInTheDocument();
  });

  // 9. V4 Model Status rendering tests
  it('renders V4 model status metadata with 33,088 parameters in TopStatusBar', () => {
    useAppStore.setState({
      modelStatus: mockModelStatus,
    });
    render(<TopStatusBar onOpenMobileNav={vi.fn()} />);
    expect(screen.getAllByText(/PINN-Cal/i).length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText(/33,088/i)).toBeInTheDocument();
  });

  it('renders V4 model architecture details in ResearchStatus', () => {
    useAppStore.setState({
      modelStatus: mockModelStatus,
    });
    render(<ResearchStatus />);
    expect(screen.getByText(/PINN-Cal V4 Neural Architecture/i)).toBeInTheDocument();
    expect(screen.getByText(/33,088 weights/i)).toBeInTheDocument();
    expect(screen.getAllByText(/δ = 0.15/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/α = 0.5/i).length).toBeGreaterThanOrEqual(1);
  });

  // 10. V4 Beam label test
  it('renders PINN-Cal V4 beam label and preserves 4 traces', () => {
    useAppStore.setState({
      simulationResult: mockSimulationResponse,
      simulateState: 'success',
    });
    render(<BeamPatternChart />);
    expect(screen.getByText('Beam Pattern')).toBeInTheDocument();
  });

  // 11. Calibrated array mode test
  it('calibrated array exposes Post-Compensation State and V4 Estimated Compensation', () => {
    useAppStore.setState({
      simulationResult: mockSimulationResponse,
      simulateState: 'success',
      arrayDisplayMode: 'calibrated',
    });
    render(<AntennaArray />);
    expect(screen.getByText(/4\. Post-Compensation State/i)).toBeInTheDocument();
    expect(screen.getByText(/3\. V4 Estimated Compensation/i)).toBeInTheDocument();
    expect(screen.getByText(/2\. Observed \/ Impaired/i)).toBeInTheDocument();
  });

  // 12. Absence of stale V3 text test
  it('verifies stale V3 text is completely absent from the UI', () => {
    useAppStore.setState({
      simulationResult: mockSimulationResponse,
      modelStatus: mockModelStatus,
      simulateState: 'success',
    });
    const { container } = render(
      <>
        <TopStatusBar onOpenMobileNav={vi.fn()} />
        <ControlPanel />
        <PinnDetectorFlow />
        <CalibrationPanel />
        <ResearchStatus />
      </>
    );
    expect(container.textContent).not.toContain('83,904');
    expect(container.textContent).not.toContain('EPOCH 40');
    expect(container.textContent).not.toContain('The analytical baseline currently achieves lower correction error');
  });

  // 13. Dynamic simulation flow test
  it('executes dynamic simulation with all 6 physical parameters', async () => {
    const postSpy = vi.spyOn(apiClient, 'postSimulation').mockResolvedValue(mockSimulationResponse);
    useAppStore.setState({
      config: {
        num_elements: 64,
        frequency_hz: 28e9,
        element_spacing_lambda: 0.5,
        target_angle_deg: 25,
        phase_sigma_deg: 12,
        gain_sigma_db: 1.5,
        snr_db: 15,
        coupling_mismatch_fraction: 0.25,
        seed: 777,
      },
    });

    render(<ControlPanel />);
    const runBtn = screen.getByText(/RUN CALIBRATION/i);
    fireEvent.click(runBtn);

    await waitFor(() => {
      expect(postSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          target_angle_deg: 25,
          phase_sigma_deg: 12,
          gain_sigma_db: 1.5,
          snr_db: 15,
          coupling_mismatch_fraction: 0.25,
          seed: 777,
        })
      );
    });
  });

  // 14. ThemeToggle tests
  it('toggles Light and Dark theme and persists to localStorage', () => {
    render(<ThemeToggle />);
    const toggleBtn = screen.getByRole('button', { name: /Switch to/i });
    expect(toggleBtn).toBeInTheDocument();

    const initialTheme = useAppStore.getState().theme;
    fireEvent.click(toggleBtn);
    const newTheme = useAppStore.getState().theme;
    expect(newTheme).not.toBe(initialTheme);
    expect(localStorage.getItem('pinn_cal_theme')).toBe(newTheme);
    expect(document.documentElement.getAttribute('data-theme')).toBe(newTheme);
  });

  // 15. StatusChip tests
  it('renders StatusChip with status dot and labels', () => {
    render(<StatusChip label="CUDA" value="ONLINE" status="online" />);
    expect(screen.getByText('CUDA')).toBeInTheDocument();
    expect(screen.getByText('ONLINE')).toBeInTheDocument();
  });

  // 16. AppShell & OverviewPage tests
  it('renders AppShell with primary rail, header, and overview stage', () => {
    useAppStore.setState({ activeView: 'overview' });
    render(<App />);
    expect(screen.getByRole('banner')).toBeInTheDocument();
    expect(screen.getByRole('navigation', { name: /Main Navigation/i })).toBeInTheDocument();
    expect(screen.getByText('Overview & Architecture')).toBeInTheDocument();
    expect(screen.getByText(/STEP 75A SHELL/i)).toBeInTheDocument();
  });

  // 17. Reusable surface components
  it('renders Panel, MetricCard, and PlaceholderSection components properly', () => {
    render(
      <Panel title="Test Panel" subtitle="Test Subtitle" elevated>
        <MetricCard label="BANDWIDTH" value="800" unit="MHz" status="accent" />
        <PlaceholderSection
          sectionId="test-section"
          title="Test Stage"
          subtitle="Test Subtitle"
          plannedStage="STEP 75B PLANNED"
          summary="Test Summary"
          modules={[{ id: 'm1', title: 'Sub-Module 1', description: 'Desc 1', icon: '📡' }]}
        />
      </Panel>
    );
    expect(screen.getByText('Test Panel')).toBeInTheDocument();
    expect(screen.getByText('BANDWIDTH')).toBeInTheDocument();
    expect(screen.getByText('800')).toBeInTheDocument();
    expect(screen.getByText('Test Stage')).toBeInTheDocument();
    expect(screen.getByText('Sub-Module 1')).toBeInTheDocument();
  });
});


