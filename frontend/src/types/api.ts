/**
 * PINN-Cal Frontend — API Type Definitions
 *
 * These interfaces mirror the Pydantic schemas in backend/app/schemas/.
 * Keep them in sync with the backend response shapes.
 */

// ---------------------------------------------------------------------------
// Simulation Request
// ---------------------------------------------------------------------------

export interface SimulationRequest {
  num_elements: number;
  frequency_hz: number;
  element_spacing_lambda: number;
  target_angle_deg: number;
  phase_sigma_deg: number;
  gain_sigma_db: number;
  snr_db: number;
  coupling_mismatch_fraction: number;
  seed: number;
}

// ---------------------------------------------------------------------------
// Per-element state
// ---------------------------------------------------------------------------

export interface ElementState {
  element_index: number;
  // Ideal excitation
  ideal_real: number;
  ideal_imag: number;
  ideal_magnitude: number;
  ideal_phase_deg: number;
  // Hardware
  hardware_real: number;
  hardware_imag: number;
  hardware_magnitude: number;
  hardware_phase_deg: number;
  gain_error_db: number;
  phase_error_deg: number;
  true_gain_error_db?: number;
  true_phase_error_deg?: number;
  // Observed
  observed_real: number;
  observed_imag: number;
  // Analytical correction
  analytical_correction_real: number;
  analytical_correction_imag: number;
  analytical_correction_magnitude: number;
  analytical_correction_phase_deg: number;
  analytical_estimated_phase_error_deg?: number;
  analytical_estimated_gain_error_db?: number;
  analytical_phase_residual_deg?: number;
  analytical_gain_residual_db?: number;
  // PINN correction (null if unavailable)
  pinn_correction_real: number | null;
  pinn_correction_imag: number | null;
  pinn_correction_magnitude: number | null;
  pinn_correction_phase_deg: number | null;
  // PINN estimated RF impairment & residuals
  pinn_estimated_phase_error_deg?: number | null;
  pinn_estimated_gain_error_db?: number | null;
  phase_error_residual_deg?: number | null;
  gain_error_residual_db?: number | null;
  compensation_phase_deg?: number | null;
  compensation_gain_db?: number | null;
}

// ---------------------------------------------------------------------------
// Beam pattern trace
// ---------------------------------------------------------------------------

export interface BeamTrace {
  label: string;
  angles_deg: number[];
  power_db: number[];
  peak_angle_deg: number;
  pointing_error_deg: number;
  beamwidth_3db_deg: number | null;
  psll_db: number | null;
  beam_correlation: number | null;
}

export interface BeamPatternResult {
  ideal: BeamTrace;
  uncalibrated: BeamTrace;
  analytical: BeamTrace;
  pinn: BeamTrace | null;
}

// ---------------------------------------------------------------------------
// Calibration metrics
// ---------------------------------------------------------------------------

export interface CalibrationMetrics {
  nmse_db: number;
  mean_phase_error_deg: number;
  mean_gain_error_db: number;
  mean_relative_error_pct: number;
  current_residual_pct: number | null;
}

export interface CalibrationComparison {
  analytical: CalibrationMetrics;
  pinn: CalibrationMetrics | null;
  pinn_available: boolean;
}

// ---------------------------------------------------------------------------
// Summary metrics (for right panel)
// ---------------------------------------------------------------------------

export interface SummaryMetrics {
  target_angle_deg: number;
  beam_peak_angle_deg: number;
  pointing_error_deg: number;
  beam_correlation: number | null;
  beamwidth_3db_deg: number | null;
  psll_db: number | null;
  analytical_nmse_db: number;
  pinn_nmse_db: number | null;
}

// ---------------------------------------------------------------------------
// Simulation Response
// ---------------------------------------------------------------------------

export interface SimulationResponse {
  request: SimulationRequest;
  pinn_available: boolean;
  pinn_compatibility_message: string;
  elements: ElementState[];
  beam: BeamPatternResult;
  calibration: CalibrationComparison;
  metrics: SummaryMetrics;
}

// ---------------------------------------------------------------------------
// System Status
// ---------------------------------------------------------------------------

export interface SystemStatus {
  python_version: string;
  pytorch_version: string;
  cuda_available: boolean;
  gpu_name: string | null;
  cuda_device_count: number;
  platform: string;
  scientific_core_available: boolean;
  scientific_core_error: string | null;
  backend_uptime_s: number;
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export interface HealthStatus {
  status: string;
  timestamp: number;
  uptime_s: number;
  scientific_core_available: boolean;
  scientific_core_error: string | null;
  pinn_model_loaded: boolean;
  pinn_model_error: string | null;
}

// ---------------------------------------------------------------------------
// Model Status
// ---------------------------------------------------------------------------

export interface ModelStatus {
  model_name: string;
  version: string;
  architecture: string;
  tikhonov_delta: number;
  residual_alpha: number;
  bounded: boolean;
  input_dim: number;
  output_dim: number;
  hidden_dims: number[];
  total_parameters: number;
  trainable_parameters: number;
  checkpoint: string;
  checkpoint_path: string;
  checkpoint_available: boolean;
  best_epoch: number;
  train_samples: number;
  val_samples: number;
  test_samples: number;
  device: string;
  loaded: boolean;
  load_error: string | null;
  load_time_s: number;
  cuda_available: boolean;
  gpu_name: string | null;
  pinn_trained_for: {
    num_elements: number;
    frequency_hz: number;
    frequency_label: string;
    note: string;
  };
  validated_results: {
    analytical: CalibrationMetrics & { mean_relative_error_pct: number };
    pinn_v3?: CalibrationMetrics & { mean_relative_error_pct: number };
    pinn_v4?: CalibrationMetrics & {
      mean_relative_error_pct: number;
      pointing_error_deg?: number;
      beam_correlation?: number;
    };
    pinn: CalibrationMetrics & { mean_relative_error_pct: number };
    multiseed?: {
      nmse_mean_db: number;
      nmse_std_db: number;
      phase_mean_deg: number;
      phase_std_deg: number;
      gain_mean_db: number;
      gain_std_db: number;
      relative_mean_pct: number;
      relative_std_pct: number;
    };
  };
  honesty_note: string;
}

// ---------------------------------------------------------------------------
// Robustness
// ---------------------------------------------------------------------------

export type SweepType = 'snr' | 'coupling' | 'steering';

export interface RobustnessRequest {
  sweep_type: SweepType;
  levels?: number[];
  num_samples: number;
  seed: number;
}

export interface RobustnessPoint {
  level: number;
  analytical_nmse_db: number;
  pinn_nmse_db: number | null;
  analytical_phase_error_deg: number;
  pinn_phase_error_deg: number | null;
  analytical_gain_error_db: number;
  pinn_gain_error_db: number | null;
  analytical_relative_error_pct: number;
  pinn_relative_error_pct: number | null;
}

export interface RobustnessResponse {
  sweep_type: string;
  levels: number[];
  points: RobustnessPoint[];
  pinn_available: boolean;
  note: string;
}

// ---------------------------------------------------------------------------
// App state types
// ---------------------------------------------------------------------------

export type ComputeState = 'idle' | 'computing' | 'success' | 'error';

export type ArrayDisplayMode =
  | 'ideal'
  | 'hardware'
  | 'pinn_diagnosis'
  | 'calibrated'
  | 'analytical'
  | 'pinn';

export type ActiveView =
  | 'overview'
  | 'simulation'
  | 'calibration'
  | 'beam'
  | 'robustness'
  | 'research'
  | 'dashboard'
  | 'system';
