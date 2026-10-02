/**
 * PINN-Cal Frontend — Global Application Store
 *
 * Uses Zustand for clean, centralized state management.
 * All API calls flow through actions in this store.
 */

import { create } from 'zustand';
import {
  fetchHealth,
  fetchSystemStatus,
  fetchModelStatus,
  postSimulation,
  postRobustness,
} from '../api/client';
import type {
  SimulationRequest,
  SimulationResponse,
  SystemStatus,
  HealthStatus,
  ModelStatus,
  RobustnessRequest,
  RobustnessResponse,
  ComputeState,
  ArrayDisplayMode,
  ActiveView,
} from '../types/api';

// ---------------------------------------------------------------------------
// Default simulation configuration
// ---------------------------------------------------------------------------

export const DEFAULT_CONFIG: SimulationRequest = {
  num_elements: 64,
  frequency_hz: 28e9,
  element_spacing_lambda: 0.5,
  target_angle_deg: 20, // Nonzero target angle for immediate live calibration showcase
  phase_sigma_deg: 20,
  gain_sigma_db: 2,
  snr_db: 20,
  coupling_mismatch_fraction: 0.2,
  seed: 42,
};

// ---------------------------------------------------------------------------
// Store interface
// ---------------------------------------------------------------------------

export type WorkflowStage =
  | 'idle'
  | 'observe'
  | 'diagnose'
  | 'calibrate'
  | 'restore'
  | 'verify'
  | 'completed';

export type CalibrationExplorerMetric =
  | 'phase_error'
  | 'gain_error'
  | 'pinn_phase_est'
  | 'pinn_gain_est'
  | 'phase_residual'
  | 'gain_residual'
  | 'correction_magnitude';

interface AppStore {
  // Navigation
  activeView: ActiveView;
  setActiveView: (view: ActiveView) => void;

  // Super Mode & Franky
  superMode: boolean;
  toggleSuperMode: () => void;
  frankyMessage: string | null;
  setFrankyMessage: (msg: string | null) => void;
  showStarCelebration: boolean;
  setShowStarCelebration: (show: boolean) => void;

  // Simulation config (user-controlled)
  config: SimulationRequest;
  setConfig: (partial: Partial<SimulationRequest>) => void;
  resetConfig: () => void;

  // Simulation result
  simulationResult: SimulationResponse | null;
  simulateState: ComputeState;
  simulateError: string | null;
  workflowStage: WorkflowStage;
  runSimulation: () => Promise<void>;

  // Array visualization mode
  arrayDisplayMode: ArrayDisplayMode;
  setArrayDisplayMode: (mode: ArrayDisplayMode) => void;

  // Selected element for detail view
  selectedElement: number | null;
  setSelectedElement: (idx: number | null) => void;

  // Calibration explorer filter
  calibrationMetric: CalibrationExplorerMetric;
  setCalibrationMetric: (metric: CalibrationExplorerMetric) => void;

  // Educational Concept & How It Works
  activeConcept: string | null;
  setActiveConcept: (concept: string | null) => void;
  isHowItWorksOpen: boolean;
  setIsHowItWorksOpen: (open: boolean) => void;

  // System status
  health: HealthStatus | null;
  systemStatus: SystemStatus | null;
  modelStatus: ModelStatus | null;
  systemLoaded: boolean;
  loadSystemStatus: () => Promise<void>;

  // Robustness
  robustnessResult: RobustnessResponse | null;
  robustnessState: ComputeState;
  robustnessError: string | null;
  runRobustness: (req: RobustnessRequest) => Promise<void>;
}

// ---------------------------------------------------------------------------
// Store implementation
// ---------------------------------------------------------------------------

export const useAppStore = create<AppStore>((set, get) => ({
  // Navigation
  activeView: 'simulation',
  setActiveView: (view) => set({ activeView: view }),

  // Super Mode & Franky
  superMode: false,
  toggleSuperMode: () =>
    set((state) => {
      const next = !state.superMode;
      return {
        superMode: next,
        frankyMessage: next
          ? "SUPER MODE ACTIVE! Franky's online in the RF calibration lab!"
          : null,
      };
    }),
  frankyMessage: null,
  setFrankyMessage: (msg) => set({ frankyMessage: msg }),
  showStarCelebration: false,
  setShowStarCelebration: (show) => set({ showStarCelebration: show }),

  // Config
  config: { ...DEFAULT_CONFIG },
  setConfig: (partial) =>
    set((state) => ({ config: { ...state.config, ...partial } })),
  resetConfig: () => set({ config: { ...DEFAULT_CONFIG } }),

  // Simulation
  simulationResult: null,
  simulateState: 'idle',
  simulateError: null,
  workflowStage: 'idle',
  runSimulation: async () => {
    const { config, superMode } = get();
    set({
      simulateState: 'computing',
      simulateError: null,
      workflowStage: 'observe',
      showStarCelebration: false,
      frankyMessage: superMode
        ? "01 OBSERVE: Reading the coupled antenna currents from 64 elements!"
        : null,
    });

    try {
      // 01 OBSERVE -> API call
      const result = await postSimulation(config);

      // 02 DIAGNOSE
      set({
        workflowStage: 'diagnose',
        frankyMessage: get().superMode
          ? "02 DIAGNOSE: Found element-level phase and gain variations!"
          : null,
      });
      await new Promise((r) => setTimeout(r, 120));

      // 03 CALIBRATE
      set({
        workflowStage: 'calibrate',
        frankyMessage: get().superMode
          ? "03 CALIBRATE: Computing the complex compensation weights!"
          : null,
      });
      await new Promise((r) => setTimeout(r, 120));

      // 04 RESTORE
      set({
        workflowStage: 'restore',
        frankyMessage: get().superMode
          ? "04 RESTORE: Applying inverse compensation to array excitation!"
          : null,
      });
      await new Promise((r) => setTimeout(r, 120));

      // 05 VERIFY
      set({
        workflowStage: 'verify',
        frankyMessage: get().superMode
          ? "05 VERIFY: SUPER! Let's verify the restored beam!"
          : null,
      });
      await new Promise((r) => setTimeout(r, 120));

      set({
        simulationResult: result,
        simulateState: 'success',
        workflowStage: 'completed',
        showStarCelebration: get().superMode,
        frankyMessage: get().superMode
          ? "★ SUPER!! Calibration complete! 64 channels compensated and beam restored!"
          : null,
        // Select element 16 or 0 by default for instant inspection if none selected
        selectedElement: get().selectedElement ?? 16,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Simulation failed';
      set({
        simulateState: 'error',
        simulateError: msg,
        workflowStage: 'idle',
        frankyMessage: get().superMode ? `Uh-oh! Simulation error: ${msg}` : null,
      });
    }
  },

  // Array visualization
  arrayDisplayMode: 'calibrated',
  setArrayDisplayMode: (mode) => set({ arrayDisplayMode: mode }),
  selectedElement: 16,
  setSelectedElement: (idx) => set({ selectedElement: idx }),

  // Calibration Explorer
  calibrationMetric: 'phase_error',
  setCalibrationMetric: (metric) => set({ calibrationMetric: metric }),

  // Educational Concept & How It Works
  activeConcept: null,
  setActiveConcept: (concept) => set({ activeConcept: concept }),
  isHowItWorksOpen: false,
  setIsHowItWorksOpen: (open) => set({ isHowItWorksOpen: open }),

  // System status
  health: null,
  systemStatus: null,
  modelStatus: null,
  systemLoaded: false,
  loadSystemStatus: async () => {
    try {
      const [health, system, model] = await Promise.all([
        fetchHealth(),
        fetchSystemStatus(),
        fetchModelStatus(),
      ]);
      set({ health, systemStatus: system, modelStatus: model, systemLoaded: true });
    } catch {
      set({ systemLoaded: false });
    }
  },

  // Robustness
  robustnessResult: null,
  robustnessState: 'idle',
  robustnessError: null,
  runRobustness: async (req) => {
    set({ robustnessState: 'computing', robustnessError: null });
    try {
      const result = await postRobustness(req);
      set({ robustnessResult: result, robustnessState: 'success' });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Robustness sweep failed';
      set({ robustnessState: 'error', robustnessError: msg });
    }
  },
}));
