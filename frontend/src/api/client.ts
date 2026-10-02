/**
 * PINN-Cal Frontend — API Client
 *
 * Typed wrappers around backend endpoints.
 * Uses axios for HTTP calls.
 */

import axios from 'axios';
import type {
  SimulationRequest,
  SimulationResponse,
  SystemStatus,
  HealthStatus,
  ModelStatus,
  RobustnessRequest,
  RobustnessResponse,
} from '../types/api';

const BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 120_000, // robustness sweeps can be slow
  headers: { 'Content-Type': 'application/json' },
});

// ---------------------------------------------------------------------------
// Health / System
// ---------------------------------------------------------------------------

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await api.get<HealthStatus>('/api/health');
  return res.data;
}

export async function fetchSystemStatus(): Promise<SystemStatus> {
  const res = await api.get<SystemStatus>('/api/system-status');
  return res.data;
}

export async function fetchModelStatus(): Promise<ModelStatus> {
  const res = await api.get<ModelStatus>('/api/model-status');
  return res.data;
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

export async function postSimulation(
  request: SimulationRequest,
): Promise<SimulationResponse> {
  const res = await api.post<SimulationResponse>('/api/simulation', request);
  return res.data;
}

// ---------------------------------------------------------------------------
// Robustness
// ---------------------------------------------------------------------------

export async function postRobustness(
  request: RobustnessRequest,
): Promise<RobustnessResponse> {
  const res = await api.post<RobustnessResponse>('/api/robustness', request);
  return res.data;
}
