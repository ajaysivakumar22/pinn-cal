import React from 'react';
import { useAppStore } from '../../store/appStore';
import { ConceptTrigger } from '../common/ConceptExplainer';

const FREQ_OPTIONS = [
  { label: '28 GHz (Validated mmWave)', value: 28e9 },
  { label: '39 GHz (Experimental)', value: 39e9 },
  { label: '60 GHz (Experimental)', value: 60e9 },
];

export const ControlPanel: React.FC = () => {
  const { config, setConfig, resetConfig, runSimulation, simulateState, superMode, modelStatus } =
    useAppStore();

  const isComputing = simulateState === 'computing';
  const pinnCompatible = config.num_elements === 64 && Math.abs(config.frequency_hz - 28e9) < 1e6;

  const applyPreset = (preset: Partial<typeof config>) => {
    setConfig(preset);
  };

  return (
    <div className="experiment-controls-bench">
      {/* Quick Scientific Presets Strip */}
      <div className="presets-bar">
        <span className="presets-label">QUICK TEST PRESETS:</span>
        <div className="presets-buttons">
          <button
            type="button"
            className={`btn-preset ${config.target_angle_deg === 20 && config.snr_db === 20 ? 'active' : ''}`}
            onClick={() => applyPreset({ target_angle_deg: 20, snr_db: 20, phase_sigma_deg: 20, gain_sigma_db: 2 })}
          >
            ★ Standard Demo (+20° Steered, 20 dB SNR)
          </button>
          <button
            type="button"
            className={`btn-preset ${config.target_angle_deg === 0 ? 'active' : ''}`}
            onClick={() => applyPreset({ target_angle_deg: 0, snr_db: 20 })}
          >
            Broadside (0° Heading)
          </button>
          <button
            type="button"
            className={`btn-preset ${config.snr_db === 10 ? 'active' : ''}`}
            onClick={() => applyPreset({ snr_db: 10, coupling_mismatch_fraction: 0.3 })}
          >
            High Noise Stress (10 dB SNR, ±30% ΔZ)
          </button>
          <button
            type="button"
            className={`btn-preset ${config.phase_sigma_deg === 35 ? 'active' : ''}`}
            onClick={() => applyPreset({ phase_sigma_deg: 35, gain_sigma_db: 4 })}
          >
            Severe Hardware Skew (σ_ϕ = 35°)
          </button>
        </div>
      </div>

      {/* 4-Domain Tactile Control Grid */}
      <div className="setup-grid-cards">
        {/* DOMAIN 1: TARGET STEERING */}
        <div className="setup-card target-domain">
          <div className="setup-card-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="domain-icon">🎯</span>
              <span className="setup-card-title">TARGET STEERING</span>
            </div>
            <ConceptTrigger conceptKey="beamforming" label="Physics" />
          </div>
          <div className="setup-card-body">
            <TactileSlider
              id="target-angle"
              label="Steering Heading (θ)"
              hint="Commanded main beam radiation angle relative to broadside."
              min={-60}
              max={60}
              step={1}
              value={config.target_angle_deg}
              format={(v) => `${v > 0 ? '+' : ''}${v.toFixed(0)}°`}
              onChange={(v) => setConfig({ target_angle_deg: v })}
            />
          </div>
        </div>

        {/* DOMAIN 2: ARRAY GEOMETRY */}
        <div className="setup-card array-domain">
          <div className="setup-card-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="domain-icon">📡</span>
              <span className="setup-card-title">ARRAY APERTURE</span>
            </div>
            <ConceptTrigger conceptKey="mutual_coupling" label="Coupling" />
          </div>
          <div className="setup-card-body">
            <div className="form-group" style={{ marginBottom: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label className="form-label" htmlFor="num-elements">
                  Array Elements (N)
                </label>
                <span className="tactile-live-val">{config.num_elements}E</span>
              </div>
              <select
                id="num-elements"
                aria-label="Elements"
                className="form-select tactile-select"
                value={config.num_elements}
                onChange={(e) => setConfig({ num_elements: Number(e.target.value) })}
              >
                {[8, 16, 32, 64, 128].map((n) => (
                  <option key={n} value={n}>
                    {n} elements {n === 64 ? '· (Validated PINN Checkpoint)' : ''}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group" style={{ marginBottom: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label className="form-label" htmlFor="frequency">
                  Operating Frequency
                </label>
                <span className="tactile-live-val">{(config.frequency_hz / 1e9).toFixed(0)} GHz</span>
              </div>
              <select
                id="frequency"
                aria-label="Frequency"
                className="form-select tactile-select"
                value={config.frequency_hz}
                onChange={(e) => setConfig({ frequency_hz: Number(e.target.value) })}
              >
                {FREQ_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            <TactileSlider
              id="element-spacing"
              label="Element Spacing (d)"
              hint="Inter-element spacing (0.50 λ = 5.35 mm at 28 GHz)."
              min={0.25}
              max={1.0}
              step={0.05}
              value={config.element_spacing_lambda}
              format={(v) => `${v.toFixed(2)}λ`}
              onChange={(v) => setConfig({ element_spacing_lambda: v })}
            />
          </div>
        </div>

        {/* DOMAIN 3: HARDWARE IMPAIRMENTS */}
        <div className="setup-card hw-domain">
          <div className="setup-card-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="domain-icon">⚡</span>
              <span className="setup-card-title">HARDWARE IMPAIRMENTS</span>
            </div>
            <ConceptTrigger conceptKey="phase_error" label="Errors" />
          </div>
          <div className="setup-card-body">
            <TactileSlider
              id="phase-sigma"
              label="Phase Variance σ"
              hint="Random phase deviation across individual transceivers."
              min={0}
              max={45}
              step={1}
              value={config.phase_sigma_deg}
              format={(v) => `${v.toFixed(0)}°`}
              onChange={(v) => setConfig({ phase_sigma_deg: v })}
              highlightColor="#ef4444"
            />
            <TactileSlider
              id="gain-sigma"
              label="Gain Variance σ"
              hint="Random power amplifier output variations."
              min={0}
              max={6}
              step={0.25}
              value={config.gain_sigma_db}
              format={(v) => `${v.toFixed(2)} dB`}
              onChange={(v) => setConfig({ gain_sigma_db: v })}
              highlightColor="#f59e0b"
            />
          </div>
        </div>

        {/* DOMAIN 4: MEASUREMENT & COUPLING */}
        <div className="setup-card env-domain">
          <div className="setup-card-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="domain-icon">📊</span>
              <span className="setup-card-title">MEASUREMENT & NOISE</span>
            </div>
            <ConceptTrigger conceptKey="snr" label="Noise" />
          </div>
          <div className="setup-card-body">
            <TactileSlider
              id="snr"
              label="Receiver SNR"
              hint="Thermal receiver noise relative to coupled antenna currents."
              min={-10}
              max={60}
              step={1}
              value={config.snr_db}
              format={(v) => `${v.toFixed(0)} dB`}
              onChange={(v) => setConfig({ snr_db: v })}
            />
            <TactileSlider
              id="coupling-mismatch"
              label="Coupling Mismatch (ΔZ)"
              hint="Hidden mutual impedance deviation from nominal model."
              min={0}
              max={0.5}
              step={0.01}
              value={config.coupling_mismatch_fraction}
              format={(v) => `±${(v * 100).toFixed(0)}%`}
              onChange={(v) => setConfig({ coupling_mismatch_fraction: v })}
            />
            <TactileSlider
              id="seed"
              label="Monte Carlo Seed"
              hint="Reproducibility seed for noise and hardware realization."
              min={0}
              max={9999}
              step={1}
              value={config.seed}
              format={(v) => `${v.toFixed(0)}`}
              onChange={(v) => setConfig({ seed: v })}
            />
          </div>
        </div>
      </div>

      {/* Compatibility Status Notice */}
      {!pinnCompatible ? (
        <div className="pinn-compat-alert warning" role="alert">
          ℹ <strong>PINN Compatibility Guard:</strong> Checkpoint was validated specifically for 64 elements at 28 GHz.
          Analytical calibration and physical beam simulation will execute normally.
        </div>
      ) : (
        <div className="pinn-compat-alert success">
          <span style={{ fontWeight: 800 }}>✓</span>
          <span>PINN-Cal V4 Active: 64 Elements @ 28 GHz · {modelStatus?.trainable_parameters?.toLocaleString() ?? '33,088'} Parameters (δ=0.15, α=0.50)</span>
        </div>
      )}

      {/* Action Command Row */}
      <div className="controls-action-row">
        <button
          className="btn-console-run"
          onClick={runSimulation}
          disabled={isComputing}
          id="btn-run-simulation"
          aria-busy={isComputing}
        >
          <span className="run-icon">{isComputing ? '⏳' : '▶'}</span>
          <span>{isComputing ? 'COMPUTING PHYSICS & CALIBRATING...' : 'RUN CALIBRATION BENCH'}</span>
          {superMode && <span>★</span>}
        </button>

        <button
          className="btn-console-reset"
          onClick={resetConfig}
          disabled={isComputing}
          id="btn-reset"
        >
          Reset Standard Parameters
        </button>
      </div>
    </div>
  );
};

const TactileSlider: React.FC<{
  id: string;
  label: string;
  hint?: string;
  min: number;
  max: number;
  step: number;
  value: number;
  format: (v: number) => string;
  onChange: (v: number) => void;
  highlightColor?: string;
}> = ({ id, label, hint, min, max, step, value, format, onChange, highlightColor }) => (
  <div className="tactile-slider-group">
    <div className="slider-label-row">
      <label className="tactile-label" htmlFor={id}>
        {label}
      </label>
      <span
        className="tactile-live-val"
        style={{ color: highlightColor ?? '#0f172a' }}
      >
        {format(value)}
      </span>
    </div>
    {hint && <div className="tactile-hint">{hint}</div>}
    <input
      id={id}
      aria-label={label}
      type="range"
      className="tactile-range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={value}
      aria-valuetext={format(value)}
    />
  </div>
);
