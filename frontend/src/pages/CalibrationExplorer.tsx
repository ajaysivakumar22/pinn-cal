import React, { useMemo } from 'react';
import { AsyncPlot as Plot } from '../components/common/AsyncPlot';
import { useAppStore, type CalibrationExplorerMetric } from '../store/appStore';
import type { ElementState, SimulationRequest } from '../types/api';
import { ConceptTrigger } from '../components/common/ConceptExplainer';

const METRIC_OPTIONS: { id: CalibrationExplorerMetric; label: string; unit: string; description: string }[] = [
  { id: 'phase_error', label: 'Phase Error (Ground Truth)', unit: '°', description: 'Simulated transceiver phase error.' },
  { id: 'gain_error', label: 'Gain Error (Ground Truth)', unit: 'dB', description: 'Simulated transceiver gain error.' },
  { id: 'pinn_phase_est', label: 'PINN Phase Estimate', unit: '°', description: 'Model-estimated phase error (-angle(C)).' },
  { id: 'pinn_gain_est', label: 'PINN Gain Estimate', unit: 'dB', description: 'Model-estimated gain error (-20*log10(|C|)).' },
  { id: 'phase_residual', label: 'Phase Residual (Post-Comp)', unit: '°', description: 'Residual phase deviation after compensation.' },
  { id: 'gain_residual', label: 'Gain Residual (Post-Comp)', unit: 'dB', description: 'Residual gain deviation after compensation.' },
  { id: 'correction_magnitude', label: 'Correction Magnitude (|C|)', unit: '', description: 'Norm of complex compensation weight.' },
];

function getElementMetricValue(elem: ElementState, metric: CalibrationExplorerMetric): number {
  switch (metric) {
    case 'phase_error':
      return elem.true_phase_error_deg ?? elem.phase_error_deg;
    case 'gain_error':
      return elem.true_gain_error_db ?? elem.gain_error_db;
    case 'pinn_phase_est':
      return elem.pinn_estimated_phase_error_deg ?? (elem.pinn_correction_phase_deg !== null ? -elem.pinn_correction_phase_deg : 0);
    case 'pinn_gain_est':
      return elem.pinn_estimated_gain_error_db ?? (elem.pinn_correction_magnitude !== null ? -20 * Math.log10(Math.max(elem.pinn_correction_magnitude, 1e-6)) : 0);
    case 'phase_residual': {
      const gt = elem.true_phase_error_deg ?? elem.phase_error_deg;
      const est = elem.pinn_estimated_phase_error_deg ?? (elem.pinn_correction_phase_deg !== null ? -elem.pinn_correction_phase_deg : 0);
      return ((((gt - est + 180) % 360) + 360) % 360) - 180;
    }
    case 'gain_residual': {
      const gt = elem.true_gain_error_db ?? elem.gain_error_db;
      const est = elem.pinn_estimated_gain_error_db ?? (elem.pinn_correction_magnitude !== null ? -20 * Math.log10(Math.max(elem.pinn_correction_magnitude, 1e-6)) : 0);
      return gt - est;
    }
    case 'correction_magnitude':
      return elem.pinn_correction_magnitude ?? elem.analytical_correction_magnitude;
  }
}

export const CalibrationExplorer: React.FC = () => {
  const {
    config,
    setConfig,
    resetConfig,
    runSimulation,
    simulateState,
    simulateError,
    simulationResult,
    calibrationMetric,
    setCalibrationMetric,
    selectedElement,
    setSelectedElement,
    setActiveView,
    superMode,
  } = useAppStore();

  const isComputing = simulateState === 'computing';
  const hasError = simulateState === 'error' || !!simulateError;
  const hasResult = !!simulationResult;

  // Preset Configurations
  const applyPreset = (preset: Partial<SimulationRequest>) => {
    setConfig(preset);
  };

  const isDefaultPreset =
    config.target_angle_deg === 20 &&
    config.snr_db === 20 &&
    config.phase_sigma_deg === 20 &&
    config.gain_sigma_db === 2 &&
    Math.abs(config.coupling_mismatch_fraction - 0.2) < 0.01 &&
    config.seed === 42;

  // Selected element resolution (defaults to Element 16)
  const elements = simulationResult?.elements;
  const numElements = elements?.length ?? 64;
  const activeIdx = selectedElement !== null && selectedElement < numElements ? selectedElement : 15;
  const activeElem = elements?.[activeIdx];

  // Element-level calculations
  const elemDetails = useMemo(() => {
    if (!activeElem) return null;
    const obsPhase = activeElem.phase_error_deg;
    const obsGain = activeElem.gain_error_db;

    const cpRe = activeElem.pinn_correction_real ?? activeElem.analytical_correction_real ?? 1.0;
    const cpIm = activeElem.pinn_correction_imag ?? activeElem.analytical_correction_imag ?? 0.0;
    const cMag = activeElem.pinn_correction_magnitude ?? Math.sqrt(cpRe * cpRe + cpIm * cpIm);
    const cPhase = activeElem.pinn_correction_phase_deg ?? (Math.atan2(cpIm, cpRe) * 180) / Math.PI;
    const cGainDb = 20 * Math.log10(Math.max(cMag, 1e-12));

    const vCorrRe = activeElem.hardware_real * cpRe - activeElem.hardware_imag * cpIm;
    const vCorrIm = activeElem.hardware_real * cpIm + activeElem.hardware_imag * cpRe;
    const corrPhase = (Math.atan2(vCorrIm, vCorrRe) * 180) / Math.PI;
    const corrMag = Math.sqrt(vCorrRe * vCorrRe + vCorrIm * vCorrIm);
    const corrGainDb = 20 * Math.log10(Math.max(corrMag, 1e-12));

    const phaseResidual = ((((corrPhase - activeElem.ideal_phase_deg + 180) % 360) + 360) % 360) - 180;
    const gainResidual = corrGainDb - 20 * Math.log10(Math.max(activeElem.ideal_magnitude, 1e-12));

    return {
      obsPhase,
      obsGain,
      rawMag: activeElem.hardware_magnitude,
      rawPhase: activeElem.hardware_phase_deg,
      cMag,
      cPhase,
      cGainDb,
      phaseResidual,
      gainResidual,
    };
  }, [activeElem]);

  // Statistical calculations from actual elements
  const currentMetricDef = METRIC_OPTIONS.find((m) => m.id === calibrationMetric) ?? METRIC_OPTIONS[0];

  const stats = useMemo(() => {
    if (!elements || elements.length === 0) return { mean: 0, std: 0, maxAbs: 0, values: [] };

    const values = elements.map((e) => getElementMetricValue(e, calibrationMetric));
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
    const std = Math.sqrt(variance);
    const maxAbs = Math.max(...values.map((v) => Math.abs(v)));

    return { mean, std, maxAbs, values };
  }, [elements, calibrationMetric]);

  // Histogram data
  const histData = useMemo(() => {
    if (stats.values.length === 0) return [];
    return [
      {
        x: stats.values,
        type: 'histogram' as const,
        marker: { color: 'rgba(8, 145, 178, 0.75)', line: { color: '#0891b2', width: 1 } },
        nbinsx: 18,
        hovertemplate: 'Bin: %{x}<br>Count: %{y}<extra></extra>',
      },
    ];
  }, [stats.values]);

  const histLayout = useMemo(
    () => ({
      margin: { t: 15, r: 15, b: 35, l: 35 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { family: "'Inter', sans-serif", size: 10, color: 'var(--color-text-secondary)' },
      xaxis: {
        title: { text: `${currentMetricDef.label} (${currentMetricDef.unit})`, font: { size: 10 } },
        gridcolor: 'var(--color-border)',
        tickfont: { size: 9 },
      },
      yaxis: {
        title: { text: 'Element Count', font: { size: 10 } },
        gridcolor: 'var(--color-border)',
        tickfont: { size: 9 },
      },
      bargap: 0.05,
    }),
    [currentMetricDef]
  );

  // Compact Beam Chart Traces
  const beam = simulationResult?.beam;
  const beamTraces = useMemo(() => {
    if (!beam) return [];

    const traceDefs = [
      { key: 'ideal', label: 'Ideal', trace: beam.ideal, color: '#16a34a', dash: 'dash', width: 1.5 },
      { key: 'uncalibrated', label: 'Uncalibrated', trace: beam.uncalibrated, color: '#dc2626', dash: 'solid', width: 1.5 },
      { key: 'analytical', label: 'Analytical Baseline', trace: beam.analytical, color: '#0891b2', dash: 'solid', width: 2 },
      { key: 'pinn', label: 'PINN-Cal V4', trace: beam.pinn, color: '#f97316', dash: 'solid', width: 2.5 },
    ];

    return traceDefs
      .filter((d) => d.trace !== null && d.trace !== undefined)
      .map((d) => ({
        x: d.trace!.angles_deg,
        y: d.trace!.power_db,
        type: 'scattergl' as const,
        mode: 'lines' as const,
        name: d.label,
        line: { color: d.color, width: d.width, dash: d.dash },
        hovertemplate: `<b>${d.label}</b><br>Angle: %{x:.1f}°<br>Power: %{y:.2f} dB<extra></extra>`,
      }));
  }, [beam]);

  const beamLayout = useMemo(
    () => ({
      margin: { t: 15, r: 15, b: 35, l: 35 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { family: "'Inter', sans-serif", size: 10, color: 'var(--color-text-secondary)' },
      showlegend: true,
      legend: {
        orientation: 'h' as const,
        x: 0,
        y: 1.15,
        font: { size: 9 },
        bgcolor: 'transparent',
      },
      xaxis: {
        title: { text: 'Angle (deg)', font: { size: 10 } },
        range: [-90, 90],
        gridcolor: 'var(--color-border)',
        tickfont: { size: 9 },
      },
      yaxis: {
        title: { text: 'Power (dB)', font: { size: 10 } },
        range: [-45, 5],
        gridcolor: 'var(--color-border)',
        tickfont: { size: 9 },
      },
      shapes: [
        {
          type: 'line' as const,
          x0: config.target_angle_deg,
          x1: config.target_angle_deg,
          y0: -45,
          y1: 5,
          line: { color: '#0284c7', width: 1.5, dash: 'dot' as const },
        },
      ],
    }),
    [config.target_angle_deg]
  );

  const cal = simulationResult?.calibration;
  const metrics = simulationResult?.metrics;
  const ana = cal?.analytical;
  const pinn = cal?.pinn;

  return (
    <div className="calibration-console-page">
      {/* 4. PAGE HEADER */}
      <header className="console-header-section">
        <div className="console-title-row">
          <div className="console-title-group">
            <h1 className="console-page-title">CALIBRATION CONSOLE</h1>
            <p className="console-page-subtitle">
              Configure hardware impairments, execute PINN-Cal V4, and inspect the resulting calibration performance.
            </p>
          </div>
          <div className="console-live-status-strip">
            <span className="status-item highlight">PINN-Cal V4</span>
            <span className="status-separator">·</span>
            <span className="status-item">64 ELEMENTS</span>
            <span className="status-separator">·</span>
            <span className="status-item">28 GHz</span>
            <span className="status-separator">·</span>
            <span className="status-item">δ = 0.15</span>
            <span className="status-separator">·</span>
            <span className="status-item">α = 0.50</span>
          </div>
        </div>
      </header>

      {/* 3. BALANCED MULTI-COLUMN LAYOUT (DESKTOP) */}
      <div className="console-upper-grid">
        {/* LEFT / PRIMARY: EXPERIMENT CONFIGURATION */}
        <section className="console-controls-col">
          <div className="card console-card console-controls-card">
            <div className="card-header console-card-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span className="header-icon blue-icon">⚙</span>
                <h2 className="console-section-title">EXPERIMENT CONFIGURATION</h2>
              </div>
              <ConceptTrigger conceptKey="phase_error" label="Physics Guide" />
            </div>

            <div className="card-body console-controls-body">
              {/* 7. EXPERIMENT PRESETS */}
              <div className="console-presets-strip">
                <span className="presets-label">QUICK PRESETS:</span>
                <div className="presets-buttons-wrap">
                  <button
                    type="button"
                    className={`console-preset-btn ${isDefaultPreset ? 'active' : ''}`}
                    onClick={() =>
                      applyPreset({
                        target_angle_deg: 20,
                        snr_db: 20,
                        phase_sigma_deg: 20,
                        gain_sigma_db: 2,
                        coupling_mismatch_fraction: 0.2,
                        seed: 42,
                      })
                    }
                  >
                    ★ DEFAULT (+20°, 20 dB)
                  </button>
                  <button
                    type="button"
                    className={`console-preset-btn ${config.snr_db === 10 ? 'active' : ''}`}
                    onClick={() => applyPreset({ snr_db: 10 })}
                  >
                    LOW SNR (10 dB)
                  </button>
                  <button
                    type="button"
                    className={`console-preset-btn ${config.snr_db === 40 ? 'active' : ''}`}
                    onClick={() => applyPreset({ snr_db: 40 })}
                  >
                    HIGH SNR (40 dB)
                  </button>
                  <button
                    type="button"
                    className={`console-preset-btn ${config.phase_sigma_deg === 35 ? 'active' : ''}`}
                    onClick={() => applyPreset({ phase_sigma_deg: 35 })}
                  >
                    HIGH PHASE ERROR (35°)
                  </button>
                  <button
                    type="button"
                    className={`console-preset-btn ${config.gain_sigma_db === 4 ? 'active' : ''}`}
                    onClick={() => applyPreset({ gain_sigma_db: 4 })}
                  >
                    HIGH GAIN ERROR (4 dB)
                  </button>
                </div>
              </div>

              {/* 5 & 6. 6 PHYSICAL CONTROLS WITH ENGINEERING PRESENTATION */}
              <div className="console-controls-list">
                {/* Control 1: TARGET STEERING */}
                <div className="console-control-row">
                  <div className="control-meta-row">
                    <label className="console-control-label" htmlFor="ctrl-target-angle">
                      TARGET STEERING
                    </label>
                    <span className="console-control-value accent-blue">
                      {config.target_angle_deg > 0 ? `+${config.target_angle_deg.toFixed(1)}°` : `${config.target_angle_deg.toFixed(1)}°`}
                    </span>
                  </div>
                  <div className="console-control-desc">
                    Target Angle (°) · Commanded main beam radiation angle relative to broadside.
                  </div>
                  <input
                    id="ctrl-target-angle"
                    aria-label="Target Angle"
                    type="range"
                    className="console-slider"
                    min={-85}
                    max={85}
                    step={1}
                    value={config.target_angle_deg}
                    onChange={(e) => setConfig({ target_angle_deg: Number(e.target.value) })}
                  />
                  <div className="slider-ticks">
                    <span>-85°</span>
                    <span>0° Broadside</span>
                    <span>+85°</span>
                  </div>
                </div>

                {/* Control 2: SNR */}
                <div className="console-control-row">
                  <div className="control-meta-row">
                    <label className="console-control-label" htmlFor="ctrl-snr">
                      SNR
                    </label>
                    <span className="console-control-value accent-blue">{config.snr_db.toFixed(0)} dB</span>
                  </div>
                  <div className="console-control-desc">
                    Signal-to-Noise Ratio (dB) · Thermal receiver noise relative to coupled antenna currents.
                  </div>
                  <input
                    id="ctrl-snr"
                    aria-label="Signal-to-Noise Ratio"
                    type="range"
                    className="console-slider"
                    min={-10}
                    max={60}
                    step={1}
                    value={config.snr_db}
                    onChange={(e) => setConfig({ snr_db: Number(e.target.value) })}
                  />
                  <div className="slider-ticks">
                    <span>-10 dB (Severe Noise)</span>
                    <span>20 dB</span>
                    <span>60 dB (Clean RF)</span>
                  </div>
                </div>

                {/* Control 3: PHASE IMPAIRMENT */}
                <div className="console-control-row">
                  <div className="control-meta-row">
                    <label className="console-control-label" htmlFor="ctrl-phase-sigma">
                      PHASE IMPAIRMENT
                    </label>
                    <span className="console-control-value accent-red">{config.phase_sigma_deg.toFixed(0)}°</span>
                  </div>
                  <div className="console-control-desc">
                    Phase Error σ (°) · Random phase deviation across individual transceivers.
                  </div>
                  <input
                    id="ctrl-phase-sigma"
                    aria-label="Phase Error Sigma"
                    type="range"
                    className="console-slider red-track"
                    min={0}
                    max={90}
                    step={1}
                    value={config.phase_sigma_deg}
                    onChange={(e) => setConfig({ phase_sigma_deg: Number(e.target.value) })}
                  />
                  <div className="slider-ticks">
                    <span>0° (Ideal Phase)</span>
                    <span>45°</span>
                    <span>90° (Severe Drift)</span>
                  </div>
                </div>

                {/* Control 4: GAIN IMPAIRMENT */}
                <div className="console-control-row">
                  <div className="control-meta-row">
                    <label className="console-control-label" htmlFor="ctrl-gain-sigma">
                      GAIN IMPAIRMENT
                    </label>
                    <span className="console-control-value accent-amber">{config.gain_sigma_db.toFixed(2)} dB</span>
                  </div>
                  <div className="console-control-desc">
                    Gain Error σ (dB) · Random power amplifier and RF front-end output variations.
                  </div>
                  <input
                    id="ctrl-gain-sigma"
                    aria-label="Gain Error Sigma"
                    type="range"
                    className="console-slider amber-track"
                    min={0}
                    max={10}
                    step={0.25}
                    value={config.gain_sigma_db}
                    onChange={(e) => setConfig({ gain_sigma_db: Number(e.target.value) })}
                  />
                  <div className="slider-ticks">
                    <span>0.00 dB (Equal Gain)</span>
                    <span>5.00 dB</span>
                    <span>10.00 dB (High Skew)</span>
                  </div>
                </div>

                {/* Control 5: COUPLING MISMATCH */}
                <div className="console-control-row">
                  <div className="control-meta-row">
                    <label className="console-control-label" htmlFor="ctrl-coupling-mismatch">
                      COUPLING MISMATCH
                    </label>
                    <span className="console-control-value accent-cyan">
                      ±{(config.coupling_mismatch_fraction * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="console-control-desc">
                    Coupling Mismatch (%) · Hidden mutual impedance deviation (ΔZ) from nominal model.
                  </div>
                  <input
                    id="ctrl-coupling-mismatch"
                    aria-label="Coupling Mismatch"
                    type="range"
                    className="console-slider cyan-track"
                    min={0}
                    max={0.99}
                    step={0.01}
                    value={config.coupling_mismatch_fraction}
                    onChange={(e) => setConfig({ coupling_mismatch_fraction: Number(e.target.value) })}
                  />
                  <div className="slider-ticks">
                    <span>0% (Known Z)</span>
                    <span>±50%</span>
                    <span>±99% (High ΔZ Mismatch)</span>
                  </div>
                </div>

                {/* Control 6: RANDOM SEED */}
                <div className="console-control-row">
                  <div className="control-meta-row">
                    <label className="console-control-label" htmlFor="ctrl-seed">
                      RANDOM SEED
                    </label>
                    <span className="console-control-value accent-purple">{config.seed}</span>
                  </div>
                  <div className="console-control-desc">
                    Random Seed · Monte Carlo reproducibility seed for noise and hardware realization.
                  </div>
                  <input
                    id="ctrl-seed"
                    aria-label="Random Seed"
                    type="range"
                    className="console-slider purple-track"
                    min={0}
                    max={9999}
                    step={1}
                    value={config.seed}
                    onChange={(e) => setConfig({ seed: Number(e.target.value) })}
                  />
                  <div className="slider-ticks">
                    <span>Seed 0</span>
                    <span>Seed 42 (Reference)</span>
                    <span>Seed 9999</span>
                  </div>
                </div>
              </div>

              {/* 8. PRIMARY ACTION: RUN CALIBRATION BENCH */}
              <div className="console-actions-footer">
                <button
                  type="button"
                  id="btn-run-calibration"
                  className={`console-run-btn ${isComputing ? 'running' : hasError ? 'error' : hasResult ? 'success' : ''}`}
                  onClick={runSimulation}
                  disabled={isComputing}
                  aria-busy={isComputing}
                >
                  <span className="btn-action-icon">{isComputing ? '⏳' : hasResult ? '✓' : '▶'}</span>
                  <span className="btn-action-label">
                    {isComputing
                      ? 'CALIBRATING ARRAY...'
                      : hasError
                      ? 'CALIBRATION FAILED — RETRY'
                      : hasResult
                      ? (superMode ? '★ SUPER CALIBRATE · RUN AGAIN' : 'CALIBRATION COMPLETE · RUN AGAIN')
                      : (superMode ? '★ SUPER CALIBRATE ARRAY' : 'RUN CALIBRATION BENCH')}
                  </span>
                </button>

                <button
                  type="button"
                  id="btn-reset-calibration"
                  className="console-reset-btn"
                  onClick={resetConfig}
                  disabled={isComputing}
                >
                  Reset Standard Parameters
                </button>
              </div>
            </div>
          </div>
        </section>

        {/* RIGHT / PRIMARY: LIVE RESULTS & EXPERIMENT SUMMARY */}
        <section className="console-results-col">
          {/* 9, 10, 18, 19: DYNAMIC BANNER STATE */}
          {isComputing ? (
            <div className="console-banner-card computing" role="status">
              <div className="banner-icon-spin">⚡</div>
              <div className="banner-text">
                <h3 className="banner-title">CALIBRATION RUNNING</h3>
                <p className="banner-subtitle">Evaluating 64-element RF state & computing neural residual...</p>
              </div>
            </div>
          ) : hasError ? (
            <div className="console-banner-card error" role="alert">
              <div className="banner-icon">⚠️</div>
              <div className="banner-text">
                <h3 className="banner-title">CALIBRATION UNAVAILABLE</h3>
                <p className="banner-subtitle">{simulateError || 'Simulation encountered an unexpected backend error.'}</p>
              </div>
              <button type="button" className="banner-retry-btn" onClick={runSimulation}>
                Retry Calibration
              </button>
            </div>
          ) : hasResult ? (
            <div className="console-banner-card success" role="status">
              <div className="banner-header-row">
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                  <span className="banner-icon success-icon">✓</span>
                  <h3 className="banner-title">
                    CALIBRATION COMPLETE
                    {superMode && <span className="super-status-highlight"> · SUPER — ARRAY RESTORED</span>}
                  </h3>
                </div>
                <span className="banner-tag">Live 64-Element Solution</span>
              </div>
              <div className="banner-summary-grid">
                <div className="banner-stat">
                  <span className="banner-stat-label">Target Angle</span>
                  <span className="banner-stat-value">
                    {metrics?.target_angle_deg > 0 ? `+${metrics.target_angle_deg.toFixed(1)}°` : `${metrics?.target_angle_deg.toFixed(1)}°`}
                  </span>
                </div>
                <div className="banner-stat">
                  <span className="banner-stat-label">Peak Angle</span>
                  <span className="banner-stat-value">
                    {metrics?.beam_peak_angle_deg > 0 ? `+${metrics.beam_peak_angle_deg.toFixed(1)}°` : `${metrics?.beam_peak_angle_deg.toFixed(1)}°`}
                  </span>
                </div>
                <div className="banner-stat">
                  <span className="banner-stat-label">Pointing Error</span>
                  <span className="banner-stat-value accent-green">{metrics?.pointing_error_deg.toFixed(2)}°</span>
                </div>
                <div className="banner-stat">
                  <span className="banner-stat-label">Beam Correlation</span>
                  <span className="banner-stat-value accent-green">
                    {metrics?.beam_correlation != null ? `${(metrics.beam_correlation * 100).toFixed(1)}%` : '—'}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="console-banner-card empty" role="status">
              <div className="banner-icon">📡</div>
              <div className="banner-text">
                <h3 className="banner-title">READY TO CALIBRATE</h3>
                <p className="banner-subtitle">
                  Configure the RF impairment conditions above and run the calibration bench to execute PINN-Cal V4.
                </p>
              </div>
            </div>
          )}

          {/* 13 & 14. CURRENT EXPERIMENT & IMPAIRMENT SUMMARY */}
          <div className="card console-card console-experiment-summary-card">
            <div className="card-header console-card-header">
              <h3 className="console-card-title">CURRENT EXPERIMENT SUMMARY</h3>
              <span className="header-meta">Input Conditions vs Output Performance</span>
            </div>
            <div className="card-body" style={{ padding: 16 }}>
              <div className="console-io-split">
                {/* Input Conditions */}
                <div className="io-panel input-panel">
                  <div className="io-panel-title">INPUT CONDITIONS</div>
                  <div className="io-items-grid">
                    <div className="io-item">
                      <span className="io-label">Target Heading:</span>
                      <span className="io-val">{config.target_angle_deg > 0 ? `+${config.target_angle_deg}°` : `${config.target_angle_deg}°`}</span>
                    </div>
                    <div className="io-item">
                      <span className="io-label">SNR:</span>
                      <span className="io-val">{config.snr_db} dB</span>
                    </div>
                    <div className="io-item">
                      <span className="io-label">Phase Impairment:</span>
                      <span className="io-val accent-red">σ = {config.phase_sigma_deg}°</span>
                    </div>
                    <div className="io-item">
                      <span className="io-label">Gain Impairment:</span>
                      <span className="io-val accent-amber">σ = {config.gain_sigma_db} dB</span>
                    </div>
                    <div className="io-item">
                      <span className="io-label">Coupling Mismatch:</span>
                      <span className="io-val accent-cyan">±{(config.coupling_mismatch_fraction * 100).toFixed(0)}% ΔZ</span>
                    </div>
                    <div className="io-item">
                      <span className="io-label">Seed:</span>
                      <span className="io-val">{config.seed}</span>
                    </div>
                  </div>
                </div>

                {/* Output Performance */}
                <div className="io-panel output-panel">
                  <div className="io-panel-title">OUTPUT PERFORMANCE</div>
                  {hasResult && cal && metrics ? (
                    <div className="io-items-grid">
                      <div className="io-item">
                        <span className="io-label">Analytical NMSE:</span>
                        <span className="io-val">{ana ? `${ana.nmse_db.toFixed(2)} dB` : '—'}</span>
                      </div>
                      <div className="io-item highlight-orange">
                        <span className="io-label">PINN-Cal V4 NMSE:</span>
                        <span className="io-val">{pinn ? `${pinn.nmse_db.toFixed(2)} dB` : '—'}</span>
                      </div>
                      <div className="io-item">
                        <span className="io-label">Beam Peak Angle:</span>
                        <span className="io-val">{metrics.beam_peak_angle_deg > 0 ? `+${metrics.beam_peak_angle_deg.toFixed(1)}°` : `${metrics.beam_peak_angle_deg.toFixed(1)}°`}</span>
                      </div>
                      <div className="io-item">
                        <span className="io-label">Pointing Error:</span>
                        <span className="io-val accent-green">{metrics.pointing_error_deg.toFixed(2)}°</span>
                      </div>
                      <div className="io-item">
                        <span className="io-label">Beam Correlation:</span>
                        <span className="io-val accent-green">{metrics.beam_correlation != null ? `${(metrics.beam_correlation * 100).toFixed(1)}%` : '—'}</span>
                      </div>
                      <div className="io-item">
                        <span className="io-label">3 dB Beamwidth:</span>
                        <span className="io-val">{metrics.beamwidth_3db_deg != null ? `${metrics.beamwidth_3db_deg.toFixed(1)}°` : '—'}</span>
                      </div>
                    </div>
                  ) : (
                    <div className="awaiting-panel-compact">
                      <span>Awaiting simulation execution...</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* 15. ELEMENT-LEVEL RESULT PREVIEW */}
          <div className="card console-card console-element-preview-card">
            <div className="card-header console-card-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="header-icon purple-icon">🔬</span>
                <h3 className="console-card-title">SELECTED ELEMENT #{activeIdx + 1}</h3>
              </div>
              <button
                type="button"
                className="console-link-btn"
                onClick={() => setActiveView('simulation')}
                title="Open detailed Live 64-Element Array inspector"
              >
                Inspect in Live Array →
              </button>
            </div>
            <div className="card-body" style={{ padding: 14 }}>
              {!hasResult || !elemDetails ? (
                <div className="awaiting-panel-compact">
                  <span>Run calibration to inspect element #{activeIdx + 1} compensation metrics.</span>
                </div>
              ) : (
                <div className="element-tri-preview">
                  <div className="tri-col impaired">
                    <span className="tri-col-title">Observed / Impaired</span>
                    <div className="tri-stat">
                      <span className="tri-label">Phase Error:</span>
                      <span className="tri-val">{elemDetails.obsPhase > 0 ? `+${elemDetails.obsPhase.toFixed(1)}°` : `${elemDetails.obsPhase.toFixed(1)}°`}</span>
                    </div>
                    <div className="tri-stat">
                      <span className="tri-label">Gain Error:</span>
                      <span className="tri-val">{elemDetails.obsGain > 0 ? `+${elemDetails.obsGain.toFixed(2)} dB` : `${elemDetails.obsGain.toFixed(2)} dB`}</span>
                    </div>
                    <div className="tri-stat">
                      <span className="tri-label">Raw Magnitude:</span>
                      <span className="tri-val">{elemDetails.rawMag.toFixed(3)}</span>
                    </div>
                  </div>

                  <div className="tri-col estimated">
                    <span className="tri-col-title">V4 Estimated Compensation</span>
                    <div className="tri-stat">
                      <span className="tri-label">Complex Weight:</span>
                      <span className="tri-val">|C| = {elemDetails.cMag.toFixed(3)}</span>
                    </div>
                    <div className="tri-stat">
                      <span className="tri-label">Phase Comp:</span>
                      <span className="tri-val">{elemDetails.cPhase > 0 ? `+${elemDetails.cPhase.toFixed(1)}°` : `${elemDetails.cPhase.toFixed(1)}°`}</span>
                    </div>
                    <div className="tri-stat">
                      <span className="tri-label">Gain Comp:</span>
                      <span className="tri-val">{elemDetails.cGainDb > 0 ? `+${elemDetails.cGainDb.toFixed(2)} dB` : `${elemDetails.cGainDb.toFixed(2)} dB`}</span>
                    </div>
                  </div>

                  <div className="tri-col post">
                    <span className="tri-col-title">Post-Compensation</span>
                    <div className="tri-stat">
                      <span className="tri-label">Residual Phase:</span>
                      <span className="tri-val accent-green">{elemDetails.phaseResidual > 0 ? `+${elemDetails.phaseResidual.toFixed(2)}°` : `${elemDetails.phaseResidual.toFixed(2)}°`}</span>
                    </div>
                    <div className="tri-stat">
                      <span className="tri-label">Residual Gain:</span>
                      <span className="tri-val accent-green">{elemDetails.gainResidual > 0 ? `+${elemDetails.gainResidual.toFixed(2)} dB` : `${elemDetails.gainResidual.toFixed(2)} dB`}</span>
                    </div>
                    <div className="tri-stat">
                      <span className="tri-label">Status:</span>
                      <span className="tri-val accent-green">Calibrated</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* 16. BEAM RESULT PREVIEW */}
          <div className="card console-card console-beam-preview-card">
            <div className="card-header console-card-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="header-icon green-icon">📡</span>
                <h3 className="console-card-title">BEAM RESTORATION PREVIEW</h3>
              </div>
              <button
                type="button"
                className="console-link-btn"
                onClick={() => setActiveView('beam')}
                title="Open detailed Beam Restoration module"
              >
                View Full Beam Restoration →
              </button>
            </div>
            <div className="card-body" style={{ padding: 12 }}>
              {!hasResult || !beam ? (
                <div className="awaiting-panel-compact" style={{ height: 180 }}>
                  <span>Awaiting calibration to generate synthesized beam pattern...</span>
                </div>
              ) : (
                <div>
                  <div style={{ height: 190, width: '100%' }}>
                    <Plot
                      data={beamTraces}
                      layout={beamLayout}
                      useResizeHandler
                      style={{ width: '100%', height: '100%' }}
                      config={{ displayModeBar: false, responsive: true }}
                    />
                  </div>
                  <div className="beam-compact-metrics-strip">
                    <div className="compact-metric">
                      <span className="cm-label">Target Angle:</span>
                      <span className="cm-val">{config.target_angle_deg > 0 ? `+${config.target_angle_deg}°` : `${config.target_angle_deg}°`}</span>
                    </div>
                    <div className="compact-metric">
                      <span className="cm-label">Peak Angle:</span>
                      <span className="cm-val">{metrics?.beam_peak_angle_deg > 0 ? `+${metrics.beam_peak_angle_deg.toFixed(1)}°` : `${metrics?.beam_peak_angle_deg.toFixed(1)}°`}</span>
                    </div>
                    <div className="compact-metric">
                      <span className="cm-label">Pointing Error:</span>
                      <span className="cm-val accent-green">{metrics?.pointing_error_deg.toFixed(2)}°</span>
                    </div>
                    <div className="compact-metric">
                      <span className="cm-label">3 dB Beamwidth:</span>
                      <span className="cm-val">{metrics?.beamwidth_3db_deg != null ? `${metrics.beamwidth_3db_deg.toFixed(1)}°` : '—'}</span>
                    </div>
                    <div className="compact-metric">
                      <span className="cm-label">PSLL:</span>
                      <span className="cm-val">{metrics?.psll_db != null ? `${metrics.psll_db.toFixed(1)} dB` : '—'}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>
      </div>

      {/* 11. PRIMARY METRIC WALL (FULL WIDTH) */}
      <section className="console-full-width-section">
        <div className="card console-card console-metric-wall-card">
          <div className="card-header console-card-header">
            <h2 className="console-section-title">PRIMARY METRIC WALL</h2>
            <span className="header-meta">Key Live Calibration Performance Indicators</span>
          </div>
          <div className="card-body" style={{ padding: 16 }}>
            <div className="console-metric-wall-grid">
              {/* Card 1: NMSE */}
              <div className="console-metric-card primary-border">
                <span className="metric-tag">ACCURACY</span>
                <div className="metric-value-large accent-orange">
                  {pinn ? `${pinn.nmse_db.toFixed(2)} dB` : metrics?.pinn_nmse_db != null ? `${metrics.pinn_nmse_db.toFixed(2)} dB` : '—'}
                </div>
                <div className="metric-title-text">NMSE (PINN-Cal V4)</div>
                <div className="metric-desc-text">Normalized Mean Squared Error across 64 elements.</div>
              </div>

              {/* Card 2: Phase Error */}
              <div className="console-metric-card">
                <span className="metric-tag">PHASE</span>
                <div className="metric-value-large">
                  {pinn ? `${pinn.mean_phase_error_deg.toFixed(2)}°` : '—'}
                </div>
                <div className="metric-title-text">Phase Error σ_post</div>
                <div className="metric-desc-text">Post-compensation mean residual phase variance.</div>
              </div>

              {/* Card 3: Gain Error */}
              <div className="console-metric-card">
                <span className="metric-tag">GAIN</span>
                <div className="metric-value-large">
                  {pinn ? `${pinn.mean_gain_error_db.toFixed(2)} dB` : '—'}
                </div>
                <div className="metric-title-text">Gain Error σ_post</div>
                <div className="metric-desc-text">Post-compensation mean residual gain variance.</div>
              </div>

              {/* Card 4: Pointing Error */}
              <div className="console-metric-card">
                <span className="metric-tag">BEAM STEERING</span>
                <div className="metric-value-large accent-green">
                  {metrics ? `${metrics.pointing_error_deg.toFixed(2)}°` : '—'}
                </div>
                <div className="metric-title-text">Pointing Error</div>
                <div className="metric-desc-text">Main lobe bore-sight deviation from target heading.</div>
              </div>

              {/* Card 5: Beam Correlation */}
              <div className="console-metric-card">
                <span className="metric-tag">FIDELITY</span>
                <div className="metric-value-large accent-green">
                  {metrics?.beam_correlation != null ? `${(metrics.beam_correlation * 100).toFixed(1)}%` : '—'}
                </div>
                <div className="metric-title-text">Beam Correlation</div>
                <div className="metric-desc-text">Radiation pattern correlation vs ideal synthesized beam.</div>
              </div>

              {/* Card 6: Beamwidth */}
              <div className="console-metric-card">
                <span className="metric-tag">APERTURE</span>
                <div className="metric-value-large">
                  {metrics?.beamwidth_3db_deg != null ? `${metrics.beamwidth_3db_deg.toFixed(1)}°` : '—'}
                </div>
                <div className="metric-title-text">Beamwidth (3 dB)</div>
                <div className="metric-desc-text">Half-power beamwidth of the synthesized radiation pattern.</div>
              </div>

              {/* Card 7: PSLL */}
              <div className="console-metric-card">
                <span className="metric-tag">SIDE LOBES</span>
                <div className="metric-value-large">
                  {metrics?.psll_db != null ? `${metrics.psll_db.toFixed(1)} dB` : '—'}
                </div>
                <div className="metric-title-text">PSLL</div>
                <div className="metric-desc-text">Peak Side-Lobe Level relative to main beam peak.</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 12. METHOD COMPARISON (FULL WIDTH) */}
      <section className="console-full-width-section">
        <div className="card console-card console-method-comparison-card">
          <div className="card-header console-card-header">
            <div>
              <h2 className="console-section-title">METHOD COMPARISON</h2>
              <span className="header-meta">ANALYTICAL BASELINE vs PINN-Cal V4</span>
            </div>
            <ConceptTrigger conceptKey="analytical_baseline" label="Physics Preconditioner" />
          </div>
          <div className="card-body" style={{ padding: 0 }}>
            <div className="comparison-description-bar">
              Neutral engineering comparison of live calibration performance across 64 elements under identical simulated hardware conditions.
            </div>
            <div className="table-responsive">
              <table className="console-comparison-table">
                <thead>
                  <tr>
                    <th>Metric Parameter</th>
                    <th>Analytical Baseline</th>
                    <th>PINN-Cal V4</th>
                    <th>Evaluation Scope</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="metric-name-cell">
                      <strong>NMSE (Normalized MSE)</strong>
                    </td>
                    <td className="ana-val">{ana ? `${ana.nmse_db.toFixed(3)} dB` : '—'}</td>
                    <td className="pinn-val">{pinn ? `${pinn.nmse_db.toFixed(3)} dB` : '—'}</td>
                    <td className="scope-cell">64 elements RF vector error</td>
                  </tr>
                  <tr>
                    <td className="metric-name-cell">
                      <strong>Mean Phase Error</strong>
                    </td>
                    <td className="ana-val">{ana ? `${ana.mean_phase_error_deg.toFixed(3)}°` : '—'}</td>
                    <td className="pinn-val">{pinn ? `${pinn.mean_phase_error_deg.toFixed(3)}°` : '—'}</td>
                    <td className="scope-cell">Transceiver phase deviation</td>
                  </tr>
                  <tr>
                    <td className="metric-name-cell">
                      <strong>Mean Gain Error</strong>
                    </td>
                    <td className="ana-val">{ana ? `${ana.mean_gain_error_db.toFixed(3)} dB` : '—'}</td>
                    <td className="pinn-val">{pinn ? `${pinn.mean_gain_error_db.toFixed(3)} dB` : '—'}</td>
                    <td className="scope-cell">Front-end gain deviation</td>
                  </tr>
                  <tr>
                    <td className="metric-name-cell">
                      <strong>Mean Relative Error</strong>
                    </td>
                    <td className="ana-val">{ana ? `${ana.mean_relative_error_pct.toFixed(2)}%` : '—'}</td>
                    <td className="pinn-val">{pinn ? `${pinn.mean_relative_error_pct.toFixed(2)}%` : '—'}</td>
                    <td className="scope-cell">Complex weight norm deviation</td>
                  </tr>
                  {ana?.current_residual_pct !== null && ana?.current_residual_pct !== undefined && (
                    <tr>
                      <td className="metric-name-cell">
                        <strong>Coupled Current Residual</strong>
                      </td>
                      <td className="ana-val">{ana.current_residual_pct.toFixed(2)}%</td>
                      <td className="pinn-val">{pinn?.current_residual_pct != null ? `${pinn.current_residual_pct.toFixed(2)}%` : '—'}</td>
                      <td className="scope-cell">Z-impedance mutual current match</td>
                    </tr>
                  )}
                  <tr>
                    <td className="metric-name-cell">
                      <strong>Beam Pointing Error</strong>
                    </td>
                    <td className="ana-val">{beam?.analytical ? `${beam.analytical.pointing_error_deg.toFixed(2)}°` : '—'}</td>
                    <td className="pinn-val">{beam?.pinn ? `${beam.pinn.pointing_error_deg.toFixed(2)}°` : '—'}</td>
                    <td className="scope-cell">Radiation peak heading shift</td>
                  </tr>
                  <tr>
                    <td className="metric-name-cell">
                      <strong>Beam Correlation</strong>
                    </td>
                    <td className="ana-val">
                      {beam?.analytical?.beam_correlation != null ? `${(beam.analytical.beam_correlation * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="pinn-val">
                      {beam?.pinn?.beam_correlation != null ? `${(beam.pinn.beam_correlation * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="scope-cell">Pattern overlap vs ideal synthesized</td>
                  </tr>
                  <tr>
                    <td className="metric-name-cell">
                      <strong>3 dB Beamwidth</strong>
                    </td>
                    <td className="ana-val">{beam?.analytical?.beamwidth_3db_deg != null ? `${beam.analytical.beamwidth_3db_deg.toFixed(1)}°` : '—'}</td>
                    <td className="pinn-val">{beam?.pinn?.beamwidth_3db_deg != null ? `${beam.pinn.beamwidth_3db_deg.toFixed(1)}°` : '—'}</td>
                    <td className="scope-cell">Main lobe angular resolution</td>
                  </tr>
                  <tr>
                    <td className="metric-name-cell">
                      <strong>Peak Side-Lobe Level (PSLL)</strong>
                    </td>
                    <td className="ana-val">{beam?.analytical?.psll_db != null ? `${beam.analytical.psll_db.toFixed(1)} dB` : '—'}</td>
                    <td className="pinn-val">{beam?.pinn?.psll_db != null ? `${beam.pinn.psll_db.toFixed(1)} dB` : '—'}</td>
                    <td className="scope-cell">Side lobe suppression level</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      {/* 15 & DIAGNOSTICS: 64-ELEMENT HEATMAP GRID & ERROR DISTRIBUTION */}
      <section className="console-full-width-section">
        <div style={{ display: 'grid', gridTemplateColumns: '1.35fr 1fr', gap: 16 }}>
          {/* 64-Element Diagnostic Map */}
          <div className="card console-card console-diagnostic-grid-card">
            <div className="card-header console-card-header" style={{ flexWrap: 'wrap', gap: 8 }}>
              <div>
                <h3 className="console-card-title">64-ELEMENT DIAGNOSTIC MAP</h3>
                <span className="header-meta">{currentMetricDef.description}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <label style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--color-text-secondary)' }}>
                  Diagnostic Metric:
                </label>
                <select
                  className="form-select"
                  value={calibrationMetric}
                  onChange={(e) => setCalibrationMetric(e.target.value as CalibrationExplorerMetric)}
                  style={{ fontSize: '0.8125rem', padding: '6px 12px', minWidth: 220 }}
                >
                  {METRIC_OPTIONS.map((opt) => (
                    <option key={opt.id} value={opt.id}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="card-body" style={{ padding: 16 }}>
              {!elements ? (
                <div className="awaiting-panel">
                  <span>Awaiting simulation execution...</span>
                </div>
              ) : (
                <div>
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(8, 1fr)',
                      gap: 6,
                      marginBottom: 16,
                    }}
                  >
                    {elements.map((elem, idx) => {
                      const val = getElementMetricValue(elem, calibrationMetric);
                      const isSelected = activeIdx === idx;

                      const maxVal = Math.max(stats.maxAbs, 1e-4);
                      const ratio = Math.max(-1, Math.min(1, val / maxVal));

                      let bg = 'var(--color-surface-2, #f1f5f9)';
                      if (ratio > 0.05) {
                        bg = `rgba(8, 145, 178, ${0.15 + 0.7 * ratio})`;
                      } else if (ratio < -0.05) {
                        bg = `rgba(220, 38, 38, ${0.15 + 0.7 * Math.abs(ratio)})`;
                      }

                      return (
                        <div
                          key={idx}
                          onClick={() => setSelectedElement(idx)}
                          style={{
                            background: bg,
                            border: isSelected ? '2px solid var(--primary, #0f172a)' : '1px solid var(--color-border)',
                            borderRadius: 6,
                            padding: '8px 4px',
                            textAlign: 'center',
                            cursor: 'pointer',
                            transition: 'all 0.15s ease',
                            boxShadow: isSelected ? '0 0 0 2px var(--color-primary)' : 'none',
                          }}
                          title={`Element #${idx + 1}: ${val.toFixed(2)} ${currentMetricDef.unit}`}
                        >
                          <div style={{ fontSize: '0.6875rem', color: 'var(--color-text-tertiary)', fontWeight: 600 }}>
                            #{idx + 1}
                          </div>
                          <div
                            style={{
                              fontFamily: 'var(--font-mono)',
                              fontSize: '0.75rem',
                              fontWeight: 700,
                              color: 'var(--color-text-primary)',
                              marginTop: 2,
                            }}
                          >
                            {val > 0 ? `+${val.toFixed(1)}` : val.toFixed(1)}
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Color Legend */}
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      fontSize: '0.75rem',
                      color: 'var(--color-text-secondary)',
                    }}
                  >
                    <span>Negative ({(-stats.maxAbs).toFixed(1)}{currentMetricDef.unit})</span>
                    <div
                      style={{
                        width: 160,
                        height: 8,
                        borderRadius: 4,
                        background: 'linear-gradient(to right, #dc2626, #f1f5f9, #0891b2)',
                      }}
                    />
                    <span>Positive (+{stats.maxAbs.toFixed(1)}{currentMetricDef.unit})</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Statistical Summary + Distribution Histogram */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Statistical Metrics Card */}
            <div className="card console-card">
              <div className="card-header console-card-header">
                <h3 className="console-card-title">POPULATION STATISTICS</h3>
              </div>
              <div className="card-body" style={{ padding: 14 }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
                  <div className="stat-pill-box">
                    <div className="stat-pill-label">SAMPLE MEAN (μ)</div>
                    <div className="stat-pill-val">
                      {stats.mean > 0 ? `+${stats.mean.toFixed(2)}` : stats.mean.toFixed(2)} {currentMetricDef.unit}
                    </div>
                  </div>

                  <div className="stat-pill-box">
                    <div className="stat-pill-label">STANDARD DEV (σ)</div>
                    <div className="stat-pill-val accent-blue">
                      {stats.std.toFixed(2)} {currentMetricDef.unit}
                    </div>
                  </div>

                  <div className="stat-pill-box">
                    <div className="stat-pill-label">MAX DEVIATION</div>
                    <div className="stat-pill-val accent-red">
                      {stats.maxAbs.toFixed(2)} {currentMetricDef.unit}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Error Distribution Histogram */}
            <div className="card console-card" style={{ flex: 1 }}>
              <div className="card-header console-card-header">
                <h3 className="console-card-title">ERROR DISTRIBUTION HISTOGRAM</h3>
              </div>
              <div className="card-body" style={{ padding: 10, minHeight: 200 }}>
                {!elements ? (
                  <div className="awaiting-panel">
                    <span>Awaiting simulation...</span>
                  </div>
                ) : (
                  <Plot
                    data={histData}
                    layout={histLayout}
                    useResizeHandler
                    style={{ width: '100%', height: '100%' }}
                    config={{ displayModeBar: false, responsive: true }}
                  />
                )}
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
