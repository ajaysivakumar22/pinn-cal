import React, { useState } from 'react';
import { AsyncPlot as Plot } from '../components/common/AsyncPlot';
import { useAppStore } from '../store/appStore';
import type { SweepType, RobustnessPoint } from '../types/api';
import { ConceptTrigger } from '../components/common/ConceptExplainer';

const SWEEP_OPTIONS: {
  value: SweepType;
  label: string;
  unit: string;
  name: string;
  levels: number[];
  description: string;
  category: string;
}[] = [
  {
    value: 'snr',
    name: 'Signal-to-Noise Ratio (SNR) Sweep',
    label: 'SNR Sweep (40 dB → 10 dB)',
    unit: 'dB',
    levels: [40, 30, 20, 15, 10],
    description:
      'Evaluates calibration resilience as receiver thermal noise increases from pristine laboratory conditions (40 dB) down to severe noise (10 dB).',
    category: 'NOISE STRESS',
  },
  {
    value: 'coupling',
    name: 'Coupling Uncertainty (ΔZ) Sweep',
    label: 'Coupling Mismatch Sweep (0% → 50%)',
    unit: 'fraction',
    levels: [0, 0.1, 0.2, 0.3, 0.5],
    description:
      'Tests tolerance to hidden mutual impedance variations from nominal model Z across 0% to 50% mismatch.',
    category: 'COUPLING STRESS',
  },
  {
    value: 'steering',
    name: 'Steering Sector Scan Sweep',
    label: 'Steering Sector Sweep (-50° → +50°)',
    unit: '°',
    levels: [-50, -30, -10, 10, 30, 50],
    description:
      'Measures angular consistency and aperture calibration accuracy across wide scan sectors from -50° to +50°.',
    category: 'GEOMETRY STRESS',
  },
];

const NUM_SAMPLES_OPTIONS = [20, 50, 100, 200];

export const RobustnessLab: React.FC = () => {
  const { robustnessResult, robustnessState, robustnessError, runRobustness, theme, superMode } = useAppStore();
  const [sweepType, setSweepType] = useState<SweepType>('snr');
  const [numSamples, setNumSamples] = useState(50);
  const [seed, setSeed] = useState(4200);

  const isComputing = robustnessState === 'computing';
  const hasResult = !!robustnessResult && robustnessState === 'success';
  const hasError = robustnessState === 'error' || !!robustnessError;
  const isDark = theme === 'dark';

  const handleRun = () => {
    runRobustness({ sweep_type: sweepType, num_samples: numSamples, seed });
  };

  const currentSweepMeta = SWEEP_OPTIONS.find((o) => o.value === sweepType) ?? SWEEP_OPTIONS[0];
  const result = robustnessResult;

  // Primary Chart: NMSE vs Parameter
  const anaNmseTrace = result
    ? {
        x: result.points.map((p: RobustnessPoint) => p.level),
        y: result.points.map((p: RobustnessPoint) => p.analytical_nmse_db),
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        name: 'Analytical Baseline',
        line: { color: '#0891b2', width: 2.5 },
        marker: { size: 7, color: '#0891b2' },
        hovertemplate: `<b>Analytical Baseline</b><br>${currentSweepMeta.name}: %{x} ${currentSweepMeta.unit}<br>NMSE: %{y:.2f} dB<extra></extra>`,
      }
    : null;

  const pinnNmseTrace =
    result && result.pinn_available
      ? {
          x: result.points.map((p: RobustnessPoint) => p.level),
          y: result.points.map((p: RobustnessPoint) => p.pinn_nmse_db ?? null),
          type: 'scatter' as const,
          mode: 'lines+markers' as const,
          name: 'PINN-Cal V4',
          line: { color: '#ea580c', width: 2.8 },
          marker: { size: 7, color: '#ea580c' },
          hovertemplate: `<b>PINN-Cal V4</b><br>${currentSweepMeta.name}: %{x} ${currentSweepMeta.unit}<br>NMSE: %{y:.2f} dB<extra></extra>`,
        }
      : null;

  // Secondary Chart: Phase Error vs Parameter
  const phaseAnaTrace = result
    ? {
        x: result.points.map((p: RobustnessPoint) => p.level),
        y: result.points.map((p: RobustnessPoint) => p.analytical_phase_error_deg),
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        name: 'Analytical Baseline',
        line: { color: '#0891b2', width: 2.2 },
        marker: { size: 6, color: '#0891b2' },
        hovertemplate: `<b>Analytical Baseline</b><br>${currentSweepMeta.name}: %{x} ${currentSweepMeta.unit}<br>Phase Error: %{y:.2f}°<extra></extra>`,
      }
    : null;

  const phasePinnTrace =
    result && result.pinn_available
      ? {
          x: result.points.map((p: RobustnessPoint) => p.level),
          y: result.points.map((p: RobustnessPoint) => p.pinn_phase_error_deg ?? null),
          type: 'scatter' as const,
          mode: 'lines+markers' as const,
          name: 'PINN-Cal V4',
          line: { color: '#ea580c', width: 2.4 },
          marker: { size: 6, color: '#ea580c' },
          hovertemplate: `<b>PINN-Cal V4</b><br>${currentSweepMeta.name}: %{x} ${currentSweepMeta.unit}<br>Phase Error: %{y:.2f}°<extra></extra>`,
        }
      : null;

  const plotLayout = (yTitle: string) => ({
    margin: { t: 25, r: 20, b: 45, l: 55 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {
      family: "'Inter', sans-serif",
      size: 11,
      color: isDark ? '#94a3b8' : '#475569',
    },
    legend: {
      orientation: 'h' as const,
      x: 0,
      y: 1.15,
      bgcolor: isDark ? 'rgba(15, 23, 42, 0.85)' : 'rgba(255, 255, 255, 0.92)',
      bordercolor: isDark ? '#334155' : '#e2e8f0',
      borderwidth: 1,
      font: { size: 10.5 },
    },
    xaxis: {
      title: { text: `${currentSweepMeta.name} (${currentSweepMeta.unit})`, font: { size: 11 } },
      gridcolor: isDark ? '#1e293b' : '#e2e8f0',
      zerolinecolor: isDark ? '#334155' : '#cbd5e1',
      tickfont: { size: 10 },
      autorange: sweepType === 'snr' ? ('reversed' as const) : true,
    },
    yaxis: {
      title: { text: yTitle, font: { size: 11 } },
      gridcolor: isDark ? '#1e293b' : '#e2e8f0',
      zerolinecolor: isDark ? '#334155' : '#cbd5e1',
      tickfont: { size: 10 },
    },
    hovermode: 'x unified' as const,
  });

  return (
    <div className="robustness-lab-page">
      {/* 3. PAGE HEADER */}
      <header className="robustness-header-section">
        <div className="robustness-title-row">
          <div className="robustness-title-group">
            <h1 className="robustness-page-title">ROBUSTNESS LAB</h1>
            <p className="robustness-page-subtitle">
              Stress-test PINN-Cal V4 under degraded noise, hardware impairment, and coupling mismatch conditions using fresh Monte Carlo realizations.
            </p>
          </div>
          <div className="robustness-identity-strip">
            <span className="strip-item">64 ELEMENTS</span>
            <span className="strip-dot">·</span>
            <span className="strip-item">28 GHz</span>
            <span className="strip-dot">·</span>
            <span className="strip-item">0.5 λ</span>
            <span className="strip-dot">·</span>
            <span className="strip-item highlight-orange">PINN-CAL V4</span>
            <span className="strip-dot">·</span>
            <span className="strip-item">δ = 0.15</span>
            <span className="strip-dot">·</span>
            <span className="strip-item">α = 0.50</span>
          </div>
        </div>
      </header>

      {/* 4. MONTE CARLO EXPERIMENT CONFIGURATION */}
      <section className="card robustness-card robustness-config-card">
        <div className="card-header robustness-card-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span className="header-icon orange-icon">🧪</span>
            <h2 className="robustness-section-title">MONTE CARLO EXPERIMENT CONFIGURATION</h2>
          </div>
          <span className="header-meta">Fresh Evaluated Realizations per Condition Level</span>
        </div>

        <div className="card-body robustness-config-body">
          <div className="robustness-controls-grid">
            {/* Control 1: Sweep Parameter */}
            <div className="robustness-form-group">
              <label className="robustness-form-label" htmlFor="ctrl-sweep-type">
                SWEEP PARAMETER
              </label>
              <select
                id="ctrl-sweep-type"
                className="form-select robustness-select"
                value={sweepType}
                onChange={(e) => setSweepType(e.target.value as SweepType)}
                disabled={isComputing}
              >
                {SWEEP_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
              <div className="robustness-form-hint">Operating stress condition domain</div>
            </div>

            {/* Control 2: Samples per Level */}
            <div className="robustness-form-group">
              <label className="robustness-form-label" htmlFor="ctrl-num-samples">
                SAMPLES / LEVEL
              </label>
              <select
                id="ctrl-num-samples"
                className="form-select robustness-select"
                value={numSamples}
                onChange={(e) => setNumSamples(Number(e.target.value))}
                disabled={isComputing}
              >
                {NUM_SAMPLES_OPTIONS.map((n) => (
                  <option key={n} value={n}>
                    {n} realizations per level
                  </option>
                ))}
              </select>
              <div className="robustness-form-hint">Statistical sample depth per stress level</div>
            </div>

            {/* Control 3: Random Seed */}
            <div className="robustness-form-group">
              <label className="robustness-form-label" htmlFor="ctrl-seed">
                RANDOM SEED
              </label>
              <input
                id="ctrl-seed"
                type="number"
                className="form-input robustness-input"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                disabled={isComputing}
                min={0}
              />
              <div className="robustness-form-hint">Monte Carlo noise seed base</div>
            </div>

            {/* 5. Live Experiment Action Button */}
            <div className="robustness-action-wrap">
              <button
                type="button"
                id="btn-run-robustness"
                className={`robustness-run-btn ${isComputing ? 'computing' : hasResult ? 'success' : ''}`}
                onClick={handleRun}
                disabled={isComputing}
                aria-busy={isComputing}
              >
                <span className="action-icon">{isComputing ? '⏳' : hasResult ? '✓' : '▶'}</span>
                <span className="action-text">
                  {isComputing
                    ? 'RUNNING MONTE CARLO...'
                    : hasError
                    ? 'MONTE CARLO FAILED — RETRY'
                    : hasResult
                    ? (superMode ? '★ SUPER MONTE CARLO COMPLETE · RE-RUN' : 'MONTE CARLO COMPLETE · RE-RUN')
                    : 'START MONTE CARLO EXPERIMENT'}
                </span>
              </button>
            </div>
          </div>

          {/* 7. ACTIVE STRESS CONDITION CALLOUT */}
          <div className="robustness-stress-callout">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="badge-tag orange-tag">{currentSweepMeta.category}</span>
              <strong className="stress-title">Active Stress Condition:</strong>
              <span className="stress-desc">{currentSweepMeta.name}</span>
            </div>
            <p className="stress-explanation">{currentSweepMeta.description}</p>
          </div>
        </div>
      </section>

      {/* 17, 18, 19: STATUS BANNERS */}
      {isComputing ? (
        <div className="card robustness-card robustness-alert-banner computing" role="status">
          <span className="alert-icon-spin">⚡</span>
          <div>
            <h3 className="alert-title">MONTE CARLO RUNNING</h3>
            <p className="alert-desc">
              Generating fresh simulated realizations and evaluating Analytical Baseline vs. PINN-Cal V4 across {currentSweepMeta.levels.length} stress levels...
            </p>
          </div>
        </div>
      ) : hasError ? (
        <div className="card robustness-card robustness-alert-banner error" role="alert">
          <span className="alert-icon">⚠️</span>
          <div>
            <h3 className="alert-title">ROBUSTNESS DATA UNAVAILABLE</h3>
            <p className="alert-desc">{robustnessError || 'Evaluation encountered an unexpected backend error.'}</p>
          </div>
          <button type="button" className="btn-alert-retry" onClick={handleRun}>
            Retry Experiment
          </button>
        </div>
      ) : !hasResult ? (
        <div className="card robustness-card robustness-alert-banner empty" role="status">
          <span className="alert-icon">🧪</span>
          <div>
            <h3 className="alert-title">READY FOR STRESS TEST</h3>
            <p className="alert-desc">
              Configure the experiment parameters above and click <strong>START MONTE CARLO EXPERIMENT</strong> to evaluate fresh statistical realizations.
            </p>
          </div>
          <ConceptTrigger conceptKey="snr" label="Noise Model Guide" />
        </div>
      ) : (
        /* 20. RESULT FRESHNESS BANNER */
        <div className="card robustness-card robustness-freshness-banner" role="status">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span className="fresh-dot">●</span>
            <span className="fresh-badge">FRESH LIVE MONTE CARLO RESULT</span>
            <span className="fresh-details">
              Evaluated {numSamples} realizations per condition level (Seed: {seed})
            </span>
          </div>
          <span className="fresh-tag">{currentSweepMeta.levels.length} Stress Levels Computed</span>
        </div>
      )}

      {/* 8 & 15. EXPERIMENT ROADMAP / STRESS PROGRESSION */}
      <section className="card robustness-card robustness-roadmap-card">
        <div className="card-header robustness-card-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <h3 className="robustness-card-title">EXPERIMENT LEVEL ROADMAP</h3>
            <span className="header-meta">Progression across stress domain ({currentSweepMeta.unit})</span>
          </div>
          {sweepType === 'snr' && (
            <div className="snr-story-badges">
              <span className="story-badge green-badge">HIGH SNR (Pristine)</span>
              <span className="story-arrow">→</span>
              <span className="story-badge yellow-badge">MODERATE NOISE</span>
              <span className="story-arrow">→</span>
              <span className="story-badge red-badge">SEVERE NOISE (10 dB)</span>
            </div>
          )}
        </div>

        <div className="card-body" style={{ padding: 16 }}>
          <div className="roadmap-levels-row">
            {currentSweepMeta.levels.map((lvl, idx, arr) => {
              // Find matching point in result if available
              const pt = result?.points.find((p) => Math.abs(p.level - lvl) < 1e-4);

              return (
                <React.Fragment key={lvl}>
                  <div className="roadmap-level-box">
                    <div className="level-badge-header">
                      <span className="level-number">LEVEL {idx + 1}</span>
                      <span className="level-val">
                        {sweepType === 'coupling'
                          ? `±${(lvl * 100).toFixed(0)}%`
                          : sweepType === 'steering'
                          ? `${lvl > 0 ? `+${lvl}°` : `${lvl}°`}`
                          : `${lvl} dB`}
                      </span>
                    </div>

                    <div className="level-metrics-summary">
                      <div className="level-metric-item">
                        <span className="lm-label">Samples:</span>
                        <span className="lm-val">{numSamples}</span>
                      </div>
                      <div className="level-metric-item">
                        <span className="lm-label">Analytical NMSE:</span>
                        <span className="lm-val cyan">{pt ? `${pt.analytical_nmse_db.toFixed(2)} dB` : '—'}</span>
                      </div>
                      <div className="level-metric-item">
                        <span className="lm-label">PINN-Cal V4:</span>
                        <span className="lm-val orange">
                          {pt?.pinn_nmse_db != null ? `${pt.pinn_nmse_db.toFixed(2)} dB` : '—'}
                        </span>
                      </div>
                    </div>
                  </div>
                  {idx < arr.length - 1 && <div className="roadmap-arrow">→</div>}
                </React.Fragment>
              );
            })}
          </div>
        </div>
      </section>

      {/* 9 & 10. PRIMARY & SECONDARY ROBUSTNESS CHARTS (SIDE-BY-SIDE ON DESKTOP) */}
      <section className="robustness-charts-section">
        <div className="robustness-charts-grid">
          {/* Primary Chart: NMSE vs Parameter */}
          <div className="card robustness-card robustness-chart-card">
            <div className="card-header robustness-card-header">
              <h3 className="robustness-card-title">
                CALIBRATION ACCURACY: NMSE vs {currentSweepMeta.name}
              </h3>
              <span className="header-meta">Normalized MSE (dB) · Lower is better</span>
            </div>
            <div className="card-body robustness-chart-body">
              {!result ? (
                <div className="awaiting-panel-large">
                  <span>Run experiment to generate accuracy curves...</span>
                </div>
              ) : (
                <Plot
                  data={[anaNmseTrace, pinnNmseTrace].filter(Boolean) as any}
                  layout={plotLayout('NMSE (dB)')}
                  useResizeHandler
                  style={{ width: '100%', height: '100%' }}
                  config={{ displayModeBar: false, responsive: true }}
                />
              )}
            </div>
          </div>

          {/* Secondary Chart: Mean Phase Error vs Parameter */}
          <div className="card robustness-card robustness-chart-card">
            <div className="card-header robustness-card-header">
              <h3 className="robustness-card-title">
                MEAN PHASE ERROR vs {currentSweepMeta.name}
              </h3>
              <span className="header-meta">Mean absolute transceiver phase error (degrees)</span>
            </div>
            <div className="card-body robustness-chart-body">
              {!result ? (
                <div className="awaiting-panel-large">
                  <span>Run experiment to generate phase error curves...</span>
                </div>
              ) : (
                <Plot
                  data={[phaseAnaTrace, phasePinnTrace].filter(Boolean) as any}
                  layout={plotLayout('Phase Error (°)')}
                  useResizeHandler
                  style={{ width: '100%', height: '100%' }}
                  config={{ displayModeBar: false, responsive: true }}
                />
              )}
            </div>
          </div>
        </div>
      </section>

      {/* 14. SCIENTIFIC INTERPRETATION PANEL */}
      <section className="card robustness-card robustness-interpretation-card">
        <div className="card-header robustness-card-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="header-icon blue-icon">💡</span>
            <h3 className="robustness-card-title">WHAT DOES THIS STRESS TEST TELL US?</h3>
          </div>
          <span className="header-meta">Data-Driven Characterization Under Stress</span>
        </div>
        <div className="card-body" style={{ padding: 18 }}>
          <div className="interpretation-content">
            <p className="interp-p">
              In this Monte Carlo stress test, operating hardware conditions are intentionally degraded to evaluate the numerical stability and regularisation limits of each calibration approach:
            </p>
            <ul className="interp-list">
              <li>
                <strong>Pristine / Low-Noise Conditions:</strong> When thermal receiver noise is minimal and coupling deviations are small, the linear analytical baseline provides an effective direct inverse of the nominal Z-impedance matrix.
              </li>
              <li>
                <strong>Severe Noise & Mismatch Conditions:</strong> As noise increases (e.g. SNR drops toward 10 dB) or coupling deviations grow, unregularised analytical matrix inversion suffers from numerical noise amplification. The <strong>PINN-Cal V4</strong> architecture combines a Tikhonov preconditioner (δ = 0.15) with bounded neural residual outputs (α = 0.50), regularising the solution and maintaining stable bounded error.
              </li>
              <li>
                <strong>Neutral Evaluation:</strong> Both methods are evaluated across identical fresh realizations without ranking badges or superiority claims, characterizing the real engineering operating boundaries.
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* 11 & 12. METHOD COMPARISON & FRESH MONTE CARLO TABLE */}
      <section className="card robustness-card robustness-table-card">
        <div className="card-header robustness-card-header">
          <div>
            <h3 className="robustness-card-title">TABULATED MONTE CARLO EVALUATION DATA</h3>
            <span className="header-meta">
              Sample-averaged aggregate metrics for each evaluated stress level ({numSamples} samples/level)
            </span>
          </div>
          <ConceptTrigger conceptKey="nmse" label="NMSE Definition" />
        </div>

        <div className="card-body" style={{ padding: 0 }}>
          {!result ? (
            <div className="awaiting-panel-large" style={{ height: 160 }}>
              <span>Awaiting Monte Carlo evaluation execution...</span>
            </div>
          ) : (
            <div className="table-responsive">
              <table className="robustness-data-table">
                <thead>
                  <tr>
                    <th>Level ({currentSweepMeta.unit})</th>
                    <th style={{ color: '#0891b2' }}>Analytical NMSE</th>
                    <th style={{ color: '#0891b2' }}>Analytical Phase</th>
                    <th style={{ color: '#0891b2' }}>Analytical Gain</th>
                    <th style={{ color: '#0891b2' }}>Analytical Relative</th>
                    <th style={{ color: '#ea580c' }}>PINN-Cal V4 NMSE</th>
                    <th style={{ color: '#ea580c' }}>PINN Phase</th>
                    <th style={{ color: '#ea580c' }}>PINN Gain</th>
                    <th style={{ color: '#ea580c' }}>PINN Relative</th>
                  </tr>
                </thead>
                <tbody>
                  {result.points.map((p: RobustnessPoint) => (
                    <tr key={p.level}>
                      <td className="level-cell">
                        {sweepType === 'coupling'
                          ? `±${(p.level * 100).toFixed(0)}%`
                          : sweepType === 'steering'
                          ? `${p.level > 0 ? `+${p.level.toFixed(1)}°` : `${p.level.toFixed(1)}°`}`
                          : `${p.level.toFixed(1)} dB`}
                      </td>
                      <td className="ana-val">{p.analytical_nmse_db.toFixed(3)} dB</td>
                      <td className="mono-val">{p.analytical_phase_error_deg.toFixed(2)}°</td>
                      <td className="mono-val">{p.analytical_gain_error_db.toFixed(2)} dB</td>
                      <td className="mono-val">{p.analytical_relative_error_pct.toFixed(2)}%</td>
                      <td className="pinn-val">
                        {p.pinn_nmse_db != null ? `${p.pinn_nmse_db.toFixed(3)} dB` : '—'}
                      </td>
                      <td className="mono-val">
                        {p.pinn_phase_error_deg != null ? `${p.pinn_phase_error_deg.toFixed(2)}°` : '—'}
                      </td>
                      <td className="mono-val">
                        {p.pinn_gain_error_db != null ? `${p.pinn_gain_error_db.toFixed(2)} dB` : '—'}
                      </td>
                      <td className="mono-val">
                        {p.pinn_relative_error_pct != null ? `${p.pinn_relative_error_pct.toFixed(2)}%` : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      {/* 6 & 28. FROZEN VALIDATED REFERENCE BENCHMARK NOTE */}
      <footer className="frozen-reference-note-footer">
        <div className="frozen-ref-card">
          <div className="frozen-ref-header">
            <span className="frozen-lock-icon">🔒</span>
            <strong className="frozen-ref-title">
              DISTINCTION: FROZEN VALIDATED REFERENCE BENCHMARK vs. FRESH MONTE CARLO
            </strong>
          </div>
          <p className="frozen-ref-body">
            The data displayed in the charts and table above represents a <strong>fresh live Monte Carlo evaluation</strong> generated during this session.
            It is distinct from the <strong>frozen reference benchmark</strong> (Step 64 final blind holdout: 5,000 samples across 10 conditions, closed for model selection: NMSE = −14.81 dB, Phase Error = 4.47°, Gain Error = 1.00 dB).
            Historical benchmark numbers are not mixed with or substituted for fresh experimental realizations.
          </p>
        </div>
      </footer>
    </div>
  );
};
