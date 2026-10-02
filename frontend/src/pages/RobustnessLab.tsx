import React, { useState } from 'react';
import { AsyncPlot as Plot } from '../components/common/AsyncPlot';
import { useAppStore } from '../store/appStore';
import type { SweepType, RobustnessPoint } from '../types/api';
import { ConceptTrigger } from '../components/common/ConceptExplainer';

const SWEEP_OPTIONS: { value: SweepType; label: string; unit: string; description: string }[] = [
  {
    value: 'snr',
    label: '1. Signal-to-Noise Ratio (SNR) Sweep',
    unit: 'dB',
    description: 'Evaluates calibration resilience as receiver thermal noise increases from 40 dB down to 10 dB.',
  },
  {
    value: 'coupling',
    label: '2. Coupling Uncertainty (ΔZ) Sweep',
    unit: 'fraction',
    description: 'Tests tolerance to hidden mutual impedance variations from 0% up to 50% mismatch.',
  },
  {
    value: 'steering',
    label: '3. Steering Sector Sweep',
    unit: '°',
    description: 'Measures angular consistency across wide scan angles (-50° to +50°).',
  },
];

const NUM_SAMPLES_OPTIONS = [20, 50, 100, 200];

export const RobustnessLab: React.FC = () => {
  const { robustnessResult, robustnessState, robustnessError, runRobustness } = useAppStore();
  const [sweepType, setSweepType] = useState<SweepType>('snr');
  const [numSamples, setNumSamples] = useState(50);
  const [seed, setSeed] = useState(4200);

  const isComputing = robustnessState === 'computing';

  const handleRun = () => {
    runRobustness({ sweep_type: sweepType, num_samples: numSamples, seed });
  };

  const result = robustnessResult;
  const currentSweepMeta = SWEEP_OPTIONS.find((o) => o.value === sweepType) ?? SWEEP_OPTIONS[0];

  const anaNmseTrace = result
    ? {
        x: result.points.map((p: RobustnessPoint) => p.level),
        y: result.points.map((p: RobustnessPoint) => p.analytical_nmse_db),
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        name: 'Analytical Baseline',
        line: { color: '#0891b2', width: 2.2 },
        marker: { size: 6, color: '#0891b2' },
        hovertemplate: `Analytical<br>${currentSweepMeta.unit}: %{x}<br>NMSE: %{y:.2f} dB<extra></extra>`,
      }
    : null;

  const pinnNmseTrace =
    result && result.pinn_available
      ? {
          x: result.points.map((p: RobustnessPoint) => p.level),
          y: result.points.map((p: RobustnessPoint) => p.pinn_nmse_db ?? null),
          type: 'scatter' as const,
          mode: 'lines+markers' as const,
          name: 'PINN Model',
          line: { color: '#7c3aed', width: 2.2 },
          marker: { size: 6, color: '#7c3aed' },
          hovertemplate: `PINN<br>${currentSweepMeta.unit}: %{x}<br>NMSE: %{y:.2f} dB<extra></extra>`,
        }
      : null;

  const phaseAnaTrace = result
    ? {
        x: result.points.map((p: RobustnessPoint) => p.level),
        y: result.points.map((p: RobustnessPoint) => p.analytical_phase_error_deg),
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        name: 'Analytical Phase Error',
        line: { color: '#0891b2', width: 2 },
        marker: { size: 6 },
        hovertemplate: `Analytical: %{y:.2f}°<extra></extra>`,
      }
    : null;

  const phasePinnTrace =
    result && result.pinn_available
      ? {
          x: result.points.map((p: RobustnessPoint) => p.level),
          y: result.points.map((p: RobustnessPoint) => p.pinn_phase_error_deg ?? null),
          type: 'scatter' as const,
          mode: 'lines+markers' as const,
          name: 'PINN Phase Error',
          line: { color: '#7c3aed', width: 2 },
          marker: { size: 6 },
          hovertemplate: `PINN: %{y:.2f}°<extra></extra>`,
        }
      : null;

  const plotLayout = (yTitle: string) => ({
    margin: { t: 20, r: 16, b: 40, l: 50 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: "'Inter', sans-serif", size: 10, color: '#475569' },
    legend: {
      orientation: 'h' as const,
      x: 0,
      y: 1.15,
      bgcolor: 'rgba(255,255,255,0.85)',
      font: { size: 10 },
    },
    xaxis: {
      title: { text: `${currentSweepMeta.label.split('.')[1]} (${currentSweepMeta.unit})`, font: { size: 10 } },
      gridcolor: '#e2e8f0',
      tickfont: { size: 9 },
    },
    yaxis: {
      title: { text: yTitle, font: { size: 10 } },
      gridcolor: '#e2e8f0',
      tickfont: { size: 9 },
    },
    hovermode: 'x unified' as const,
  });

  return (
    <div className="robustness-lab-page" style={{ maxWidth: 1200, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Page Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 10 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, margin: 0, color: 'var(--color-text-primary)' }}>
              ROBUSTNESS LAB — MONTE CARLO STRESS TEST
            </h2>
            <ConceptTrigger conceptKey="snr" label="Noise Model" />
          </div>
          <p style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)', margin: '4px 0 0 0' }}>
            Stress the calibration algorithms under degraded operating conditions with fresh backend Monte Carlo evaluations.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span
            style={{
              padding: '3px 8px',
              borderRadius: 4,
              background: '#f1f5f9',
              color: 'var(--color-text-secondary)',
              fontSize: '0.6875rem',
              fontWeight: 600,
            }}
          >
            Live Monte Carlo Mode
          </span>
        </div>
      </div>

      {/* Control Console */}
      <div className="card">
        <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h4 style={{ margin: 0, fontSize: '0.875rem' }}>Experiment Configuration</h4>
          <span style={{ fontSize: '0.6875rem', color: 'var(--color-text-tertiary)' }}>
            Fresh simulated realizations per level
          </span>
        </div>

        <div className="card-body" style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr auto', gap: 12, alignItems: 'flex-end' }}>
            <div>
              <label className="form-label">Sweep Parameter</label>
              <select
                className="form-select"
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
            </div>

            <div>
              <label className="form-label">Samples / Level</label>
              <select
                className="form-select"
                value={numSamples}
                onChange={(e) => setNumSamples(Number(e.target.value))}
                disabled={isComputing}
              >
                {NUM_SAMPLES_OPTIONS.map((n) => (
                  <option key={n} value={n}>
                    {n} realizations
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="form-label">Random Seed</label>
              <input
                type="number"
                className="form-input"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                disabled={isComputing}
              />
            </div>

            <button
              className="btn btn-primary"
              onClick={handleRun}
              disabled={isComputing}
              style={{ minWidth: 160, height: 36 }}
            >
              {isComputing ? 'Computing Sweep…' : '▶ START EXPERIMENT'}
            </button>
          </div>

          <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)', background: '#f8fafc', padding: '10px 14px', borderRadius: 6, border: '1px solid #e2e8f0', lineHeight: 1.5 }}>
            <strong>Active Stress Condition:</strong> {currentSweepMeta.description}
          </div>
        </div>
      </div>

      {robustnessError && (
        <div style={{ padding: '10px 14px', background: '#fee2e2', border: '1px solid #fecaca', borderRadius: 6, color: '#991b1b', fontSize: '0.875rem' }}>
          <strong>Experiment Error:</strong> {robustnessError}
        </div>
      )}

      {/* Pre-Experiment Roadmap / What Are We Testing Card (Eliminates Empty Space) */}
      {!result && !isComputing && (
        <div className="card" style={{ padding: '20px 24px', background: '#ffffff', border: '1px solid #cbd5e1' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
            <span style={{ fontSize: '1.25rem' }}>🧪</span>
            <h3 style={{ margin: 0, fontSize: '1.125rem', fontWeight: 800, color: '#0f172a' }}>
              WHAT ARE WE TESTING IN THIS MONTE CARLO STRESS TEST?
            </h3>
          </div>

          <p style={{ fontSize: '0.9375rem', lineHeight: 1.6, color: '#334155', margin: '0 0 16px 0' }}>
            {sweepType === 'snr' &&
              'We progressively degrade the signal-to-noise ratio from pristine laboratory conditions (40 dB) down to severe receiver noise (10 dB). This tests whether the analytical pseudoinverse amplifies measurement noise while checking the neural network regularization stability.'}
            {sweepType === 'coupling' &&
              'We introduce mismatch between the nominal mutual impedance model Z and hidden hardware coupling (from 0% up to 50%). This measures whether calibration is robust to unexpected electromagnetic cross-talk.'}
            {sweepType === 'steering' &&
              'We command the beam to scan across wide scan sectors from -50° to +50°. This verifies that calibration accuracy does not degrade near wide scan angles.'}
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--color-text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              EXPERIMENT LEVEL ROADMAP ({currentSweepMeta.unit})
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
              {sweepType === 'snr' &&
                [40, 30, 20, 15, 10].map((lvl, idx, arr) => (
                  <React.Fragment key={lvl}>
                    <div style={{ padding: '8px 16px', background: '#f1f5f9', border: '1px solid #cbd5e1', borderRadius: 6, fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '0.875rem', color: '#0f172a' }}>
                      {lvl} dB
                    </div>
                    {idx < arr.length - 1 && <span style={{ color: '#94a3b8', fontWeight: 800 }}>→</span>}
                  </React.Fragment>
                ))}
              {sweepType === 'coupling' &&
                [0, 0.1, 0.2, 0.3, 0.4, 0.5].map((lvl, idx, arr) => (
                  <React.Fragment key={lvl}>
                    <div style={{ padding: '8px 16px', background: '#f1f5f9', border: '1px solid #cbd5e1', borderRadius: 6, fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '0.875rem', color: '#0f172a' }}>
                      ±{(lvl * 100).toFixed(0)}%
                    </div>
                    {idx < arr.length - 1 && <span style={{ color: '#94a3b8', fontWeight: 800 }}>→</span>}
                  </React.Fragment>
                ))}
              {sweepType === 'steering' &&
                [-50, -30, 0, 30, 50].map((lvl, idx, arr) => (
                  <React.Fragment key={lvl}>
                    <div style={{ padding: '8px 16px', background: '#f1f5f9', border: '1px solid #cbd5e1', borderRadius: 6, fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '0.875rem', color: '#0f172a' }}>
                      {lvl > 0 ? `+${lvl}°` : `${lvl}°`}
                    </div>
                    {idx < arr.length - 1 && <span style={{ color: '#94a3b8', fontWeight: 800 }}>→</span>}
                  </React.Fragment>
                ))}
            </div>
          </div>

          <div style={{ marginTop: 18, display: 'flex', alignItems: 'center', gap: 12, fontSize: '0.8125rem', color: 'var(--color-text-secondary)' }}>
            <span style={{ padding: '2px 8px', background: '#e0f2fe', color: '#0369a1', borderRadius: 4, fontWeight: 700 }}>
              LIVE COMPUTATION
            </span>
            <span>Click <strong>▶ START EXPERIMENT</strong> to trigger fresh backend Monte Carlo calculation.</span>
          </div>
        </div>
      )}

      {/* Results View */}
      {result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Scientific Interpretation Callout */}
          <div
            style={{
              padding: '12px 16px',
              background: '#eff6ff',
              border: '1px solid #bfdbfe',
              borderRadius: 8,
              fontSize: '0.75rem',
              color: '#1e40af',
              lineHeight: 1.5,
            }}
          >
            <strong>Scientific Result Interpretation:</strong> This live experiment measures calibration error as operating conditions degrade.
            Notice that at high SNR (e.g. 30–40 dB), the analytical baseline achieves low NMSE. Under severe noise (SNR = 10 dB), analytical matrix
            inversion experiences noise amplification, while neural regularisation maintains stable bounded performance. Both methods are displayed
            neutrally without declaring a universal winner.
          </div>

          {/* Charts Row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div className="card">
              <div className="card-header">
                <h4 style={{ margin: 0, fontSize: '0.875rem' }}>Calibration Accuracy (NMSE in dB)</h4>
              </div>
              <div className="card-body" style={{ padding: 10, height: 260 }}>
                <Plot
                  data={[anaNmseTrace, pinnNmseTrace].filter(Boolean) as any}
                  layout={plotLayout('NMSE (dB)')}
                  useResizeHandler
                  style={{ width: '100%', height: '100%' }}
                  config={{ displayModeBar: false, responsive: true }}
                />
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <h4 style={{ margin: 0, fontSize: '0.875rem' }}>Mean Phase Error (Degrees)</h4>
              </div>
              <div className="card-body" style={{ padding: 10, height: 260 }}>
                <Plot
                  data={[phaseAnaTrace, phasePinnTrace].filter(Boolean) as any}
                  layout={plotLayout('Phase Error (°)')}
                  useResizeHandler
                  style={{ width: '100%', height: '100%' }}
                  config={{ displayModeBar: false, responsive: true }}
                />
              </div>
            </div>
          </div>

          {/* Numerical Results Table */}
          <div className="card">
            <div className="card-header">
              <h4 style={{ margin: 0, fontSize: '0.875rem' }}>Tabulated Fresh Monte Carlo Data</h4>
            </div>
            <div className="card-body" style={{ padding: 0 }}>
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th>Level ({currentSweepMeta.unit})</th>
                    <th style={{ color: 'var(--color-analytical)' }}>Analytical NMSE (dB)</th>
                    <th style={{ color: 'var(--color-analytical)' }}>Analytical Phase (°)</th>
                    <th style={{ color: 'var(--color-analytical)' }}>Analytical Gain (dB)</th>
                    {result.pinn_available && (
                      <>
                        <th style={{ color: 'var(--color-pinn)' }}>PINN NMSE (dB)</th>
                        <th style={{ color: 'var(--color-pinn)' }}>PINN Phase (°)</th>
                        <th style={{ color: 'var(--color-pinn)' }}>PINN Gain (dB)</th>
                      </>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {result.points.map((p: RobustnessPoint) => (
                    <tr key={p.level}>
                      <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{p.level}</td>
                      <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-analytical)' }}>{p.analytical_nmse_db.toFixed(3)}</td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{p.analytical_phase_error_deg.toFixed(2)}°</td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{p.analytical_gain_error_db.toFixed(2)} dB</td>
                      {result.pinn_available && (
                        <>
                          <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-pinn)', fontWeight: 600 }}>
                            {p.pinn_nmse_db?.toFixed(3) ?? '—'}
                          </td>
                          <td style={{ fontFamily: 'var(--font-mono)' }}>{p.pinn_phase_error_deg?.toFixed(2) ?? '—'}°</td>
                          <td style={{ fontFamily: 'var(--font-mono)' }}>{p.pinn_gain_error_db?.toFixed(2) ?? '—'} dB</td>
                        </>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
