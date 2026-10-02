import React, { useMemo } from 'react';
import { AsyncPlot as Plot } from '../components/common/AsyncPlot';
import { useAppStore, type CalibrationExplorerMetric } from '../store/appStore';
import type { ElementState } from '../types/api';
import { ConceptTrigger } from '../components/common/ConceptExplainer';

const METRIC_OPTIONS: { id: CalibrationExplorerMetric; label: string; unit: string; description: string }[] = [
  { id: 'phase_error', label: 'Phase Error (Ground Truth)', unit: '°', description: 'Simulated transceiver phase error.' },
  { id: 'gain_error', label: 'Gain Error (Ground Truth)', unit: 'dB', description: 'Simulated transceiver gain error.' },
  { id: 'pinn_phase_est', label: 'PINN Phase Estimate', unit: '°', description: 'Model-estimated phase error (-angle(C)).' },
  { id: 'pinn_gain_est', label: 'PINN Gain Estimate', unit: 'dB', description: 'Model-estimated gain error (-20*log10(|C|)).' },
  { id: 'phase_residual', label: 'Phase Residual (Error)', unit: '°', description: 'Difference between ground truth and estimate.' },
  { id: 'gain_residual', label: 'Gain Residual (Error)', unit: 'dB', description: 'Difference between ground truth and estimate.' },
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
  const { simulationResult, calibrationMetric, setCalibrationMetric, selectedElement, setSelectedElement } = useAppStore();

  const elements = simulationResult?.elements;

  const currentMetricDef = METRIC_OPTIONS.find((m) => m.id === calibrationMetric) ?? METRIC_OPTIONS[0];

  // Statistical calculations from actual data
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
        marker: { color: 'rgba(8, 145, 178, 0.7)', line: { color: '#0891b2', width: 1 } },
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
      font: { family: "'Inter', sans-serif", size: 10, color: '#475569' },
      xaxis: {
        title: { text: `${currentMetricDef.label} (${currentMetricDef.unit})`, font: { size: 10 } },
        gridcolor: '#e2e8f0',
        tickfont: { size: 9 },
      },
      yaxis: {
        title: { text: 'Element Count', font: { size: 10 } },
        gridcolor: '#e2e8f0',
        tickfont: { size: 9 },
      },
      bargap: 0.05,
    }),
    [currentMetricDef]
  );

  return (
    <div className="calibration-explorer-page" style={{ maxWidth: 1200, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Page Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 10 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, margin: 0, color: 'var(--color-text-primary)' }}>
              CALIBRATION EXPLORER
            </h2>
            <ConceptTrigger conceptKey="nmse" label="NMSE" />
          </div>
          <p style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)', margin: '4px 0 0 0' }}>
            Inspect element-level calibration accuracy, error distributions, and residual metrics across the 64-element array.
          </p>
        </div>

        {/* Metric Selector Dropdown / Pills */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-text-secondary)' }}>
            Active Diagnostic Map:
          </label>
          <select
            className="form-select"
            value={calibrationMetric}
            onChange={(e) => setCalibrationMetric(e.target.value as CalibrationExplorerMetric)}
            style={{ fontSize: '0.75rem', padding: '5px 12px', minWidth: 220 }}
          >
            {METRIC_OPTIONS.map((opt) => (
              <option key={opt.id} value={opt.id}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Grid: Left Matrix + Right Distribution & Telemetry */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 16 }}>
        {/* Left: 64-Element Diagnostic Matrix */}
        <div className="card">
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h4 style={{ margin: 0, fontSize: '0.875rem' }}>64-Element Heatmap Grid</h4>
              <span style={{ fontSize: '0.6875rem', color: 'var(--color-text-tertiary)' }}>
                {currentMetricDef.description}
              </span>
            </div>
            <span style={{ fontSize: '0.6875rem', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
              Unit: {currentMetricDef.unit || 'dimensionless'}
            </span>
          </div>

          <div className="card-body" style={{ padding: 16 }}>
            {!elements ? (
              <div className="awaiting-panel">
                <span>Awaiting simulation data...</span>
              </div>
            ) : (
              <div>
                {/* 8x8 Grid of antenna elements */}
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
                    const isSelected = selectedElement === idx;

                    // Compute normalized color based on value & maxAbs
                    const maxVal = Math.max(stats.maxAbs, 1e-4);
                    const ratio = Math.max(-1, Math.min(1, val / maxVal));

                    // Red for negative, Cyan for positive, neutral slate for near 0
                    let bg = '#f1f5f9';
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
                          border: isSelected ? '2px solid #0f172a' : '1px solid rgba(0,0,0,0.08)',
                          borderRadius: 6,
                          padding: '8px 4px',
                          textAlign: 'center',
                          cursor: 'pointer',
                          transition: 'all 0.15s ease',
                          boxShadow: isSelected ? '0 0 0 2px var(--color-primary)' : 'none',
                        }}
                        title={`Element #${idx + 1}: ${val.toFixed(2)} ${currentMetricDef.unit}`}
                      >
                        <div style={{ fontSize: '0.625rem', color: 'var(--color-text-tertiary)', fontWeight: 600 }}>
                          #{idx + 1}
                        </div>
                        <div
                          style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: '0.6875rem',
                            fontWeight: 700,
                            color: '#0f172a',
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
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.6875rem', color: 'var(--color-text-secondary)' }}>
                  <span>Negative ({(-stats.maxAbs).toFixed(1)}{currentMetricDef.unit})</span>
                  <div
                    style={{
                      width: 140,
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

        {/* Right: Statistical Summary + Distribution Histogram */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Statistical Metrics Card */}
          <div className="card">
            <div className="card-header">
              <h4 style={{ margin: 0, fontSize: '0.875rem' }}>Population Statistics</h4>
            </div>
            <div className="card-body" style={{ padding: 14 }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
                <div style={{ padding: '8px 10px', background: '#f8fafc', borderRadius: 6, border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '0.625rem', color: 'var(--color-text-tertiary)', fontWeight: 600 }}>SAMPLE MEAN (μ)</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1rem', fontWeight: 700, color: '#0f172a', marginTop: 2 }}>
                    {stats.mean > 0 ? `+${stats.mean.toFixed(2)}` : stats.mean.toFixed(2)} {currentMetricDef.unit}
                  </div>
                </div>

                <div style={{ padding: '8px 10px', background: '#f8fafc', borderRadius: 6, border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '0.625rem', color: 'var(--color-text-tertiary)', fontWeight: 600 }}>STANDARD DEV (σ)</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1rem', fontWeight: 700, color: 'var(--color-primary)', marginTop: 2 }}>
                    {stats.std.toFixed(2)} {currentMetricDef.unit}
                  </div>
                </div>

                <div style={{ padding: '8px 10px', background: '#f8fafc', borderRadius: 6, border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '0.625rem', color: 'var(--color-text-tertiary)', fontWeight: 600 }}>MAX PEAK DEVIATION</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1rem', fontWeight: 700, color: '#dc2626', marginTop: 2 }}>
                    {stats.maxAbs.toFixed(2)} {currentMetricDef.unit}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Histogram Card */}
          <div className="card" style={{ flex: 1 }}>
            <div className="card-header">
              <h4 style={{ margin: 0, fontSize: '0.875rem' }}>Error Distribution Histogram</h4>
            </div>
            <div className="card-body" style={{ padding: 10, height: 200 }}>
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
    </div>
  );
};
