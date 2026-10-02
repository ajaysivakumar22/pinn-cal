import React, { useMemo } from 'react';
import { AsyncPlot as Plot } from '../common/AsyncPlot';
import { useAppStore } from '../../store/appStore';

const TRACE_COLORS = {
  ideal:          '#16a34a',  // green
  uncalibrated:   '#dc2626',  // red
  analytical:     '#0891b2',  // cyan
  pinn:           '#7c3aed',  // purple
};

const TRACE_DASH: Record<string, string> = {
  ideal:         'dash',
  uncalibrated:  'solid',
  analytical:    'solid',
  pinn:          'solid',
};

export const BeamPatternChart: React.FC = () => {
  const { simulationResult, simulateState, config } = useAppStore();

  const beam = simulationResult?.beam;

  const traces = useMemo(() => {
    if (!beam) return [];

    const defs = [
      { key: 'ideal',        label: 'Ideal',          trace: beam.ideal },
      { key: 'uncalibrated', label: 'Uncalibrated',   trace: beam.uncalibrated },
      { key: 'analytical',   label: 'Analytical',     trace: beam.analytical },
      { key: 'pinn',         label: 'PINN',           trace: beam.pinn },
    ];

    return defs
      .filter((d) => d.trace !== null)
      .map((d) => ({
        x: d.trace!.angles_deg,
        y: d.trace!.power_db,
        type: 'scattergl' as const,
        mode: 'lines' as const,
        name: d.label,
        line: {
          color: TRACE_COLORS[d.key as keyof typeof TRACE_COLORS],
          width: d.key === 'ideal' ? 1.5 : 2,
          dash: TRACE_DASH[d.key],
        },
        hovertemplate: `<b>${d.label}</b><br>Angle: %{x:.1f}°<br>Power: %{y:.2f} dB<extra></extra>`,
      }));
  }, [beam]);

  // Vertical line at target angle
  const shapes = useMemo(() => {
    if (!beam) return [];
    return [{
      type: 'line' as const,
      x0: config.target_angle_deg,
      x1: config.target_angle_deg,
      y0: -60,
      y1: 2,
      line: { color: 'var(--color-text-tertiary)', width: 1, dash: 'dot' as const },
    }];
  }, [config.target_angle_deg, !!beam]);

  const layout = useMemo(() => ({
    margin: { t: 20, r: 16, b: 50, l: 50 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: "'Inter', sans-serif", size: 11, color: '#475569' },
    legend: {
      x: 0.02, y: 0.02,
      bgcolor: 'rgba(255,255,255,0.85)',
      bordercolor: '#e2e6ee',
      borderwidth: 1,
      font: { size: 11 },
    },
    xaxis: {
      title: { text: 'Angle (degrees)', font: { size: 11 } },
      range: [-90, 90],
      zeroline: true,
      zerolinecolor: '#e2e6ee',
      gridcolor: '#e2e6ee',
      tickfont: { size: 10 },
    },
    yaxis: {
      title: { text: 'Normalized Power (dB)', font: { size: 11 } },
      range: [-60, 3],
      zeroline: false,
      gridcolor: '#e2e6ee',
      tickfont: { size: 10 },
    },
    shapes,
    hovermode: 'x unified' as const,
  }), [shapes]);

  const isEmpty = !beam && simulateState !== 'computing';

  return (
    <div className="card" style={{ flex: 1 }}>
      <div className="card-header">
        <h4>Beam Pattern</h4>
        {beam && (
          <div style={{ display: 'flex', gap: 12, fontSize: '0.6875rem', color: 'var(--color-text-secondary)' }}>
            {beam.analytical && (
              <span>
                Peak: <strong className="mono">{beam.analytical.peak_angle_deg.toFixed(1)}°</strong>
              </span>
            )}
            {beam.analytical?.beamwidth_3db_deg && (
              <span>
                BW<sub>3dB</sub>: <strong className="mono">{beam.analytical.beamwidth_3db_deg.toFixed(2)}°</strong>
              </span>
            )}
            {beam.analytical?.psll_db !== null && beam.analytical?.psll_db !== undefined && (
              <span>
                PSLL: <strong className="mono">{beam.analytical.psll_db.toFixed(1)} dB</strong>
              </span>
            )}
          </div>
        )}
      </div>
      <div className="card-body" style={{ padding: '12px 8px' }}>
        {isEmpty && (
          <div className="awaiting-panel" style={{ minHeight: 240 }}>
            <svg width={40} height={40} viewBox="0 0 40 40" fill="none">
              <path d="M20 8 L20 32 M8 20 Q20 4 32 20 Q20 36 8 20Z"
                stroke="#94a3b8" strokeWidth={1.5} fill="none" />
            </svg>
            <span>Run a simulation to see beam patterns</span>
          </div>
        )}
        {simulateState === 'computing' && (
          <div className="awaiting-panel" style={{ minHeight: 240 }}>
            <span>Computing beam patterns…</span>
          </div>
        )}
        {beam && traces.length > 0 && (
          <Plot
            data={traces}
            layout={layout}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%', height: 300 }}
            useResizeHandler
          />
        )}
      </div>
    </div>
  );
};
