import React, { useMemo } from 'react';
import { AsyncPlot as Plot } from '../common/AsyncPlot';
import { useAppStore } from '../../store/appStore';
import { ConceptTrigger } from '../common/ConceptExplainer';

export const BeforeAfterView: React.FC = () => {
  const { simulationResult, config } = useAppStore();

  const beam = simulationResult?.beam;
  const metrics = simulationResult?.metrics;
  const targetAngle = config.target_angle_deg;

  // Uncalibrated (Before) vs Calibrated (After) Plotly Traces
  const chartData = useMemo(() => {
    if (!beam) return [];

    const uncal = beam.uncalibrated;
    const ana = beam.analytical;
    const pinn = beam.pinn;
    const ideal = beam.ideal;

    const traces: any[] = [
      {
        x: ideal.angles_deg,
        y: ideal.power_db,
        type: 'scatter',
        mode: 'lines',
        name: 'Ideal Target Pattern',
        line: { color: '#94a3b8', width: 2, dash: 'dot' },
        hovertemplate: 'Ideal: %{y:.1f} dB at %{x:.1f}°<extra></extra>',
      },
      {
        x: uncal.angles_deg,
        y: uncal.power_db,
        type: 'scatter',
        mode: 'lines',
        name: 'BEFORE Calibration (Distorted Hardware)',
        line: { color: '#ef4444', width: 2.8 },
        hovertemplate: 'Before: %{y:.1f} dB at %{x:.1f}°<extra></extra>',
      },
      {
        x: ana.angles_deg,
        y: ana.power_db,
        type: 'scatter',
        mode: 'lines',
        name: 'AFTER: Analytical Calibrated',
        line: { color: '#0891b2', width: 2.5 },
        hovertemplate: 'Analytical: %{y:.1f} dB at %{x:.1f}°<extra></extra>',
      },
    ];

    if (pinn && simulationResult?.pinn_available) {
      traces.push({
        x: pinn.angles_deg,
        y: pinn.power_db,
        type: 'scatter',
        mode: 'lines',
        name: 'AFTER: PINN Calibrated',
        line: { color: '#7c3aed', width: 2.5 },
        hovertemplate: 'PINN: %{y:.1f} dB at %{x:.1f}°<extra></extra>',
      });
    }

    return traces;
  }, [beam, simulationResult?.pinn_available]);

  const shapes = useMemo(() => {
    return [
      {
        type: 'line' as const,
        x0: targetAngle,
        x1: targetAngle,
        y0: -50,
        y1: 2,
        line: { color: '#16a34a', width: 2, dash: 'dash' },
      },
    ];
  }, [targetAngle]);

  const annotations = useMemo(() => {
    return [
      {
        x: targetAngle,
        y: 1.2,
        xref: 'x' as const,
        yref: 'y' as const,
        text: `Target: ${targetAngle > 0 ? `+${targetAngle}°` : `${targetAngle}°`}`,
        showarrow: true,
        arrowhead: 2,
        arrowsize: 1,
        arrowwidth: 1.5,
        arrowcolor: '#16a34a',
        font: { family: "'JetBrains Mono', monospace", size: 11, color: '#166534' },
        bgcolor: '#dcfce7',
        bordercolor: '#86efac',
        borderwidth: 1,
        borderpad: 3,
      },
    ];
  }, [targetAngle]);

  const layout = useMemo(
    () => ({
      margin: { t: 30, r: 20, b: 45, l: 55 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { family: "'Inter', sans-serif", size: 12, color: '#334155' },
      legend: {
        orientation: 'h' as const,
        x: 0,
        y: 1.15,
        bgcolor: 'rgba(255, 255, 255, 0.95)',
        bordercolor: '#e2e8f0',
        borderwidth: 1,
        font: { size: 11, family: "'Inter', sans-serif" },
      },
      xaxis: {
        title: { text: 'Spatial Radiation Angle θ (degrees)', font: { size: 12, weight: 600 } },
        range: [-80, 80],
        gridcolor: '#e2e8f0',
        tickfont: { size: 11, family: "'JetBrains Mono', monospace" },
      },
      yaxis: {
        title: { text: 'Normalized Radiation Power (dB)', font: { size: 12, weight: 600 } },
        range: [-45, 3],
        gridcolor: '#e2e8f0',
        tickfont: { size: 11, family: "'JetBrains Mono', monospace" },
      },
      shapes,
      annotations,
      hovermode: 'x unified' as const,
    }),
    [shapes, annotations]
  );

  return (
    <div
      className="card before-after-card"
      style={{
        marginBottom: 20,
        background: '#ffffff',
        border: '1px solid #cbd5e1',
        borderRadius: 8,
      }}
    >
      <div
        className="card-header"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%)',
          borderBottom: '1px solid #e2e8f0',
          padding: '12px 18px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: '1.25rem' }}>📡</span>
          <div>
            <h3 style={{ fontSize: '1.0625rem', fontWeight: 800, margin: 0, color: '#0f172a', letterSpacing: '-0.01em' }}>
              BEAM RESTORATION: BEFORE → AFTER CALIBRATION
            </h3>
            <span style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)', fontWeight: 500 }}>
              Spatial power distribution synthesized across the far-field aperture (−80° to +80°)
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <ConceptTrigger conceptKey="beam_correlation" label="Correlation" />
          <ConceptTrigger conceptKey="psll" label="PSLL" />
        </div>
      </div>

      <div className="card-body" style={{ padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 14 }}>
        {/* Physical Interpretation Callout */}
        <div
          style={{
            padding: '10px 14px',
            background: '#f8fafc',
            borderLeft: '4px solid #0284c7',
            borderRadius: '0 6px 6px 0',
            fontSize: '0.875rem',
            color: 'var(--color-text-secondary)',
            lineHeight: 1.6,
          }}
        >
          <strong style={{ color: '#0f172a' }}>Why the Beam Restores:</strong> In an antenna array, every element emits
          waves that constructively reinforce in the commanded direction. Hardware errors and mutual coupling perturb these
          spatial phase relationships, deflecting the beam peak and spilling energy into sidelobes. Applying compensation
          weights (<span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>C = H⁻¹</span>) realigns the aperture
          excitation, bringing the main lobe back toward the target direction (<strong>{targetAngle > 0 ? `+${targetAngle}°` : `${targetAngle}°`}</strong>).
        </div>

        {/* Prominent Radiation Pattern Chart */}
        <div style={{ width: '100%', height: 350, position: 'relative' }}>
          {!beam ? (
            <div className="awaiting-panel" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <span style={{ fontSize: '0.9375rem', color: 'var(--color-text-tertiary)' }}>Awaiting simulation data...</span>
            </div>
          ) : (
            <Plot
              data={chartData}
              layout={layout}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
              config={{ displayModeBar: false, responsive: true }}
            />
          )}
        </div>

        {/* Quantified Telemetry Panel */}
        {beam && metrics && (
          <div className="beam-telemetry-panel">
            {/* Target Steering */}
            <div className="telemetry-block target">
              <span className="telemetry-label">TARGET DIRECTION</span>
              <span className="telemetry-val green">{targetAngle > 0 ? `+${targetAngle.toFixed(1)}°` : `${targetAngle.toFixed(1)}°`}</span>
              <span className="telemetry-sub">Commanded Heading</span>
            </div>

            {/* Before Peak */}
            <div className="telemetry-block before">
              <span className="telemetry-label">BEFORE PEAK</span>
              <span className="telemetry-val red">
                {beam.uncalibrated.peak_angle_deg > 0 ? `+${beam.uncalibrated.peak_angle_deg.toFixed(1)}°` : `${beam.uncalibrated.peak_angle_deg.toFixed(1)}°`}
              </span>
              <span className="telemetry-sub">Error: {Math.abs(beam.uncalibrated.pointing_error_deg).toFixed(2)}°</span>
            </div>

            {/* After Peak */}
            <div className="telemetry-block after">
              <span className="telemetry-label">AFTER PEAK</span>
              <span className="telemetry-val blue">
                {metrics.beam_peak_angle_deg > 0 ? `+${metrics.beam_peak_angle_deg.toFixed(1)}°` : `${metrics.beam_peak_angle_deg.toFixed(1)}°`}
              </span>
              <span className="telemetry-sub">Residual: {Math.abs(metrics.pointing_error_deg).toFixed(2)}°</span>
            </div>

            {/* Pointing Error Improvement */}
            <div className="telemetry-block">
              <span className="telemetry-label">POINTING ERROR (Δθ)</span>
              <span className="telemetry-val">
                {Math.abs(beam.uncalibrated.pointing_error_deg).toFixed(1)}° →{' '}
                <span style={{ color: '#10b981' }}>{Math.abs(metrics.pointing_error_deg).toFixed(2)}°</span>
              </span>
              <span className="telemetry-sub">Aperture Realignment</span>
            </div>

            {/* Beam Correlation Restored */}
            <div className="telemetry-block">
              <span className="telemetry-label">BEAM CORRELATION</span>
              <span className="telemetry-val green">
                {(metrics.beam_correlation ?? 0).toFixed(4)}
              </span>
              <span className="telemetry-sub">Similarity to Ideal</span>
            </div>

            {/* 3dB Beamwidth & PSLL */}
            <div className="telemetry-block">
              <span className="telemetry-label">BEAMWIDTH / PSLL</span>
              <span className="telemetry-val" style={{ fontSize: '1rem' }}>
                {(metrics.beamwidth_3db_deg ?? 1.8).toFixed(2)}° / {(metrics.psll_db ?? -13.2).toFixed(1)} dB
              </span>
              <span className="telemetry-sub">Array Resolution</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
