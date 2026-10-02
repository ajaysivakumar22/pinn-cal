import React, { useMemo } from 'react';
import { AsyncPlot as Plot } from '../components/common/AsyncPlot';
import { useAppStore } from '../store/appStore';
import { ConceptTrigger } from '../components/common/ConceptExplainer';

export const BeamLab: React.FC = () => {
  const { simulationResult, config, setConfig, runSimulation, simulateState } = useAppStore();

  const beam = simulationResult?.beam;
  const metrics = simulationResult?.metrics;
  const targetAngle = config.target_angle_deg;

  // Beam traces with rich styling
  const chartData = useMemo(() => {
    if (!beam) return [];

    const defs = [
      { key: 'ideal', label: 'Ideal Target Pattern', trace: beam.ideal, color: '#16a34a', width: 2, dash: 'dash' },
      { key: 'uncalibrated', label: 'Uncalibrated (Impaired)', trace: beam.uncalibrated, color: '#dc2626', width: 2.2, dash: 'solid' },
      { key: 'analytical', label: 'Analytical Baseline Restored', trace: beam.analytical, color: '#0891b2', width: 2.5, dash: 'solid' },
    ];

    if (beam.pinn && simulationResult?.pinn_available) {
      defs.push({ key: 'pinn', label: 'PINN Neural Restored', trace: beam.pinn, color: '#7c3aed', width: 2.5, dash: 'solid' });
    }

    return defs.map((d) => ({
      x: d.trace.angles_deg,
      y: d.trace.power_db,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: d.label,
      line: { color: d.color, width: d.width, dash: d.dash as any },
      hovertemplate: `<b>${d.label}</b><br>Angle: %{x:.1f}°<br>Power: %{y:.2f} dB<extra></extra>`,
    }));
  }, [beam, simulationResult?.pinn_available]);

  // Target angle marker
  const shapes = useMemo(() => {
    return [
      {
        type: 'line' as const,
        x0: targetAngle,
        x1: targetAngle,
        y0: -50,
        y1: 3,
        line: { color: '#0f172a', width: 2, dash: 'dot' },
      },
    ];
  }, [targetAngle]);

  const layout = useMemo(
    () => ({
      margin: { t: 25, r: 20, b: 45, l: 50 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { family: "'Inter', sans-serif", size: 11, color: '#475569' },
      legend: {
        x: 0.02,
        y: 0.04,
        bgcolor: 'rgba(255, 255, 255, 0.92)',
        bordercolor: '#e2e8f0',
        borderwidth: 1,
        font: { size: 10 },
      },
      xaxis: {
        title: { text: 'Far-Field Angle (θ in degrees)', font: { size: 11 } },
        range: [-90, 90],
        gridcolor: '#e2e8f0',
        tickfont: { size: 10 },
      },
      yaxis: {
        title: { text: 'Normalized Radiation Power (dB)', font: { size: 11 } },
        range: [-50, 3],
        gridcolor: '#e2e8f0',
        tickfont: { size: 10 },
      },
      shapes,
      hovermode: 'x unified' as const,
    }),
    [shapes]
  );

  return (
    <div className="beam-lab-page" style={{ maxWidth: 1200, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Page Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 10 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, margin: 0, color: 'var(--color-text-primary)' }}>
              BEAM LAB — 5G SPATIAL RADIATION BENCH
            </h2>
            <ConceptTrigger conceptKey="beamforming" label="Radiation" />
          </div>
          <p style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)', margin: '4px 0 0 0' }}>
            Investigate how collective 64-element wave interference synthesizes spatial beam directivity and how calibration cancels sidelobes.
          </p>
        </div>

        {/* Steering Quick Dial */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, background: '#ffffff', padding: '6px 14px', borderRadius: 8, border: '1px solid var(--color-border)' }}>
          <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-text-secondary)' }}>
            Command Steering:
          </label>
          <input
            type="range"
            min={-60}
            max={60}
            step={1}
            value={config.target_angle_deg}
            onChange={(e) => setConfig({ target_angle_deg: Number(e.target.value) })}
            style={{ width: 140 }}
          />
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem', fontWeight: 700, minWidth: 46 }}>
            {config.target_angle_deg > 0 ? `+${config.target_angle_deg}°` : `${config.target_angle_deg}°`}
          </span>
          <button
            className="btn btn-primary"
            onClick={runSimulation}
            disabled={simulateState === 'computing'}
            style={{ fontSize: '0.75rem', padding: '4px 10px' }}
          >
            {simulateState === 'computing' ? 'Synthesizing...' : 'Update Beam'}
          </button>
        </div>
      </div>

      {/* Physics Concept Flow Banner */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: 12,
          padding: '12px 16px',
          background: '#f8fafc',
          border: '1px solid var(--color-border)',
          borderRadius: 8,
          fontSize: '0.75rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: '1.25rem' }}>1️⃣</span>
          <div>
            <strong>ELEMENT EXCITATION:</strong>
            <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.6875rem' }}>
              Each patch radiator receives progressive phase delay a_n(θ) = exp(j k x_n sin θ).
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: '1.25rem' }}>2️⃣</span>
          <div>
            <strong>ARRAY INTERFERENCE:</strong>
            <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.6875rem' }}>
              Waves constructively reinforce toward angle θ, destructively canceling elsewhere.
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: '1.25rem' }}>3️⃣</span>
          <div>
            <strong>CALIBRATION IMPACT:</strong>
            <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.6875rem' }}>
              Phase/gain corrections sharpen half-power beamwidth and suppress sidelobe leakage.
            </div>
          </div>
        </div>
      </div>

      {/* Chart + Telemetry Row */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16 }}>
        {/* Main Radiation Pattern Plot */}
        <div className="card" style={{ height: 420 }}>
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h4 style={{ margin: 0, fontSize: '0.875rem' }}>Far-Field Radiation Pattern P(θ)</h4>
            <span style={{ fontSize: '0.6875rem', color: 'var(--color-text-tertiary)' }}>
              Normalized decibels (0 dB = peak isotropic gain)
            </span>
          </div>
          <div className="card-body" style={{ padding: 10, height: 360 }}>
            {!beam ? (
              <div className="awaiting-panel">
                <span>Awaiting simulation data...</span>
              </div>
            ) : (
              <Plot
                data={chartData}
                layout={layout}
                useResizeHandler
                style={{ width: '100%', height: '100%' }}
                config={{ displayModeBar: true, responsive: true }}
              />
            )}
          </div>
        </div>

        {/* Right: Beam Telemetry Cards */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div className="card">
            <div className="card-header">
              <h4 style={{ margin: 0, fontSize: '0.875rem' }}>Radiation Metrics</h4>
            </div>
            <div className="card-body" style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 10 }}>
              <MetricItem
                label="Target Angle"
                value={`${targetAngle.toFixed(1)}°`}
                desc="Commanded steering direction"
              />
              <MetricItem
                label="Analytical Beam Peak"
                value={`${(metrics?.beam_peak_angle_deg ?? 0).toFixed(2)}°`}
                desc={`Pointing error: ${Math.abs(metrics?.pointing_error_deg ?? 0).toFixed(3)}°`}
                color="var(--color-primary)"
              />
              <MetricItem
                label="Beam Correlation"
                value={(metrics?.beam_correlation ?? 0).toFixed(4)}
                desc="Similarity against ideal beam (1.0 = perfect)"
                color="#16a34a"
              />
              <MetricItem
                label="3dB Beamwidth (HPBW)"
                value={`${(metrics?.beamwidth_3db_deg ?? 0).toFixed(2)}°`}
                desc="Angular width at half-power (-3 dB)"
              />
              <MetricItem
                label="Peak Sidelobe Level (PSLL)"
                value={`${(metrics?.psll_db ?? 0).toFixed(1)} dB`}
                desc="Suppression ratio against adjacent leakage lobes"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const MetricItem: React.FC<{ label: string; value: string; desc: string; color?: string }> = ({
  label,
  value,
  desc,
  color,
}) => (
  <div style={{ padding: '8px 10px', background: '#f8fafc', borderRadius: 6, border: '1px solid #e2e8f0' }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
      <span style={{ fontSize: '0.6875rem', color: 'var(--color-text-secondary)', fontWeight: 600 }}>{label}</span>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9375rem', fontWeight: 700, color: color || '#0f172a' }}>
        {value}
      </span>
    </div>
    <div style={{ fontSize: '0.625rem', color: 'var(--color-text-tertiary)', marginTop: 2 }}>{desc}</div>
  </div>
);
