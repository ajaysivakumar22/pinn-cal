import React from 'react';
import { useAppStore } from '../../store/appStore';

function fmt(v: number | null | undefined, decimals = 3, suffix = ''): string {
  if (v === null || v === undefined || !isFinite(v)) return '—';
  return `${v.toFixed(decimals)}${suffix}`;
}

function nmseColor(nmse: number | null): string {
  if (nmse === null) return 'var(--color-text-tertiary)';
  if (nmse < -12) return 'var(--color-success)';
  if (nmse < -8) return 'var(--color-warning)';
  return 'var(--color-error)';
}

export const MetricsPanel: React.FC = () => {
  const { simulationResult, simulateState, config } = useAppStore();
  const metrics = simulationResult?.metrics;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        padding: '0 4px',
      }}
    >
      <div className="section-title">Live Metrics</div>

      {/* Beam metrics */}
      <div className="section-title" style={{ marginTop: 4 }}>Beam</div>
      <MetricCard
        id="target-angle"
        label="Target Angle"
        value={fmt(config.target_angle_deg, 1, '°')}
      />
      <MetricCard
        id="peak-angle"
        label="Beam Peak (Ana)"
        value={fmt(metrics?.beam_peak_angle_deg, 2, '°')}
        loading={simulateState === 'computing'}
      />
      <MetricCard
        id="pointing-error"
        label="Pointing Error"
        value={fmt(metrics?.pointing_error_deg, 3, '°')}
        loading={simulateState === 'computing'}
      />
      <MetricCard
        id="beam-correlation"
        label="Beam Correlation"
        value={metrics?.beam_correlation != null
          ? metrics.beam_correlation.toFixed(4)
          : '—'
        }
        loading={simulateState === 'computing'}
      />
      <MetricCard
        id="beamwidth"
        label="Beamwidth (−3dB)"
        value={fmt(metrics?.beamwidth_3db_deg, 2, '°')}
        loading={simulateState === 'computing'}
      />
      <MetricCard
        id="psll"
        label="PSLL"
        value={fmt(metrics?.psll_db, 1, ' dB')}
        loading={simulateState === 'computing'}
      />

      {/* Calibration metrics */}
      <div className="section-title" style={{ marginTop: 8 }}>Calibration</div>
      <MetricCard
        id="ana-nmse"
        label="Analytical NMSE"
        value={fmt(metrics?.analytical_nmse_db, 2, ' dB')}
        valueColor={nmseColor(metrics?.analytical_nmse_db ?? null)}
        sub="Analytical baseline"
        loading={simulateState === 'computing'}
      />
      <MetricCard
        id="pinn-nmse"
        label="PINN NMSE"
        value={fmt(metrics?.pinn_nmse_db ?? null, 2, ' dB')}
        valueColor={nmseColor(metrics?.pinn_nmse_db ?? null)}
        sub="PINN model"
        loading={simulateState === 'computing'}
        unavailable={!simulationResult?.pinn_available}
      />
      <MetricCard
        id="cal-phase-err"
        label="Phase Error (Ana / PINN)"
        value={
          simulationResult?.calibration
            ? `${fmt(simulationResult.calibration.analytical.mean_phase_error_deg, 2, '°')} / ${simulationResult.calibration.pinn ? fmt(simulationResult.calibration.pinn.mean_phase_error_deg, 2, '°') : '—'}`
            : '—'
        }
        loading={simulateState === 'computing'}
      />
      <MetricCard
        id="cal-gain-err"
        label="Gain Error (Ana / PINN)"
        value={
          simulationResult?.calibration
            ? `${fmt(simulationResult.calibration.analytical.mean_gain_error_db, 2, ' dB')} / ${simulationResult.calibration.pinn ? fmt(simulationResult.calibration.pinn.mean_gain_error_db, 2, ' dB') : '—'}`
            : '—'
        }
        loading={simulateState === 'computing'}
      />
      <MetricCard
        id="cal-rel-err"
        label="Relative Error (Ana / PINN)"
        value={
          simulationResult?.calibration
            ? `${fmt(simulationResult.calibration.analytical.mean_relative_error_pct, 1, '%')} / ${simulationResult.calibration.pinn ? fmt(simulationResult.calibration.pinn.mean_relative_error_pct, 1, '%') : '—'}`
            : '—'
        }
        loading={simulateState === 'computing'}
      />

      {/* Config echo */}
      <div className="section-title" style={{ marginTop: 8 }}>Configuration</div>
      <MetricCard id="n-elements" label="Elements" value={`${config.num_elements}`} />
      <MetricCard id="freq" label="Frequency" value={`${(config.frequency_hz / 1e9).toFixed(0)} GHz`} />
      <MetricCard id="snr" label="SNR" value={`${config.snr_db.toFixed(0)} dB`} />
      <MetricCard id="phase-sig" label="Phase σ" value={`${config.phase_sigma_deg.toFixed(0)}°`} />
      <MetricCard id="gain-sig" label="Gain σ" value={`${config.gain_sigma_db.toFixed(2)} dB`} />
      <MetricCard id="coupling" label="Coupling ΔZ" value={`±${(config.coupling_mismatch_fraction * 100).toFixed(0)}%`} />
    </div>
  );
};

const MetricCard: React.FC<{
  id: string;
  label: string;
  value: string;
  sub?: string;
  loading?: boolean;
  valueColor?: string;
  unavailable?: boolean;
}> = ({ id, label, value, sub, loading, valueColor, unavailable }) => (
  <div className="metric-card" id={id}>
    <div className="metric-label">{label}</div>
    {loading ? (
      <div className="skeleton" style={{ height: 20, width: '60%', marginTop: 2 }} />
    ) : (
      <div
        className="metric-value"
        style={{ color: valueColor ?? (unavailable ? 'var(--color-text-tertiary)' : undefined) }}
      >
        {unavailable ? 'N/A' : value}
      </div>
    )}
    {sub && !loading && (
      <div className="metric-sub">{sub}</div>
    )}
  </div>
);
