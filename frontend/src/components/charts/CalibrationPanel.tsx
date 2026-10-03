import React from 'react';
import { useAppStore } from '../../store/appStore';

export const CalibrationPanel: React.FC = () => {
  const { simulationResult } = useAppStore();

  if (!simulationResult) {
    return (
      <div className="card">
        <div className="card-header"><h4>Calibration Comparison</h4></div>
        <div className="card-body">
          <div className="awaiting-panel" style={{ padding: 24 }}>
            <span style={{ fontSize: '0.8125rem' }}>Awaiting simulation</span>
          </div>
        </div>
      </div>
    );
  }

  const { calibration, pinn_available, pinn_compatibility_message } = simulationResult;
  const ana = calibration.analytical;
  const pinn = calibration.pinn;

  return (
    <div className="card">
      <div className="card-header">
        <h4>Calibration Comparison</h4>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <div
            style={{
              width: 8, height: 8, borderRadius: '50%',
              background: pinn_available ? 'var(--color-pinn)' : 'var(--color-text-tertiary)',
            }}
          />
          <span style={{ fontSize: '0.6875rem', color: 'var(--color-text-tertiary)' }}>
            PINN-Cal V4 {pinn_available ? 'active' : 'unavailable'}
          </span>
        </div>
      </div>
      <div className="card-body" style={{ padding: 0 }}>

        {/* Live simulation comparison summary */}
        <div style={{ padding: '8px 16px', borderBottom: '1px solid var(--color-border)' }}>
          <div className="honesty-notice" style={{ background: '#f8fafc', color: 'var(--color-text-secondary)', borderLeft: '3px solid #0284c7' }}>
            Live Simulation Results: Analytical Baseline NMSE = {ana.nmse_db.toFixed(2)} dB{pinn ? ` · PINN-Cal V4 NMSE = ${pinn.nmse_db.toFixed(2)} dB` : ''}.
            Dynamic metrics evaluated across 64 elements for the commanded steering angle.
          </div>
        </div>

        <table className="comparison-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th style={{ color: 'var(--color-analytical)' }}>Analytical Baseline</th>
              {pinn_available && pinn && (
                <th style={{ color: 'var(--color-pinn)' }}>PINN-Cal V4</th>
              )}
            </tr>
          </thead>
          <tbody>
            <MetricRow
              label="NMSE"
              unit="dB"
              ana={ana.nmse_db.toFixed(3)}
              pinn={pinn?.nmse_db.toFixed(3) ?? null}
              pinnAvail={pinn_available && !!pinn}
            />
            <MetricRow
              label="Phase Error"
              unit="°"
              ana={ana.mean_phase_error_deg.toFixed(3)}
              pinn={pinn?.mean_phase_error_deg.toFixed(3) ?? null}
              pinnAvail={pinn_available && !!pinn}
            />
            <MetricRow
              label="Gain Error"
              unit="dB"
              ana={ana.mean_gain_error_db.toFixed(3)}
              pinn={pinn?.mean_gain_error_db.toFixed(3) ?? null}
              pinnAvail={pinn_available && !!pinn}
            />
            <MetricRow
              label="Relative Error"
              unit="%"
              ana={ana.mean_relative_error_pct.toFixed(2)}
              pinn={pinn?.mean_relative_error_pct.toFixed(2) ?? null}
              pinnAvail={pinn_available && !!pinn}
            />
            {ana.current_residual_pct !== null && (
              <MetricRow
                label="Current Residual"
                unit="%"
                ana={ana.current_residual_pct?.toFixed(2) ?? '—'}
                pinn={pinn?.current_residual_pct?.toFixed(2) ?? null}
                pinnAvail={pinn_available && !!pinn}
              />
            )}
          </tbody>
        </table>

        {!pinn_available && (
          <div
            style={{
              padding: '8px 16px',
              borderTop: '1px solid var(--color-border)',
              fontSize: '0.6875rem',
              color: 'var(--color-text-tertiary)',
            }}
          >
            {pinn_compatibility_message}
          </div>
        )}
      </div>
    </div>
  );
};

const MetricRow: React.FC<{
  label: string;
  unit: string;
  ana: string;
  pinn: string | null;
  pinnAvail: boolean;
}> = ({ label, unit, ana, pinn, pinnAvail }) => (
  <tr>
    <td style={{ fontFamily: 'var(--font-sans)', color: 'var(--color-text-secondary)', fontSize: '0.8125rem' }}>
      {label} ({unit})
    </td>
    <td>{ana}</td>
    {pinnAvail && <td>{pinn ?? '—'}</td>}
  </tr>
);
