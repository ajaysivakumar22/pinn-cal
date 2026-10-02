import React from 'react';
import { useAppStore } from '../../store/appStore';

export const SimulationStory: React.FC = () => {
  const { simulationResult, simulateState, config, superMode } = useAppStore();

  if (simulateState !== 'success' || !simulationResult) {
    return null;
  }

  const { metrics, beam, calibration, pinn_available } = simulationResult;
  const targetAngle = config.target_angle_deg;
  const peakAngle = metrics.beam_peak_angle_deg;
  const pointingErr = Math.abs(metrics.pointing_error_deg);
  const correlation = metrics.beam_correlation ?? 0;
  const anaNmse = calibration.analytical.nmse_db;
  const pinnNmse = calibration.pinn?.nmse_db;

  const uncalPeak = beam.uncalibrated.peak_angle_deg;
  const uncalErr = Math.abs(beam.uncalibrated.pointing_error_deg);

  // Factual phrasing based strictly on computed metrics
  const uncalPointingSentence =
    uncalErr < 0.05
      ? `The uncalibrated beam peak remained aligned at ${uncalPeak > 0 ? `+${uncalPeak.toFixed(1)}°` : `${uncalPeak.toFixed(1)}°`}, producing a pointing error of ${uncalErr.toFixed(2)}°.`
      : `Hardware phase and gain errors shifted the uncalibrated beam peak to ${uncalPeak > 0 ? `+${uncalPeak.toFixed(1)}°` : `${uncalPeak.toFixed(1)}°`}, resulting in an uncalibrated pointing error of ${uncalErr.toFixed(2)}°.`;

  const restorationSentence =
    pointingErr < 0.05
      ? `Following calibration, the restored beam peak aligned with the target at ${peakAngle > 0 ? `+${peakAngle.toFixed(1)}°` : `${peakAngle.toFixed(1)}°`} (residual pointing error ${pointingErr.toFixed(2)}°).`
      : `Following calibration, the restored beam peaked at ${peakAngle > 0 ? `+${peakAngle.toFixed(1)}°` : `${peakAngle.toFixed(1)}°`}, reducing pointing error to ${pointingErr.toFixed(2)}°.`;

  return (
    <div
      className="card simulation-story-card"
      style={{
        background: '#ffffff',
        border: '1px solid #cbd5e1',
        borderRadius: 8,
        marginBottom: 20,
        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.03)',
      }}
    >
      <div
        className="card-header"
        style={{
          background: 'linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%)',
          borderBottom: '1px solid #e2e8f0',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '10px 18px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: '1.125rem' }}>📋</span>
          <h3
            style={{
              fontSize: '1rem',
              fontWeight: 800,
              color: '#0f172a',
              letterSpacing: '-0.01em',
              margin: 0,
            }}
          >
            WHAT JUST HAPPENED?
          </h3>
          {superMode && (
            <span
              style={{
                fontSize: '0.6875rem',
                fontWeight: 700,
                color: '#854d0e',
                background: '#fef08a',
                padding: '2px 8px',
                borderRadius: 4,
              }}
            >
              ★ FRANKY REPORT
            </span>
          )}
        </div>
        <span style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)', fontWeight: 600 }}>
          Dynamically generated from current simulation realization
        </span>
      </div>

      <div
        className="card-body"
        style={{
          padding: '16px 20px',
          fontSize: '0.9375rem',
          lineHeight: 1.65,
          color: '#1e293b',
          display: 'flex',
          flexDirection: 'column',
          gap: 10,
        }}
      >
        <p style={{ margin: 0 }}>
          <strong>Command:</strong> The {config.num_elements}-element uniform linear array was commanded to steer beam
          energy toward <strong>{targetAngle > 0 ? `+${targetAngle.toFixed(1)}°` : `${targetAngle.toFixed(1)}°`}</strong> at{' '}
          <strong>{(config.frequency_hz / 1e9).toFixed(0)} GHz</strong> (0.50λ inter-element spacing).
        </p>

        <p style={{ margin: 0 }}>
          <strong>Impairment:</strong> Random element-level phase skews (<strong>σ = {config.phase_sigma_deg.toFixed(1)}°</strong>)
          and gain perturbations (<strong>σ = {config.gain_sigma_db.toFixed(1)} dB</strong>) degraded spatial excitation coherence. Mutual
          impedance coupling with <strong>±{(config.coupling_mismatch_fraction * 100).toFixed(0)}%</strong> uncertainty and thermal receiver
          noise (<strong>SNR = {config.snr_db.toFixed(0)} dB</strong>) were applied to the observed RF port currents.{' '}
          {uncalPointingSentence}
        </p>

        <p style={{ margin: 0 }}>
          <strong>Diagnosis & Calibration:</strong> Calibration algorithms processed the coupled RF observations to estimate
          channel impairments and compute inverse compensation weights ($C = H^{'{'}-1{'}'}$).
        </p>

        <p style={{ margin: 0 }}>
          <strong>Restoration & Verification:</strong> {restorationSentence} Restored beam correlation reached{' '}
          <strong>{correlation.toFixed(4)}</strong>.{' '}
          {pinn_available && pinnNmse !== undefined
            ? `On this realization, the analytical baseline achieved an NMSE of ${anaNmse.toFixed(2)} dB, while the PINN achieved ${pinnNmse.toFixed(2)} dB.`
            : `On this realization, the analytical baseline achieved an NMSE of ${anaNmse.toFixed(2)} dB.`}
        </p>
      </div>
    </div>
  );
};
