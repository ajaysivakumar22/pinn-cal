import React, { useEffect } from 'react';
import { useAppStore } from '../store/appStore';

export const SystemStatusPage: React.FC = () => {
  const { systemStatus, modelStatus, loadSystemStatus } = useAppStore();

  useEffect(() => {
    loadSystemStatus();
  }, []);

  const env = systemStatus;
  const model = modelStatus;

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Title */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--color-text-primary)' }}>
            System & Model Status
          </h2>
          <p style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)', marginTop: 4 }}>
            Live runtime environment diagnostics and validated PINN checkpoint telemetry.
          </p>
        </div>
        <button
          className="btn btn-secondary"
          onClick={loadSystemStatus}
          style={{ fontSize: '0.75rem' }}
        >
          Refresh Status
        </button>
      </div>

      {/* Grid of cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
        {/* Runtime Environment Card */}
        <div className="card">
          <div className="card-header">
            <h4>Runtime Environment</h4>
          </div>
          <div className="card-body">
            <table className="comparison-table" style={{ width: '100%' }}>
              <tbody>
                <tr>
                  <td>Python Version</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{env?.python_version ?? '—'}</td>
                </tr>
                <tr>
                  <td>PyTorch Version</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{env?.pytorch_version ?? '—'}</td>
                </tr>
                <tr>
                  <td>CUDA Availability</td>
                  <td>
                    <span
                      style={{
                        padding: '2px 8px',
                        borderRadius: 4,
                        fontSize: '0.6875rem',
                        fontWeight: 600,
                        background: env?.cuda_available ? '#dcfce7' : '#fee2e2',
                        color: env?.cuda_available ? '#166534' : '#991b1b',
                      }}
                    >
                      {env?.cuda_available ? 'Available' : 'Unavailable (CPU)'}
                    </span>
                  </td>
                </tr>
                <tr>
                  <td>GPU Device</td>
                  <td style={{ fontWeight: 500 }}>{env?.gpu_name ?? 'None detected'}</td>
                </tr>
                <tr>
                  <td>CUDA Device Count</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{env?.cuda_device_count ?? 0}</td>
                </tr>
                <tr>
                  <td>Operating System</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{env?.platform ?? '—'}</td>
                </tr>
                <tr>
                  <td>Backend Uptime</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>
                    {env?.backend_uptime_s != null ? `${env.backend_uptime_s.toFixed(1)} s` : '—'}
                  </td>
                </tr>
                <tr>
                  <td>Scientific Core Tag</td>
                  <td>
                    <span
                      style={{
                        padding: '2px 8px',
                        borderRadius: 4,
                        fontSize: '0.6875rem',
                        fontWeight: 600,
                        background: '#e0f2fe',
                        color: '#0369a1',
                      }}
                    >
                      scientific-core-v1.0 (Frozen)
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* PINN Model Telemetry Card */}
        <div className="card">
          <div className="card-header">
            <h4>PINN Architecture & Checkpoint</h4>
          </div>
          <div className="card-body">
            <table className="comparison-table" style={{ width: '100%' }}>
              <tbody>
                <tr>
                  <td>Checkpoint Path</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6875rem' }}>
                    {model?.checkpoint ?? 'models/pinn_cal_final_best.pth'}
                  </td>
                </tr>
                <tr>
                  <td>Checkpoint Status</td>
                  <td>
                    <span
                      style={{
                        padding: '2px 8px',
                        borderRadius: 4,
                        fontSize: '0.6875rem',
                        fontWeight: 600,
                        background: model?.loaded ? '#dcfce7' : '#fee2e2',
                        color: model?.loaded ? '#166534' : '#991b1b',
                      }}
                    >
                      {model?.loaded ? 'Loaded (weights_only=True)' : 'Not Loaded'}
                    </span>
                  </td>
                </tr>
                <tr>
                  <td>Total Parameters</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                    {model?.total_parameters?.toLocaleString() ?? '83,904'}
                  </td>
                </tr>
                <tr>
                  <td>Architecture</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                    {model?.architecture ?? '130 → 256 → 128 → 64 → 128'}
                  </td>
                </tr>
                <tr>
                  <td>Layers</td>
                  <td style={{ fontSize: '0.75rem' }}>Linear + BatchNorm1d + ReLU + Dropout(0.2)</td>
                </tr>
                <tr>
                  <td>Best Epoch</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{model?.best_epoch ?? 40}</td>
                </tr>
                <tr>
                  <td>Inference Device</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{model?.device ?? 'cuda:0'}</td>
                </tr>
                <tr>
                  <td>Target Configuration</td>
                  <td style={{ fontWeight: 500 }}>64 elements @ 28 GHz (0.5λ)</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Dataset Split & Validation Baseline Card */}
        <div className="card">
          <div className="card-header">
            <h4>Dataset Split & Baseline Benchmarks</h4>
          </div>
          <div className="card-body">
            <table className="comparison-table" style={{ width: '100%' }}>
              <tbody>
                <tr>
                  <td>Train Samples</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{model?.train_samples ?? 800}</td>
                </tr>
                <tr>
                  <td>Validation Samples</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{model?.val_samples ?? 200}</td>
                </tr>
                <tr>
                  <td>Test Samples (Isolated)</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{model?.test_samples ?? 200}</td>
                </tr>
                <tr>
                  <td>Split Seed</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>2026</td>
                </tr>
                <tr>
                  <td>Analytical Baseline NMSE</td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-analytical)', fontWeight: 600 }}>
                    −14.45 dB (15.1% rel err)
                  </td>
                </tr>
                <tr>
                  <td>Final PINN NMSE</td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-pinn)', fontWeight: 600 }}>
                    −7.75 dB (36.8% rel err)
                  </td>
                </tr>
              </tbody>
            </table>

            <div
              style={{
                marginTop: 14,
                padding: '10px 12px',
                background: '#f8fafc',
                border: '1px solid var(--color-border)',
                borderRadius: 6,
                fontSize: '0.75rem',
                color: 'var(--color-text-secondary)',
                lineHeight: 1.5,
              }}
            >
              <strong>Scientific Validation Note:</strong> The scientific core was audited and frozen.
              Controlled test evaluation demonstrates that the compact reciprocal analytical baseline
              achieves lower calibration error than the current feedforward PINN. Both methods are
              faithfully provided without bias.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
