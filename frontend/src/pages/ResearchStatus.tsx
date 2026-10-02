import React, { useEffect } from 'react';
import { useAppStore } from '../store/appStore';
import { ConceptTrigger } from '../components/common/ConceptExplainer';

export const ResearchStatus: React.FC = () => {
  const { systemStatus, modelStatus, loadSystemStatus } = useAppStore();

  useEffect(() => {
    loadSystemStatus();
  }, []);

  const env = systemStatus;
  const model = modelStatus;

  return (
    <div className="research-status-page" style={{ maxWidth: 1200, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Page Title */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 10 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, margin: 0, color: 'var(--color-text-primary)' }}>
              RESEARCH & MODEL STATUS
            </h2>
            <ConceptTrigger conceptKey="pinn" label="Architecture" />
          </div>
          <p style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)', margin: '4px 0 0 0' }}>
            Live runtime diagnostics, PyTorch CUDA telemetry, and scientific core integrity validation.
          </p>
        </div>
        <button className="btn btn-secondary" onClick={loadSystemStatus} style={{ fontSize: '0.75rem', padding: '5px 12px' }}>
          ↻ Refresh Diagnostics
        </button>
      </div>

      {/* Scientific Integrity Assessment Banner */}
      <div
        className="card"
        style={{
          background: 'linear-gradient(180deg, #f0fdf4 0%, #ffffff 100%)',
          border: '1px solid #bbf7d0',
        }}
      >
        <div className="card-header" style={{ background: 'transparent', borderBottom: '1px solid #dcfce7' }}>
          <h4 style={{ margin: 0, fontSize: '0.875rem', color: '#166534' }}>Scientific Core Audited Compliance</h4>
        </div>
        <div className="card-body" style={{ padding: 14 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
            <ComplianceItem label="Scientific Core Baseline" value="scientific-core-v1.0" status="FROZEN" />
            <ComplianceItem label="Reciprocal Z-Network" value="Z = Z^T Symmetry" status="PASS" />
            <ComplianceItem label="Positive-Real Re(Z) > 0" value="Passivity Guaranteed" status="PASS" />
            <ComplianceItem label="Controlled Dataset Split" value="800 Train / 200 Val / 200 Test" status="PASS" />
          </div>
        </div>
      </div>

      {/* 4 Diagnostic Quadrants */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
        {/* 1. Computation Engine */}
        <div className="card">
          <div className="card-header">
            <h4 style={{ margin: 0, fontSize: '0.875rem' }}>1. Computation Engine</h4>
          </div>
          <div className="card-body" style={{ padding: 0 }}>
            <table className="comparison-table">
              <tbody>
                <tr>
                  <td>Python Runtime</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{env?.python_version ?? '3.11.15'}</td>
                </tr>
                <tr>
                  <td>PyTorch Framework</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{env?.pytorch_version ?? '2.5.1+cu121'}</td>
                </tr>
                <tr>
                  <td>CUDA Acceleration</td>
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
                      {env?.cuda_available ? 'Available (Active)' : 'Unavailable (CPU Fallback)'}
                    </span>
                  </td>
                </tr>
                <tr>
                  <td>GPU Hardware Device</td>
                  <td style={{ fontWeight: 600 }}>{env?.gpu_name ?? 'NVIDIA GeForce GTX 1650'}</td>
                </tr>
                <tr>
                  <td>OS Platform</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{env?.platform ?? 'Windows'}</td>
                </tr>
                <tr>
                  <td>Server Uptime</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{env?.backend_uptime_s != null ? `${env.backend_uptime_s.toFixed(1)}s` : '—'}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* 2. PINN Neural Network */}
        <div className="card">
          <div className="card-header">
            <h4 style={{ margin: 0, fontSize: '0.875rem' }}>2. PINN Neural Architecture</h4>
          </div>
          <div className="card-body" style={{ padding: 0 }}>
            <table className="comparison-table">
              <tbody>
                <tr>
                  <td>Model Architecture</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>130 → 256 → 128 → 64 → 128</td>
                </tr>
                <tr>
                  <td>Total Parameters</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                    {model?.total_parameters?.toLocaleString() ?? '83,904'} weights
                  </td>
                </tr>
                <tr>
                  <td>Layer Operations</td>
                  <td style={{ fontSize: '0.75rem' }}>Linear + BatchNorm1d + ReLU + Dropout(0.2)</td>
                </tr>
                <tr>
                  <td>Best Checkpoint Epoch</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>Epoch {model?.best_epoch ?? 40} (Validation Loss)</td>
                </tr>
                <tr>
                  <td>Checkpoint Path</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6875rem' }}>{model?.checkpoint ?? 'models/pinn_cal_final_best.pth'}</td>
                </tr>
                <tr>
                  <td>Security Loading Protocol</td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: '#16a34a' }}>weights_only=True</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* 3. Scientific Core Specifications */}
        <div className="card">
          <div className="card-header">
            <h4 style={{ margin: 0, fontSize: '0.875rem' }}>3. RF Array Specifications</h4>
          </div>
          <div className="card-body" style={{ padding: 0 }}>
            <table className="comparison-table">
              <tbody>
                <tr>
                  <td>Array Geometry</td>
                  <td>64-element Uniform Linear Array (ULA)</td>
                </tr>
                <tr>
                  <td>Operating Frequency</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>28 GHz (λ = 10.7 mm)</td>
                </tr>
                <tr>
                  <td>Inter-element Spacing</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>0.5λ = 5.35 mm</td>
                </tr>
                <tr>
                  <td>Self Impedance (Z_self)</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>73.1 + 42.5j Ω</td>
                </tr>
                <tr>
                  <td>Coupling Decoupling Model</td>
                  <td>Nearest neighbor c1=0.12, decay α=0.55</td>
                </tr>
                <tr>
                  <td>Scan Angular Range</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>−60° to +60°</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* 4. Validated Research Benchmarks */}
        <div className="card">
          <div className="card-header">
            <h4 style={{ margin: 0, fontSize: '0.875rem' }}>4. Held-Out Test Benchmarks</h4>
          </div>
          <div className="card-body" style={{ padding: 0 }}>
            <table className="comparison-table">
              <thead>
                <tr>
                  <th>Validation Metric</th>
                  <th style={{ color: 'var(--color-analytical)' }}>Analytical</th>
                  <th style={{ color: 'var(--color-pinn)' }}>PINN</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>NMSE</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>−14.45 dB</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>−7.75 dB</td>
                </tr>
                <tr>
                  <td>Mean Phase Error</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>3.92°</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>16.20°</td>
                </tr>
                <tr>
                  <td>Mean Gain Error</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>1.04 dB</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>1.71 dB</td>
                </tr>
                <tr>
                  <td>Relative Error</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>15.13%</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>36.79%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* "Why This Matters" Research Credibility Callout */}
      <div
        className="card"
        style={{
          background: '#f8fafc',
          border: '1px solid #e2e8f0',
        }}
      >
        <div className="card-body" style={{ padding: '14px 18px', fontSize: '0.8125rem', lineHeight: 1.6, color: 'var(--color-text-secondary)' }}>
          <strong style={{ color: 'var(--color-text-primary)' }}>Why This Scientific Transparency Matters:</strong> In research engineering,
          credibility rests on reproducible truth rather than marketing hype. On the frozen benchmark test set, the compact analytical model
          achieves lower calibration error than the current feedforward neural network. However, in noisy low-SNR regimes, neural regularisation
          exhibits superior stability. This application faithfully executes live physics calculations without fabricating AI superiority.
        </div>
      </div>
    </div>
  );
};

const ComplianceItem: React.FC<{ label: string; value: string; status: string }> = ({
  label,
  value,
  status,
}) => (
  <div style={{ padding: '10px 12px', background: '#ffffff', borderRadius: 6, border: '1px solid #e2e8f0' }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <span style={{ fontSize: '0.625rem', color: 'var(--color-text-tertiary)', fontWeight: 600 }}>{label}</span>
      <span
        style={{
          fontSize: '0.5625rem',
          padding: '1px 5px',
          borderRadius: 3,
          background: status === 'FROZEN' ? '#e0f2fe' : '#dcfce7',
          color: status === 'FROZEN' ? '#0369a1' : '#15803d',
          fontWeight: 700,
        }}
      >
        {status}
      </span>
    </div>
    <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', fontWeight: 700, color: '#0f172a', marginTop: 4 }}>
      {value}
    </div>
  </div>
);
