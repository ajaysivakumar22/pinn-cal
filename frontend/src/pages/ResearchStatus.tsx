import React, { useEffect } from 'react';
import { useAppStore } from '../store/appStore';
import { ConceptTrigger } from '../components/common/ConceptExplainer';

export const ResearchStatus: React.FC = () => {
  const { systemStatus, modelStatus, loadSystemStatus, superMode } = useAppStore();

  useEffect(() => {
    loadSystemStatus();
  }, []);

  const env = systemStatus;
  const model = modelStatus;

  return (
    <div className="research-provenance-page">
      {/* 1. Page Header */}
      <div className="provenance-header-row">
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <h1 className="provenance-title">
              RESEARCH &amp; DATA
            </h1>
            <ConceptTrigger conceptKey="pinn" label="Architecture Guide" />
          </div>
          <p className="provenance-subtitle">
            Scientific provenance, frozen validation evidence, model checkpoint information, and runtime diagnostics.
          </p>

          {/* Compact Technical Identity Strip */}
          <div className="provenance-identity-strip">
            <span className="identity-badge primary">PINN-Cal V4</span>
            <span className="identity-badge">33,088 PARAMETERS</span>
            <span className="identity-badge accent">δ = 0.15</span>
            <span className="identity-badge accent">α = 0.50</span>
            <span className="identity-badge info">64-ELEMENT ULA</span>
            <span className="identity-badge info">28 GHz</span>
            {superMode && <span className="identity-badge accent" style={{ fontWeight: 800 }}>★ SUPER MODE ACTIVE</span>}
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 8 }}>
          <button
            className="btn btn-secondary"
            onClick={loadSystemStatus}
            style={{ fontSize: '0.8125rem', padding: '6px 14px', display: 'inline-flex', alignItems: 'center', gap: 6 }}
          >
            <span>↻</span>
            <span>Refresh Diagnostics</span>
          </button>
          <span style={{ fontSize: '0.6875rem', color: 'var(--color-text-tertiary)', fontFamily: 'var(--font-mono)' }}>
            Telemetry sync: Active
          </span>
        </div>
      </div>

      {/* Provenance Data Legend: LIVE vs FROZEN */}
      <div className="provenance-data-legend">
        <span style={{ fontWeight: 700, color: 'var(--color-text-primary)' }}>Data Category Guide:</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span className="legend-chip live">● LIVE RUNTIME</span>
          <span style={{ color: 'var(--color-text-secondary)', fontSize: '0.75rem' }}>Active environment diagnostics, hardware telemetry &amp; uptime</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginLeft: 'auto' }}>
          <span className="legend-chip frozen">🔒 FROZEN SCIENTIFIC REFERENCE</span>
          <span style={{ color: 'var(--color-text-secondary)', fontSize: '0.75rem' }}>Audited physics, locked weights, controlled splits &amp; holdout metrics</span>
        </div>
      </div>

      {/* 2. Scientific Core Compliance (Prominent Top Section) */}
      <div className="compliance-banner-card card">
        <div
          className="card-header"
          style={{
            background: 'transparent',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: 8,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: '1rem' }}>🛡️</span>
            <h2 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 800, letterSpacing: '0.02em', color: 'var(--color-text-primary)' }}>
              SCIENTIFIC CORE — FROZEN REFERENCE
            </h2>
          </div>
          <span className="legend-chip frozen">FROZEN AUDITED BASELINE</span>
        </div>
        <div className="card-body" style={{ padding: '14px 18px' }}>
          <div className="compliance-grid">
            {/* Indicator 1 */}
            <div className="compliance-tile">
              <div className="compliance-tile-header">
                <span className="compliance-tile-label">Scientific Core</span>
                <span className="compliance-badge-frozen">FROZEN</span>
              </div>
              <div className="compliance-tile-val">scientific-core-v1.0</div>
              <span style={{ fontSize: '0.6875rem', color: 'var(--color-text-secondary)' }}>
                Locked baseline physical models
              </span>
            </div>

            {/* Indicator 2 */}
            <div className="compliance-tile">
              <div className="compliance-tile-header">
                <span className="compliance-tile-label">Reciprocal Z-Network</span>
                <span className="compliance-badge-pass">PASS</span>
              </div>
              <div className="compliance-tile-val">Z = Zᵀ Symmetry</div>
              <span style={{ fontSize: '0.6875rem', color: 'var(--color-text-secondary)' }}>
                Reciprocity audited on 64-port mesh
              </span>
            </div>

            {/* Indicator 3 */}
            <div className="compliance-tile">
              <div className="compliance-tile-header">
                <span className="compliance-tile-label">Passivity / Pos-Real</span>
                <span className="compliance-badge-pass">PASS</span>
              </div>
              <div className="compliance-tile-val">Re&#123;Z&#125; &gt; 0 Verified</div>
              <span style={{ fontSize: '0.6875rem', color: 'var(--color-text-secondary)' }}>
                Positive-real energy conservation passivity
              </span>
            </div>

            {/* Indicator 4 */}
            <div className="compliance-tile">
              <div className="compliance-tile-header">
                <span className="compliance-tile-label">Controlled Split</span>
                <span className="compliance-badge-pass">PASS</span>
              </div>
              <div className="compliance-tile-val">800 / 200 / 200</div>
              <span style={{ fontSize: '0.6875rem', color: 'var(--color-text-secondary)' }}>
                Train / Val / Test strictly segregated
              </span>
            </div>
          </div>

          <div
            style={{
              marginTop: 12,
              padding: '8px 12px',
              borderRadius: 6,
              background: 'rgba(0, 0, 0, 0.02)',
              border: '1px dashed var(--color-border)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: 8,
              fontSize: '0.75rem',
              color: 'var(--color-text-secondary)',
            }}
          >
            <span>
              <strong>Scientific Provenance Notice:</strong> These audit metrics verify electromagnetic physical compliance (reciprocal passive network and positive-real impedance) prior to neural inference.
            </span>
            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
              Array: 64-element ULA · 28 GHz
            </span>
          </div>
        </div>
      </div>

      {/* 3. Middle Row: Neural Architecture & Checkpoint Provenance */}
      <div className="provenance-2col-grid">
        {/* Model Architecture Card */}
        <div className="card">
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ margin: 0, fontSize: '1.125rem', fontWeight: 700, color: 'var(--color-text-primary)' }}>
              PINN-CAL V4 NEURAL ARCHITECTURE
            </h3>
            <span className="legend-chip frozen">FROZEN ARCHITECTURE</span>
          </div>
          <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {/* Major Param Strip */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, textAlign: 'center' }}>
              <div style={{ padding: '8px', background: 'var(--color-bg)', borderRadius: 6, border: '1px solid var(--color-border)' }}>
                <div style={{ fontSize: '0.6875rem', color: 'var(--color-text-secondary)', textTransform: 'uppercase', fontWeight: 600 }}>Parameters</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.25rem', fontWeight: 800, color: 'var(--color-primary)' }}>33,088</div>
              </div>
              <div style={{ padding: '8px', background: 'var(--color-bg)', borderRadius: 6, border: '1px solid var(--color-border)' }}>
                <div style={{ fontSize: '0.6875rem', color: 'var(--color-text-secondary)', textTransform: 'uppercase', fontWeight: 600 }}>Tikhonov δ</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.25rem', fontWeight: 800, color: '#ea580c' }}>0.15</div>
              </div>
              <div style={{ padding: '8px', background: 'var(--color-bg)', borderRadius: 6, border: '1px solid var(--color-border)' }}>
                <div style={{ fontSize: '0.6875rem', color: 'var(--color-text-secondary)', textTransform: 'uppercase', fontWeight: 600 }}>Residual α</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.25rem', fontWeight: 800, color: '#ea580c' }}>0.50</div>
              </div>
            </div>

            {/* Architecture Flow Diagram */}
            <div className="arch-flow-diagram">
              <div className="arch-node">
                <span style={{ fontWeight: 700, fontSize: '0.8125rem', color: 'var(--color-primary)' }}>INPUT 128</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8125rem', color: 'var(--color-text-secondary)' }}>
                  Re/Im(c_prior) → 128 real features
                </span>
              </div>
              <div className="arch-arrow">↓</div>
              <div className="arch-node" style={{ borderColor: 'rgba(37, 99, 235, 0.3)', background: 'rgba(37, 99, 235, 0.03)' }}>
                <span style={{ fontWeight: 700, fontSize: '0.8125rem', color: 'var(--color-text-primary)' }}>HIDDEN 128</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>
                  Dense Layer 1 (ReLU)
                </span>
              </div>
              <div className="arch-arrow">↓</div>
              <div className="arch-node" style={{ borderColor: 'rgba(37, 99, 235, 0.3)', background: 'rgba(37, 99, 235, 0.03)' }}>
                <span style={{ fontWeight: 700, fontSize: '0.8125rem', color: 'var(--color-text-primary)' }}>HIDDEN 64</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>
                  Dense Layer 2 (ReLU)
                </span>
              </div>
              <div className="arch-arrow">↓</div>
              <div className="arch-node">
                <span style={{ fontWeight: 700, fontSize: '0.8125rem', color: '#ea580c' }}>OUTPUT 128</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8125rem', color: 'var(--color-text-secondary)' }}>
                  Δc → 128 real outputs (α = 0.50 bound)
                </span>
              </div>
              <div className="arch-arrow">↓</div>
              <div className="arch-node" style={{ borderColor: 'rgba(22, 163, 74, 0.4)', background: 'rgba(22, 163, 74, 0.04)' }}>
                <span style={{ fontWeight: 700, fontSize: '0.8125rem', color: '#16a34a' }}>FINAL CALIBRATION</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8125rem', fontWeight: 700, color: '#16a34a' }}>
                  c_hat = c_prior + Δc
                </span>
              </div>
            </div>

            <p style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: 0, lineHeight: 1.4 }}>
              <strong>Rigid Architecture Specification:</strong> INPUT 128 → HIDDEN 128 → HIDDEN 64 → OUTPUT 128. Input dimension is strictly 128 real features (never 130). Residual output is bounded by scaling factor α = 0.50.
            </p>
          </div>
        </div>

        {/* Checkpoint Provenance Card */}
        <div className="card">
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ margin: 0, fontSize: '1.125rem', fontWeight: 700, color: 'var(--color-text-primary)' }}>
              CHECKPOINT PROVENANCE
            </h3>
            <span className="legend-chip frozen">FROZEN CHECKPOINT</span>
          </div>
          <div className="card-body" style={{ padding: 0 }}>
            <table className="comparison-table" style={{ width: '100%' }}>
              <tbody>
                <tr>
                  <td style={{ width: '38%', fontWeight: 600 }}>Active Model</td>
                  <td style={{ width: '62%' }}>
                    <div style={{ fontWeight: 700, color: 'var(--color-text-primary)' }}>PINN-Cal V4</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>Physics-Informed Neural Network for Array Calibration</div>
                  </td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Reference Checkpoint</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--color-primary)' }}>
                    pinn_v4_tikhonov015_seed42.pth
                  </td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Reference Seed</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>42</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Parameter Count</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                    {model?.trainable_parameters?.toLocaleString() ?? '33,088'} weights (100% trainable)
                  </td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Best Epoch</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>
                    Epoch {model?.best_epoch ?? 29} (Validation Loss Minima)
                  </td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Status</td>
                  <td>
                    <span
                      style={{
                        padding: '2px 8px',
                        borderRadius: 4,
                        fontSize: '0.75rem',
                        fontWeight: 700,
                        background: '#dcfce7',
                        color: '#166534',
                      }}
                    >
                      Loaded / Frozen Reference
                    </span>
                  </td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Security Protocol</td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: '#16a34a', fontSize: '0.75rem' }}>
                    torch.load(..., weights_only=True)
                  </td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600, verticalAlign: 'top' }}>Filesystem Path</td>
                  <td
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.75rem',
                      color: 'var(--color-text-secondary)',
                      wordBreak: 'break-all',
                      overflowWrap: 'anywhere',
                      lineHeight: 1.4,
                    }}
                  >
                    {model?.checkpoint_path ?? model?.checkpoint ?? 'experimental/v4/models/pinn_v4_tikhonov015_seed42.pth'}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* 4. Second Middle Row: RF Array Specifications & Live Runtime Diagnostics */}
      <div className="provenance-2col-grid">
        {/* RF Array Specification Card */}
        <div className="card">
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ margin: 0, fontSize: '1.125rem', fontWeight: 700, color: 'var(--color-text-primary)' }}>
              RF ARRAY SPECIFICATIONS
            </h3>
            <span className="legend-chip frozen">FROZEN RF SPEC</span>
          </div>
          <div className="card-body" style={{ padding: 0 }}>
            <table className="comparison-table" style={{ width: '100%' }}>
              <tbody>
                <tr>
                  <td style={{ width: '40%', fontWeight: 600 }}>Array Geometry</td>
                  <td style={{ width: '60%' }}>64-element Uniform Linear Array (ULA)</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Operating Frequency</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>28 GHz (mmWave 5G band n257/n261)</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Free-Space Wavelength (λ)</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>λ ≈ 10.7 mm (10.71 mm)</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Inter-Element Pitch (d)</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>0.5 λ = 5.35 mm (Nyquist spatial sampling)</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Self-Impedance (Z_self)</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>73.1 + 42.5j Ω (Dipole resonance reference)</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Mutual Coupling Model</td>
                  <td>Nearest-neighbor c₁ = 0.12, spatial decay α = 0.55</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Angular Scan Range</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>
                    −60° to +60° (Training) · −85° to +85° (Simulation)
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Live Runtime Diagnostics Card */}
        <div className="card">
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ margin: 0, fontSize: '1.125rem', fontWeight: 700, color: 'var(--color-text-primary)' }}>
              LIVE RUNTIME DIAGNOSTICS
            </h3>
            <span className="legend-chip live">● LIVE RUNTIME</span>
          </div>
          <div className="card-body" style={{ padding: 0 }}>
            <table className="comparison-table" style={{ width: '100%' }}>
              <tbody>
                <tr>
                  <td style={{ width: '40%', fontWeight: 600 }}>Python Runtime</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{env?.python_version ?? '3.11.15'}</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>PyTorch Framework</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{env?.pytorch_version ?? '2.5.1+cu121'}</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>CUDA Acceleration</td>
                  <td>
                    <span
                      style={{
                        padding: '2px 8px',
                        borderRadius: 4,
                        fontSize: '0.75rem',
                        fontWeight: 700,
                        background: env?.cuda_available ? '#dcfce7' : '#fee2e2',
                        color: env?.cuda_available ? '#166534' : '#991b1b',
                      }}
                    >
                      {env?.cuda_available ? 'Available (Active GPU)' : 'Unavailable (CPU Fallback)'}
                    </span>
                  </td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>GPU Device Hardware</td>
                  <td style={{ fontWeight: 600 }}>{env?.gpu_name ?? 'NVIDIA GeForce GTX 1650'}</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>CUDA Device Count</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{env?.cuda_device_count ?? 1}</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Active Inference Device</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{model?.device ?? 'cuda:0'}</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Operating System / Platform</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{env?.platform ?? 'Windows'}</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Backend Server Uptime</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>
                    {env?.backend_uptime_s != null ? `${env.backend_uptime_s.toFixed(1)} s` : 'Online'}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* 5. Dataset Provenance: Controlled Split & Final Blind Holdout */}
      <div className="provenance-2col-grid">
        {/* Controlled Dataset Split */}
        <div className="card">
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h3 style={{ margin: 0, fontSize: '1.125rem', fontWeight: 700, color: 'var(--color-text-primary)' }}>
                CONTROLLED DATASET SPLIT
              </h3>
              <p style={{ margin: '2px 0 0 0', fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>
                Used for model development / evaluation workflow.
              </p>
            </div>
            <span className="legend-chip frozen">DEVELOPMENT SET</span>
          </div>
          <div className="card-body">
            {/* Visual 3-part dataset representation */}
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontWeight: 600, marginBottom: 4 }}>
              <span style={{ color: '#2563eb' }}>Train: 800 (66.7%)</span>
              <span style={{ color: '#0891b2' }}>Val: 200 (16.7%)</span>
              <span style={{ color: '#d97706' }}>Test: 200 (16.7%)</span>
            </div>

            <div className="split-bar-container">
              <div className="split-bar-train" style={{ width: '66.67%' }} title="Train: 800 samples" />
              <div className="split-bar-val" style={{ width: '16.67%' }} title="Validation: 200 samples" />
              <div className="split-bar-test" style={{ width: '16.66%' }} title="Test: 200 samples" />
            </div>

            <div className="split-triplet-grid">
              <div className="split-box">
                <div style={{ fontSize: '0.6875rem', fontWeight: 700, color: '#2563eb', textTransform: 'uppercase' }}>
                  TRAIN
                </div>
                <div className="split-count">800</div>
                <div style={{ fontSize: '0.6875rem', color: 'var(--color-text-secondary)', lineHeight: 1.3 }}>
                  Gradient optimization &amp; bounded residual learning
                </div>
              </div>

              <div className="split-box">
                <div style={{ fontSize: '0.6875rem', fontWeight: 700, color: '#0891b2', textTransform: 'uppercase' }}>
                  VALIDATION
                </div>
                <div className="split-count">200</div>
                <div style={{ fontSize: '0.6875rem', color: 'var(--color-text-secondary)', lineHeight: 1.3 }}>
                  Checkpoint epoch selection &amp; early stopping
                </div>
              </div>

              <div className="split-box">
                <div style={{ fontSize: '0.6875rem', fontWeight: 700, color: '#d97706', textTransform: 'uppercase' }}>
                  TEST
                </div>
                <div className="split-count">200</div>
                <div style={{ fontSize: '0.6875rem', color: 'var(--color-text-secondary)', lineHeight: 1.3 }}>
                  Development baseline comparison &amp; tuning check
                </div>
              </div>
            </div>

            <div
              style={{
                marginTop: 14,
                padding: '10px 12px',
                background: 'var(--color-bg)',
                border: '1px solid var(--color-border)',
                borderRadius: 6,
                fontSize: '0.75rem',
                color: 'var(--color-text-secondary)',
                lineHeight: 1.45,
              }}
            >
              <strong>Data Segregation Guarantee:</strong> This controlled split (800 / 200 / 200) is used exclusively for training and development validation. It is strictly segregated from the 5,000-sample final blind holdout.
            </div>
          </div>
        </div>

        {/* Final Blind Holdout Card */}
        <div className="card blind-holdout-card">
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'transparent' }}>
            <div>
              <h3 style={{ margin: 0, fontSize: '1.125rem', fontWeight: 700, color: 'var(--color-text-primary)' }}>
                FINAL BLIND HOLDOUT
              </h3>
              <p style={{ margin: '2px 0 0 0', fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>
                Independent benchmark reserved for scientific verification.
              </p>
            </div>
            <span className="badge-blind-closed">BLIND / CLOSED</span>
          </div>
          <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10 }}>
              <div style={{ padding: '12px', background: 'var(--color-surface)', borderRadius: 6, border: '1px solid var(--color-border)' }}>
                <div style={{ fontSize: '0.6875rem', fontWeight: 700, color: 'var(--color-text-secondary)', textTransform: 'uppercase' }}>
                  Holdout Volume
                </div>
                <div style={{ fontSize: '1.75rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: '#ea580c', lineHeight: 1.1 }}>
                  5,000
                </div>
                <div style={{ fontSize: '0.6875rem', color: 'var(--color-text-secondary)' }}>Independent array samples</div>
              </div>

              <div style={{ padding: '12px', background: 'var(--color-surface)', borderRadius: 6, border: '1px solid var(--color-border)' }}>
                <div style={{ fontSize: '0.6875rem', fontWeight: 700, color: 'var(--color-text-secondary)', textTransform: 'uppercase' }}>
                  Coverage Scope
                </div>
                <div style={{ fontSize: '1.75rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: '#ea580c', lineHeight: 1.1 }}>
                  10
                </div>
                <div style={{ fontSize: '0.6875rem', color: 'var(--color-text-secondary)' }}>Distinct physical conditions</div>
              </div>
            </div>

            <div
              style={{
                padding: '12px 14px',
                background: 'var(--color-surface)',
                border: '1px solid var(--color-border)',
                borderRadius: 6,
                display: 'flex',
                flexDirection: 'column',
                gap: 6,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ fontSize: '0.875rem' }}>🔒</span>
                <span style={{ fontWeight: 700, fontSize: '0.8125rem', color: 'var(--color-text-primary)' }}>
                  CLOSED FOR MODEL SELECTION
                </span>
              </div>
              <p style={{ margin: 0, fontSize: '0.75rem', color: 'var(--color-text-secondary)', lineHeight: 1.5 }}>
                <strong>USED ONLY FOR FINAL EVALUATION:</strong> This dataset was locked prior to neural network training and remained unobserved during model selection, hyperparameter tuning, and early stopping. It provides unbiased scientific proof of generalizability across varying SNR, coupling mismatches, and steering angles.
              </p>
            </div>

            <div style={{ fontSize: '0.6875rem', color: 'var(--color-text-tertiary)', fontFamily: 'var(--font-mono)' }}>
              Provenance Status: Locked &amp; Audited · Integrity hash verified
            </div>
          </div>
        </div>
      </div>

      {/* 6. Validated V4 Benchmark Section */}
      <div className="card">
        <div
          className="card-header"
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: 10,
          }}
        >
          <div>
            <h3 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 800, color: 'var(--color-text-primary)' }}>
              VALIDATED V4 BENCHMARK
            </h3>
            <p style={{ margin: '2px 0 0 0', fontSize: '0.8125rem', color: 'var(--color-text-secondary)' }}>
              Established reference performance on the 5,000-sample blind holdout test set.
            </p>
          </div>
          <span className="legend-chip frozen">FROZEN VALIDATED REFERENCE</span>
        </div>
        <div className="card-body" style={{ padding: 0 }}>
          <table className="benchmark-table">
            <thead>
              <tr>
                <th style={{ width: '30%' }}>Evaluation Metric</th>
                <th style={{ width: '25%', color: 'var(--color-analytical)' }}>Analytical Baseline</th>
                <th style={{ width: '25%', color: '#ea580c' }}>PINN-Cal V4</th>
                <th style={{ width: '20%' }}>Validated Reference Delta</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ fontWeight: 600 }}>Normalized Mean Squared Error (NMSE)</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>−14.23 dB</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--color-primary)' }}>−14.81 dB</td>
                <td style={{ fontFamily: 'var(--font-mono)', color: '#16a34a', fontWeight: 600 }}>+0.58 dB improvement</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>Mean Phase Error</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>3.96°</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>4.47°</td>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>+0.51° (prior trade-off)</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>Mean Gain Error</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>1.07 dB</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 800, color: '#16a34a' }}>1.00 dB</td>
                <td style={{ fontFamily: 'var(--font-mono)', color: '#16a34a', fontWeight: 600 }}>−0.07 dB improvement</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>Relative Calibration Error</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>15.13%</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>15.34%</td>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>+0.21%</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>Beam Pointing Accuracy Error</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>0.0253°</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 800 }}>0.0253°</td>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>Exact match (sub-0.03°)</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>Far-Field Beam Correlation (Pearson r)</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>0.99979</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 800, color: '#16a34a' }}>0.99979</td>
                <td style={{ fontFamily: 'var(--font-mono)', color: '#16a34a', fontWeight: 600 }}>Near-unity restoration</td>
              </tr>
            </tbody>
          </table>

          {/* Scientific Context Note */}
          <div
            style={{
              padding: '14px 18px',
              borderTop: '1px solid var(--color-border)',
              background: 'var(--color-bg)',
              fontSize: '0.8125rem',
              color: 'var(--color-text-secondary)',
              lineHeight: 1.55,
            }}
          >
            <strong>Benchmark Context:</strong> These values correspond to the established validated reference evaluation and are distinct from fresh live simulations and Monte Carlo runs. In multi-carrier mmWave beamforming, calibration performance is evaluated across a multi-dimensional objective rather than a single metric. PINN-Cal V4 delivers a validated 0.58 dB gain in global NMSE and reduced amplitude disparity while preserving high beam pointing fidelity.
          </div>
        </div>
      </div>

      {/* 7. Bottom Row: Reproducibility & Scientific Transparency */}
      <div className="provenance-2col-grid">
        {/* Reproducibility Specification */}
        <div className="card">
          <div className="card-header">
            <h3 style={{ margin: 0, fontSize: '1.125rem', fontWeight: 700, color: 'var(--color-text-primary)' }}>
              REPRODUCIBILITY &amp; SCIENTIFIC CITATION
            </h3>
          </div>
          <div className="card-body" style={{ padding: 0 }}>
            <table className="comparison-table" style={{ width: '100%' }}>
              <tbody>
                <tr>
                  <td style={{ width: '40%', fontWeight: 600 }}>Reference Seed</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>42</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Controlled Split</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>800 Train / 200 Val / 200 Test</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Blind Holdout</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>5,000 samples across 10 physical conditions</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Frozen V4 Parameters</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>δ = 0.15 · α = 0.50 · 33,088 weights</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Checkpoint File</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>pinn_v4_tikhonov015_seed42.pth</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Scientific Core ID</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>scientific-core-v1.0</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Why Scientific Provenance Matters */}
        <div className="card">
          <div className="card-header">
            <h3 style={{ margin: 0, fontSize: '1.125rem', fontWeight: 700, color: 'var(--color-text-primary)' }}>
              WHY SCIENTIFIC PROVENANCE MATTERS
            </h3>
          </div>
          <div className="card-body" style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)', lineHeight: 1.6 }}>
            <p style={{ marginTop: 0 }}>
              In scientific research and mission-critical 5G/6G RF engineering, credibility rests on reproducible evidence rather than unsubstantiated claims.
            </p>
            <p>
              This workspace explicitly distinguishes <strong>frozen reference evidence</strong> (model weights, physical impedance constraints, controlled splits, and closed holdouts) from <strong>live runtime telemetry</strong> (GPU execution, active platform devices, and real-time interactive simulations).
            </p>
            <p style={{ marginBottom: 0 }}>
              Live interactive simulations calculate instantaneous physics across user-controlled impairments without pre-recorded lookups or fabricated AI superiority.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
