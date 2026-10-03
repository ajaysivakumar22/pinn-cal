import React, { useState } from 'react';

interface StepItem {
  num: string;
  title: string;
  icon: string;
  simpleDesc: string;
  techDesc: string;
  formula?: string;
}

const STEPS: StepItem[] = [
  {
    num: '1',
    title: 'Ideal Excitation',
    icon: '🎯',
    simpleDesc: 'The 5G transmitter commands each antenna to emit waves with precise phase steps so they align toward the mobile user.',
    techDesc: 'Array steering vector commanded as a_n(θ) = exp(j k x_n sin(θ)) across all N = 64 elements.',
    formula: 'V_ideal = [exp(j k x_0 sin θ), ..., exp(j k x_{N-1} sin θ)]^T',
  },
  {
    num: '2',
    title: 'Hardware Impairments',
    icon: '⚡',
    simpleDesc: 'Real RF chips have small manufacturing flaws: power amplifiers have gain ripples, and phase shifters have phase skews.',
    techDesc: 'Element-wise multiplicative complex hardware channel distortion H = diag(g_n exp(j ϕ_n)).',
    formula: 'V_hw = H · V_ideal, where ϕ_n ~ N(0, σ_ϕ²), g_n ~ LN(0, σ_g²)',
  },
  {
    num: '3',
    title: 'Mutual Coupling',
    icon: '🔄',
    simpleDesc: 'Because 64 antennas sit packed closely together (5.35 mm apart), each element sprays a little energy into its neighbors.',
    techDesc: 'Electromagnetic cross-talk modeled via reciprocal complex mutual impedance matrix Z_coupled.',
    formula: 'V_coupled = Z_coupled · I_coupled',
  },
  {
    num: '4',
    title: 'Noisy Observation',
    icon: '📡',
    simpleDesc: 'Calibration receivers sample the resulting coupled currents, with inevitable background thermal noise.',
    techDesc: 'Observed RF current vector contaminated with additive complex Gaussian receiver noise.',
    formula: 'I_obs = (Z_coupled + Z_0 I)^{-1} V_hw + n_noise, SNR = 20 dB',
  },
  {
    num: '5',
    title: 'PINN-Cal V4 Compensation',
    icon: '🧠',
    simpleDesc: 'Our physics-informed neural network inspects observed currents and computes compensation weights across all 64 channels.',
    techDesc: 'Two-stage architecture: Tikhonov physics preconditioner (δ=0.15) combined with bounded neural residual denoiser (128 → 128 → 64 → 128, α=0.50, 33,088 parameters). Input: 128 real features [Re/Im(c_prior)], Output: 128 real features [Δc].',
    formula: 'ĉ = c_prior + Δc,  where  Δc = α · tanh(r_θ(Re/Im(c_prior)))',
  },
  {
    num: '6',
    title: 'Correction Estimation',
    icon: '⚖️',
    simpleDesc: 'For each antenna, we convert the prediction into phase compensation and gain compensation to undo the errors.',
    techDesc: 'Per-element impairment estimation and inverse compensation weight generation.',
    formula: 'θ̂_n = -arg(C_n),  Ĝ_n = -20 log10 |C_n|,  w_comp = C_n',
  },
  {
    num: '7',
    title: 'Corrected Excitation',
    icon: '🔧',
    simpleDesc: 'We apply the compensation weights before transmission, balancing out the hardware distortion.',
    techDesc: 'Aperture excitation synthesized via element-wise Hadamard product.',
    formula: 'V_corr = C ⊙ V_hw ≈ V_ideal',
  },
  {
    num: '8',
    title: 'Restored Beam',
    icon: '📶',
    simpleDesc: 'Waves once again add up in phase, restoring a tight, focused, high-speed 5G beam toward the target angle!',
    techDesc: 'Far-field spatial radiation array factor synthesized with main-lobe alignment and low sidelobes.',
    formula: 'AF(θ) = ∑_{n=0}^{N-1} V_{corr, n} · exp(j k x_n sin θ)',
  },
];

export const HowItWorksModal: React.FC<{ isOpen: boolean; onClose: () => void }> = ({
  isOpen,
  onClose,
}) => {
  const [viewMode, setViewMode] = useState<'simple' | 'technical'>('simple');

  if (!isOpen) return null;

  return (
    <div
      className="modal-backdrop"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="how-it-works-title"
    >
      <div
        className="modal-content card"
        onClick={(e) => e.stopPropagation()}
        style={{
          maxWidth: 780,
          width: '94%',
          maxHeight: '90vh',
          overflowY: 'auto',
          boxShadow: '0 25px 35px -10px rgba(0, 0, 0, 0.2)',
          borderRadius: 12,
        }}
      >
        <div
          className="card-header"
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '16px 22px',
            background: 'linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%)',
            borderBottom: '1px solid #e2e8f0',
          }}
        >
          <div>
            <h2 id="how-it-works-title" style={{ fontSize: '1.25rem', fontWeight: 800, margin: 0, color: '#0f172a' }}>
              How PINN-Cal Works
            </h2>
            <span style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)', fontWeight: 500 }}>
              End-to-End 8-Stage Lifecycle: From RF Impairment to Beam Restoration
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {/* View Mode Switcher */}
            <div className="btn-group" style={{ display: 'flex', background: '#e2e8f0', borderRadius: 6, padding: 2 }}>
              <button
                className={`btn ${viewMode === 'simple' ? 'btn-primary' : 'btn-ghost'}`}
                onClick={() => setViewMode('simple')}
                style={{ fontSize: '0.75rem', padding: '4px 10px', borderRadius: 4, fontWeight: 700 }}
              >
                ★ Franky's Simple View
              </button>
              <button
                className={`btn ${viewMode === 'technical' ? 'btn-primary' : 'btn-ghost'}`}
                onClick={() => setViewMode('technical')}
                style={{ fontSize: '0.75rem', padding: '4px 10px', borderRadius: 4, fontWeight: 700 }}
              >
                📐 Technical Physics View
              </button>
            </div>

            <button
              className="btn btn-secondary"
              onClick={onClose}
              style={{ padding: '4px 10px', fontSize: '0.875rem', fontWeight: 700 }}
              aria-label="Close modal"
            >
              ✕
            </button>
          </div>
        </div>

        <div className="card-body" style={{ padding: '20px 22px' }}>
          <div className="how-it-works-pipeline">
            {STEPS.map((s, idx) => (
              <div key={s.num} className="how-it-works-step">
                <div className="step-badge-col">
                  <div className="step-num-badge">{s.num}</div>
                  {idx < STEPS.length - 1 && <div className="step-connector-line" />}
                </div>

                <div className="step-body-col">
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <span style={{ fontSize: '1.25rem' }}>{s.icon}</span>
                    <h3 style={{ fontSize: '1.0625rem', fontWeight: 700, margin: 0, color: '#0f172a' }}>
                      {s.title}
                    </h3>
                  </div>

                  <p style={{ fontSize: '0.9375rem', lineHeight: 1.6, color: '#334155', margin: '4px 0 8px 0' }}>
                    {viewMode === 'simple' ? s.simpleDesc : s.techDesc}
                  </p>

                  {viewMode === 'technical' && s.formula && (
                    <div
                      style={{
                        padding: '6px 10px',
                        background: '#f8fafc',
                        border: '1px solid #cbd5e1',
                        borderRadius: 6,
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.8125rem',
                        color: '#0f172a',
                      }}
                    >
                      {s.formula}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          <div
            style={{
              marginTop: 20,
              padding: '12px 16px',
              background: '#f0fdf4',
              border: '1px solid #bbf7d0',
              borderRadius: 8,
              fontSize: '0.875rem',
              color: '#166534',
              lineHeight: 1.5,
            }}
          >
            <strong>Engineering Verification:</strong> The PINN-Cal application uses real PyTorch neural computation and
            validated electromagnetic physics. Ground truth is provided in simulation mode for evaluation purposes, while the
            PINN infers compensation solely from observed port currents.
          </div>
        </div>
      </div>
    </div>
  );
};
