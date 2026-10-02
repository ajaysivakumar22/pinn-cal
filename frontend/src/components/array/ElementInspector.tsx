import React from 'react';
import { useAppStore } from '../../store/appStore';
import { ConceptTrigger } from '../common/ConceptExplainer';

export const ElementInspector: React.FC = () => {
  const { simulationResult, selectedElement, setSelectedElement } = useAppStore();

  const elements = simulationResult?.elements;
  const numElements = elements?.length ?? 64;

  if (!elements || selectedElement === null || selectedElement >= elements.length) {
    return (
      <div className="card element-inspector-card empty-state" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 40 }}>
        <div style={{ textAlign: 'center', color: 'var(--color-text-secondary)' }}>
          <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>🔬</div>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 800, color: '#0f172a', marginBottom: 6 }}>
            Select an Antenna Element
          </h3>
          <p style={{ fontSize: '0.9375rem', color: 'var(--color-text-secondary)', maxWidth: 360, margin: '0 auto' }}>
            Click any patch in the live 64-element array above to inspect its transceiver RF metrics and neural diagnosis.
          </p>
        </div>
      </div>
    );
  }

  const elem = elements[selectedElement];
  const n = elem.element_index + 1;

  // Derived Values
  const truePhase = elem.true_phase_error_deg ?? elem.phase_error_deg;
  const trueGain = elem.true_gain_error_db ?? elem.gain_error_db;

  const pinnEstPhase =
    elem.pinn_estimated_phase_error_deg ??
    (elem.pinn_correction_phase_deg !== null ? -elem.pinn_correction_phase_deg : null);
  const pinnEstGain =
    elem.pinn_estimated_gain_error_db ??
    (elem.pinn_correction_magnitude !== null
      ? -20 * Math.log10(Math.max(elem.pinn_correction_magnitude, 1e-6))
      : null);

  const phaseRes =
    elem.phase_error_residual_deg ??
    (pinnEstPhase !== null
      ? ((((truePhase - pinnEstPhase + 180) % 360) + 360) % 360) - 180
      : null);
  const gainRes =
    elem.gain_error_residual_db ?? (pinnEstGain !== null ? trueGain - pinnEstGain : null);

  const compPhase =
    elem.compensation_phase_deg ??
    (pinnEstPhase !== null ? -pinnEstPhase : elem.analytical_correction_phase_deg);
  const compGain =
    elem.compensation_gain_db ??
    (pinnEstGain !== null
      ? -pinnEstGain
      : 20 * Math.log10(Math.max(elem.analytical_correction_magnitude, 1e-6)));

  const isPinnAvailable = simulationResult?.pinn_available ?? false;

  // Displayed estimate
  const displayEstPhase = pinnEstPhase ?? elem.analytical_estimated_phase_error_deg ?? 0;
  const displayEstGain = pinnEstGain ?? elem.analytical_estimated_gain_error_db ?? 0;
  const displayPhaseRes = phaseRes ?? elem.analytical_phase_residual_deg ?? 0;
  const displayGainRes = gainRes ?? elem.analytical_gain_residual_db ?? 0;

  return (
    <div className="card element-inspector-card" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Inspector Header with Stepper */}
      <div
        className="card-header"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%)',
          borderBottom: '1px solid #cbd5e1',
          padding: '14px 20px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span
            style={{
              padding: '6px 14px',
              borderRadius: 6,
              background: '#0f172a',
              color: '#ffffff',
              fontFamily: 'var(--font-mono)',
              fontWeight: 800,
              fontSize: '1rem',
              letterSpacing: '0.04em',
            }}
          >
            ELEMENT #{n}
          </span>
          <div>
            <h3 style={{ fontSize: '1.0625rem', color: '#0f172a', fontWeight: 800, margin: 0 }}>
              RF Transceiver Diagnostics
            </h3>
            <span style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)' }}>
              Port impedance: Z_in = 73.1 + 42.5j Ω
            </span>
          </div>
        </div>

        {/* Stepper Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <button
            className="btn btn-secondary"
            onClick={() => setSelectedElement(Math.max(0, selectedElement - 1))}
            disabled={selectedElement === 0}
            style={{ padding: '6px 12px', fontSize: '0.875rem', fontWeight: 700 }}
            title="Inspect previous element"
          >
            ◀ Prev
          </button>
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.875rem',
              color: 'var(--color-text-primary)',
              fontWeight: 700,
              minWidth: 60,
              textAlign: 'center',
            }}
          >
            {n} / {numElements}
          </span>
          <button
            className="btn btn-secondary"
            onClick={() => setSelectedElement(Math.min(numElements - 1, selectedElement + 1))}
            disabled={selectedElement === numElements - 1}
            style={{ padding: '6px 12px', fontSize: '0.875rem', fontWeight: 700 }}
            title="Inspect next element"
          >
            Next ▶
          </button>
        </div>
      </div>

      <div className="card-body" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: 18 }}>
        {/* SECTION 1: PINN DETECTOR HERO (Primary Diagnostic) */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: '1.125rem' }}>🎯</span>
              <h4 style={{ fontSize: '0.9375rem', fontWeight: 800, color: '#0f172a', letterSpacing: '0.04em' }}>
                NEURAL DIAGNOSIS (PINN DETECTOR)
              </h4>
            </div>
            <span
              style={{
                fontSize: '0.75rem',
                fontWeight: 700,
                color: isPinnAvailable ? '#6d28d9' : '#0e7490',
                background: isPinnAvailable ? '#f3e8ff' : '#cffafe',
                padding: '2px 8px',
                borderRadius: 4,
              }}
            >
              {isPinnAvailable ? 'PINN INFERENCE ACTIVE' : 'ANALYTICAL BACKUP'}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {/* Phase Impairment Detector Card */}
            <div
              style={{
                padding: '16px',
                borderRadius: 8,
                border: '2px solid #e9d5ff',
                background: 'linear-gradient(180deg, #faf5ff 0%, #ffffff 100%)',
                boxShadow: '0 2px 4px rgba(124, 58, 237, 0.05)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <span style={{ fontSize: '0.8125rem', fontWeight: 700, color: '#6b21a8' }}>
                  PHASE IMPAIRMENT
                </span>
                <ConceptTrigger conceptKey="phase_error" label="Theory" />
              </div>

              {/* Large Estimated Value */}
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, margin: '6px 0' }}>
                <span
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '2rem',
                    fontWeight: 900,
                    color: '#581c87',
                    letterSpacing: '-0.02em',
                  }}
                >
                  {displayEstPhase > 0 ? `+${displayEstPhase.toFixed(2)}°` : `${displayEstPhase.toFixed(2)}°`}
                </span>
                <span
                  style={{
                    fontSize: '0.6875rem',
                    fontWeight: 800,
                    background: '#8b5cf6',
                    color: '#ffffff',
                    padding: '2px 6px',
                    borderRadius: 3,
                  }}
                >
                  PINN ESTIMATE
                </span>
              </div>

              {/* Circular Polar Gauge Needle */}
              <div style={{ display: 'flex', justifyContent: 'center', margin: '8px 0' }}>
                <PhaseGaugeSVG truePhase={truePhase} estPhase={displayEstPhase} />
              </div>

              <div style={{ fontSize: '0.8125rem', color: '#6b21a8', lineHeight: 1.4 }}>
                Aperture phase deviation detected from coupled currents.
              </div>
            </div>

            {/* Gain Impairment Detector Card */}
            <div
              style={{
                padding: '16px',
                borderRadius: 8,
                border: '2px solid #e9d5ff',
                background: 'linear-gradient(180deg, #faf5ff 0%, #ffffff 100%)',
                boxShadow: '0 2px 4px rgba(124, 58, 237, 0.05)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <span style={{ fontSize: '0.8125rem', fontWeight: 700, color: '#6b21a8' }}>
                  GAIN IMPAIRMENT
                </span>
                <ConceptTrigger conceptKey="gain_error" label="Theory" />
              </div>

              {/* Large Estimated Value */}
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, margin: '6px 0' }}>
                <span
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '2rem',
                    fontWeight: 900,
                    color: '#581c87',
                    letterSpacing: '-0.02em',
                  }}
                >
                  {displayEstGain > 0 ? `+${displayEstGain.toFixed(2)} dB` : `${displayEstGain.toFixed(2)} dB`}
                </span>
                <span
                  style={{
                    fontSize: '0.6875rem',
                    fontWeight: 800,
                    background: '#8b5cf6',
                    color: '#ffffff',
                    padding: '2px 6px',
                    borderRadius: 3,
                  }}
                >
                  PINN ESTIMATE
                </span>
              </div>

              {/* Horizontal Gain Bar Gauge */}
              <div style={{ margin: '8px 0' }}>
                <GainGaugeSVG trueGain={trueGain} estGain={displayEstGain} />
              </div>

              <div style={{ fontSize: '0.8125rem', color: '#6b21a8', lineHeight: 1.4 }}>
                Amplifier amplitude gain mismatch relative to nominal 0 dB.
              </div>
            </div>
          </div>
        </div>

        {/* SECTION 2: COMPENSATION ACTION FLOW */}
        <div
          style={{
            padding: '14px 18px',
            background: '#f0fdf4',
            border: '1.5px solid #86efac',
            borderRadius: 8,
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span style={{ fontSize: '0.875rem', fontWeight: 800, color: '#166534' }}>
              ⚡ INVERSE COMPENSATION WEIGHTS APPLIED (C = H⁻¹)
            </span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8125rem', color: '#15803d', fontWeight: 700 }}>
              Corrected State: V_corr = V_hw · C
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
            <div style={{ background: '#ffffff', padding: '10px 14px', borderRadius: 6, border: '1px solid #bbf7d0' }}>
              <span style={{ fontSize: '0.75rem', color: '#166534', fontWeight: 600 }}>PHASE COMPENSATION (Δϕ):</span>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.25rem', fontWeight: 800, color: '#15803d' }}>
                {compPhase !== null ? (compPhase > 0 ? `+${compPhase.toFixed(2)}°` : `${compPhase.toFixed(2)}°`) : '0.00°'}
              </div>
            </div>

            <div style={{ background: '#ffffff', padding: '10px 14px', borderRadius: 6, border: '1px solid #bbf7d0' }}>
              <span style={{ fontSize: '0.75rem', color: '#166534', fontWeight: 600 }}>GAIN COMPENSATION (ΔG):</span>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.25rem', fontWeight: 800, color: '#15803d' }}>
                {compGain !== null ? (compGain > 0 ? `+${compGain.toFixed(2)} dB` : `${compGain.toFixed(2)} dB`) : '0.00 dB'}
              </div>
            </div>
          </div>
        </div>

        {/* SECTION 3: OBSERVED PORT SIGNAL */}
        <div
          style={{
            padding: '12px 16px',
            background: '#f8fafc',
            border: '1px solid #e2e8f0',
            borderRadius: 6,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <div>
            <span style={{ fontSize: '0.8125rem', fontWeight: 700, color: '#334155' }}>
              OBSERVED CURRENT AT TRANSCEIVER PORT (I_obs)
            </span>
            <div style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)', marginTop: 2 }}>
              Result of excitation coupled through mutual impedance matrix Z_coupled
            </div>
          </div>
          <div style={{ display: 'flex', gap: 14, fontFamily: 'var(--font-mono)', fontSize: '0.9375rem' }}>
            <span>Re: <strong>{elem.observed_real.toFixed(4)}</strong></span>
            <span>Im: <strong>{elem.observed_imag.toFixed(4)}</strong></span>
          </div>
        </div>

        {/* SECTION 4: SIMULATION GROUND TRUTH (VALIDATION REFERENCE) */}
        <div
          style={{
            padding: '14px 16px',
            background: '#fef2f2',
            border: '1px dashed #fca5a5',
            borderRadius: 8,
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: '0.875rem' }}>🔍</span>
              <span style={{ fontSize: '0.8125rem', fontWeight: 800, color: '#991b1b' }}>
                SIMULATION GROUND TRUTH & RESIDUAL
              </span>
            </div>
            <span
              style={{
                fontSize: '0.6875rem',
                fontWeight: 800,
                background: '#fee2e2',
                color: '#991b1b',
                padding: '2px 8px',
                borderRadius: 4,
                border: '1px solid #fecaca',
              }}
            >
              SIMULATION ONLY · NOT ACCESSIBLE TO PHYSICAL PINN
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, fontSize: '0.875rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#7f1d1d' }}>True Phase Error:</span>
              <strong style={{ fontFamily: 'var(--font-mono)', color: '#991b1b' }}>
                {truePhase > 0 ? `+${truePhase.toFixed(2)}°` : `${truePhase.toFixed(2)}°`}
              </strong>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#7f1d1d' }}>Phase Residual:</span>
              <strong style={{ fontFamily: 'var(--font-mono)', color: '#16a34a' }}>
                {displayPhaseRes > 0 ? `+${displayPhaseRes.toFixed(2)}°` : `${displayPhaseRes.toFixed(2)}°`}
              </strong>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#7f1d1d' }}>True Gain Error:</span>
              <strong style={{ fontFamily: 'var(--font-mono)', color: '#991b1b' }}>
                {trueGain > 0 ? `+${trueGain.toFixed(2)} dB` : `${trueGain.toFixed(2)} dB`}
              </strong>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#7f1d1d' }}>Gain Residual:</span>
              <strong style={{ fontFamily: 'var(--font-mono)', color: '#16a34a' }}>
                {displayGainRes > 0 ? `+${displayGainRes.toFixed(2)} dB` : `${displayGainRes.toFixed(2)} dB`}
              </strong>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Subcomponent: Circular Phase Gauge SVG (Polar Dial)
const PhaseGaugeSVG: React.FC<{ truePhase: number; estPhase: number }> = ({ truePhase, estPhase }) => {
  const clamp = (v: number) => Math.max(-45, Math.min(45, v));
  const trueAngle = (clamp(truePhase) / 45) * 60;
  const estAngle = (clamp(estPhase) / 45) * 60;

  return (
    <svg width="150" height="80" viewBox="0 0 150 80">
      {/* Gauge Arc Track */}
      <path d="M 22 70 A 55 55 0 0 1 128 70" fill="none" stroke="#e2e8f0" strokeWidth="10" strokeLinecap="round" />
      <path d="M 50 70 A 55 55 0 0 1 100 70" fill="none" stroke="#bbf7d0" strokeWidth="10" />

      {/* Center Pivot */}
      <circle cx="75" cy="70" r="5" fill="#0f172a" />

      {/* True Phase Indicator (Red Tick, Ground Truth) */}
      <g transform={`rotate(${trueAngle} 75 70)`}>
        <line x1="75" y1="70" x2="75" y2="18" stroke="#dc2626" strokeWidth="2.5" strokeDasharray="3,2" />
        <circle cx="75" cy="18" r="3.5" fill="#dc2626" />
      </g>

      {/* PINN Estimated Needle (Purple) */}
      <g transform={`rotate(${estAngle} 75 70)`}>
        <line x1="75" y1="70" x2="75" y2="20" stroke="#7c3aed" strokeWidth="3.5" strokeLinecap="round" />
      </g>

      {/* Scale Labels */}
      <text x="18" y="78" fontSize="9" fill="#94a3b8" fontFamily="var(--font-mono)">-45°</text>
      <text x="75" y="60" fontSize="11" fill="#581c87" fontWeight="800" textAnchor="middle" fontFamily="var(--font-mono)">
        {estPhase > 0 ? `+${estPhase.toFixed(1)}°` : `${estPhase.toFixed(1)}°`}
      </text>
      <text x="132" y="78" fontSize="9" fill="#94a3b8" fontFamily="var(--font-mono)">+45°</text>
    </svg>
  );
};

// Subcomponent: Horizontal Gain Gauge Bar SVG
const GainGaugeSVG: React.FC<{ trueGain: number; estGain: number }> = ({ trueGain, estGain }) => {
  const toX = (db: number) => {
    const clamped = Math.max(-4, Math.min(4, db));
    return ((clamped + 4) / 8) * 190;
  };

  const trueX = toX(trueGain);
  const estX = toX(estGain);

  return (
    <svg width="100%" height="48" viewBox="0 0 190 48">
      {/* Background Track */}
      <rect x="0" y="16" width="190" height="12" rx="6" fill="#f1f5f9" stroke="#cbd5e1" strokeWidth="1" />
      {/* Center 0 dB Reference */}
      <line x1="95" y1="12" x2="95" y2="32" stroke="#64748b" strokeWidth="1.5" strokeDasharray="2,2" />

      {/* Ground Truth Marker (Red Triangle) */}
      <polygon points={`${trueX-4},8 ${trueX+4},8 ${trueX},15`} fill="#dc2626" />

      {/* PINN Estimate Pill (Purple) */}
      <circle cx={estX} cy="22" r="7" fill="#7c3aed" stroke="#ffffff" strokeWidth="2" />

      {/* Scale Labels */}
      <text x="2" y="42" fontSize="9" fill="#94a3b8" fontFamily="var(--font-mono)">-4 dB</text>
      <text x="95" y="42" fontSize="9" fill="#64748b" textAnchor="middle" fontFamily="var(--font-mono)">0 dB</text>
      <text x="188" y="42" fontSize="9" fill="#94a3b8" textAnchor="end" fontFamily="var(--font-mono)">+4 dB</text>
    </svg>
  );
};
