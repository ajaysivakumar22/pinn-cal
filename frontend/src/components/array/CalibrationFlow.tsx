import React from 'react';
import { useAppStore } from '../../store/appStore';
import { ConceptTrigger } from '../common/ConceptExplainer';

export const CalibrationFlow: React.FC = () => {
  const { simulationResult, selectedElement } = useAppStore();

  const elements = simulationResult?.elements;
  const numElements = elements?.length ?? 64;
  const currentIdx =
    selectedElement !== null && selectedElement < numElements
      ? selectedElement
      : 0;

  const elem = elements ? elements[currentIdx] : null;

  // Actual values for selected element or representative element
  const truePhase = elem ? (elem.true_phase_error_deg ?? elem.phase_error_deg) : 20.0;
  const trueGain = elem ? (elem.true_gain_error_db ?? elem.gain_error_db) : 2.0;

  const pinnEstPhase = elem
    ? (elem.pinn_estimated_phase_error_deg ??
      (elem.pinn_correction_phase_deg !== null ? -elem.pinn_correction_phase_deg : 18.5))
    : 18.5;
  const pinnEstGain = elem
    ? (elem.pinn_estimated_gain_error_db ??
      (elem.pinn_correction_magnitude !== null
        ? -20 * Math.log10(Math.max(elem.pinn_correction_magnitude, 1e-6))
        : 1.8))
    : 1.8;

  const compPhase = -pinnEstPhase;
  const compGain = -pinnEstGain;

  // Corrected residual state
  const residualPhase = elem ? (elem.phase_error_residual_deg ?? truePhase - pinnEstPhase) : truePhase - pinnEstPhase;
  const residualGain = elem ? (elem.gain_error_residual_db ?? trueGain - pinnEstGain) : trueGain - pinnEstGain;

  return (
    <div
      className="card calibration-flow-card"
      style={{
        background: '#ffffff',
        border: '1px solid #cbd5e1',
        borderRadius: 8,
        padding: '14px 18px',
        marginBottom: 20,
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 12,
          flexWrap: 'wrap',
          gap: 10,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: '1rem', fontWeight: 800, color: '#0f172a' }}>
            RF IMPAIRMENT & COMPENSATION LIFECYCLE
          </span>
          <span
            style={{
              fontSize: '0.75rem',
              fontWeight: 700,
              padding: '2px 8px',
              borderRadius: 4,
              background: '#0f172a',
              color: '#ffffff',
              fontFamily: 'var(--font-mono)',
            }}
          >
            ELEMENT #{currentIdx + 1}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)', fontWeight: 500 }}>
            Mathematical Transformation:
          </span>
          <ConceptTrigger conceptKey="compensation_weights" label="C = H⁻¹" />
        </div>
      </div>

      {/* 4-Stage Horizontal Transformation Diagram */}
      <div className="calibration-flow-grid">
        {/* 1. IMPAIRED STATE */}
        <div className="flow-step-box impaired">
          <div className="flow-step-label">1. IMPAIRED STATE (HW)</div>
          <div className="flow-step-desc">Hardware channel error</div>
          <div className="flow-step-values">
            <div className="val-row">
              <span className="name">Phase Error:</span>
              <span className="val red">{truePhase > 0 ? `+${truePhase.toFixed(2)}°` : `${truePhase.toFixed(2)}°`}</span>
            </div>
            <div className="val-row">
              <span className="name">Gain Error:</span>
              <span className="val red">{trueGain > 0 ? `+${trueGain.toFixed(2)} dB` : `${trueGain.toFixed(2)} dB`}</span>
            </div>
          </div>
          <div className="flow-subtext">Ground truth in simulation</div>
        </div>

        <div className="flow-arrow" aria-hidden="true">→</div>

        {/* 2. PINN ESTIMATE */}
        <div className="flow-step-box estimate">
          <div className="flow-step-label">2. PINN ESTIMATE</div>
          <div className="flow-step-desc">Derived from predicted C</div>
          <div className="flow-step-values">
            <div className="val-row">
              <span className="name">Est Phase:</span>
              <span className="val purple">{pinnEstPhase > 0 ? `+${pinnEstPhase.toFixed(2)}°` : `${pinnEstPhase.toFixed(2)}°`}</span>
            </div>
            <div className="val-row">
              <span className="name">Est Gain:</span>
              <span className="val purple">{pinnEstGain > 0 ? `+${pinnEstGain.toFixed(2)} dB` : `${pinnEstGain.toFixed(2)} dB`}</span>
            </div>
          </div>
          <div className="flow-subtext">θ̂ = −arg(C), Ĝ = −20log|C|</div>
        </div>

        <div className="flow-arrow" aria-hidden="true">→</div>

        {/* 3. INVERSE COMPENSATION */}
        <div className="flow-step-box compensation">
          <div className="flow-step-label">3. COMPENSATION</div>
          <div className="flow-step-desc">Opposite direction weight</div>
          <div className="flow-step-values">
            <div className="val-row">
              <span className="name">Phase Weight:</span>
              <span className="val blue">{compPhase > 0 ? `+${compPhase.toFixed(2)}°` : `${compPhase.toFixed(2)}°`}</span>
            </div>
            <div className="val-row">
              <span className="name">Gain Weight:</span>
              <span className="val blue">{compGain > 0 ? `+${compGain.toFixed(2)} dB` : `${compGain.toFixed(2)} dB`}</span>
            </div>
          </div>
          <div className="flow-subtext">w_comp = 1 / H_est</div>
        </div>

        <div className="flow-arrow" aria-hidden="true">→</div>

        {/* 4. POST-COMPENSATION STATE */}
        <div className="flow-step-box corrected">
          <div className="flow-step-label">4. POST-COMPENSATION STATE</div>
          <div className="flow-step-desc">Restored excitation residual</div>
          <div className="flow-step-values">
            <div className="val-row">
              <span className="name">Phase Residual:</span>
              <span className="val green">{residualPhase > 0 ? `+${residualPhase.toFixed(2)}°` : `${residualPhase.toFixed(2)}°`}</span>
            </div>
            <div className="val-row">
              <span className="name">Gain Residual:</span>
              <span className="val green">{residualGain > 0 ? `+${residualGain.toFixed(2)} dB` : `${residualGain.toFixed(2)} dB`}</span>
            </div>
          </div>
          <div className="flow-subtext">Post-compensation residual</div>
        </div>
      </div>
    </div>
  );
};
