import React from 'react';
import { useAppStore } from '../../store/appStore';
import { ConceptTrigger } from '../common/ConceptExplainer';

export const ElementInspector: React.FC = () => {
  const { simulationResult, selectedElement, setSelectedElement } = useAppStore();

  const elements = simulationResult?.elements;
  const numElements = elements?.length ?? 64;

  // Active element index fallback
  const activeIdx = selectedElement !== null && selectedElement < numElements ? selectedElement : 16;
  const elem = elements?.[activeIdx];
  const n = activeIdx + 1;

  if (!elements || !elem) {
    return (
      <div className="card element-inspector-card empty-state" style={{ padding: 40, textAlign: 'center' }}>
        <div style={{ color: 'var(--color-text-secondary)' }}>
          <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>🔬</div>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--color-text-primary)', marginBottom: 6 }}>
            Select an Antenna Element
          </h3>
          <p style={{ fontSize: '1rem', color: 'var(--color-text-secondary)', maxWidth: 420, margin: '0 auto' }}>
            Click any element in the live 64-element ULA above to inspect its transceiver RF metrics and neural compensation state.
          </p>
        </div>
      </div>
    );
  }

  // 1. Observed / Impaired Values
  const obsPhaseError = elem.phase_error_deg;
  const obsGainError = elem.gain_error_db;
  const hwPhase = elem.hardware_phase_deg;
  const hwMag = elem.hardware_magnitude;
  const isSevereImpaired = Math.abs(obsGainError) > 1.2 || Math.abs(obsPhaseError) > 12;

  // 2. V4 Estimated Compensation Values: c_hat = c_prior + Δc
  const cpRe = elem.pinn_correction_real ?? elem.analytical_correction_real ?? 1.0;
  const cpIm = elem.pinn_correction_imag ?? elem.analytical_correction_imag ?? 0.0;
  let cMag = elem.pinn_correction_magnitude;
  let cPhase = elem.pinn_correction_phase_deg;
  if (cMag == null) {
    cMag = Math.sqrt(cpRe * cpRe + cpIm * cpIm);
  }
  if (cPhase == null) {
    cPhase = (Math.atan2(cpIm, cpRe) * 180) / Math.PI;
  }
  const cGainDb = 20 * Math.log10(Math.max(cMag, 1e-12));
  const isPinnAvailable = simulationResult?.pinn_available ?? false;

  // 3. Post-Compensation Values: V_post,n = V_hw,n · c_hat,n
  const vCorrRe = elem.hardware_real * cpRe - elem.hardware_imag * cpIm;
  const vCorrIm = elem.hardware_real * cpIm + elem.hardware_imag * cpRe;
  const corrPhase = (Math.atan2(vCorrIm, vCorrRe) * 180) / Math.PI;
  const corrMag = Math.sqrt(vCorrRe * vCorrRe + vCorrIm * vCorrIm);
  const corrGainDb = 20 * Math.log10(Math.max(corrMag, 1e-12));

  // Residual relative to ideal reference
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

  const displayPhaseRes =
    elem.phase_error_residual_deg ??
    (pinnEstPhase !== null
      ? ((((truePhase - pinnEstPhase + 180) % 360) + 360) % 360) - 180
      : elem.analytical_phase_residual_deg ?? (corrPhase - elem.ideal_phase_deg));

  const displayGainRes =
    elem.gain_error_residual_db ??
    (pinnEstGain !== null
      ? trueGain - pinnEstGain
      : elem.analytical_gain_residual_db ?? corrGainDb);

  return (
    <div className="card element-inspector-card" role="region" aria-label="Selected Element Detail Inspector">
      {/* 1. INSPECTOR HEADER WITH ELEMENT NAVIGATION (Requirements 10 & 11) */}
      <div className="inspector-header-bar">
        {/* Element Indicator with Large Typography (24–32px) */}
        <div className="inspector-badge-group">
          <div className="inspector-element-badge">
            ELEMENT #{n}
          </div>
          <div className="inspector-title-sub">
            <h3 className="inspector-title">
              Transceiver RF Microstrip Port #{n}
            </h3>
            <span className="inspector-port-impedance">
              Port Input Impedance: Z_in = 73.1 + 42.5j Ω · Reciprocal S-Parameters
            </span>
          </div>
        </div>

        {/* Stepper & Numeric Selector Controls */}
        <div className="inspector-nav-controls">
          <button
            type="button"
            className="btn btn-secondary inspector-nav-btn"
            onClick={() => setSelectedElement(Math.max(0, activeIdx - 1))}
            disabled={activeIdx === 0}
            title="Inspect previous element"
          >
            ◀ Prev
          </button>

          <div className="inspector-element-counter" title="Current element out of 64">
            <span className="counter-curr">{n}</span>
            <span className="counter-sep">/</span>
            <span className="counter-total">{numElements}</span>
          </div>

          <button
            type="button"
            className="btn btn-secondary inspector-nav-btn"
            onClick={() => setSelectedElement(Math.min(numElements - 1, activeIdx + 1))}
            disabled={activeIdx === numElements - 1}
            title="Inspect next element"
          >
            Next ▶
          </button>

          {/* Quick Element Selector Dropdown */}
          <div className="inspector-dropdown-wrapper">
            <label htmlFor="element-quick-select" className="sr-only">Jump to element</label>
            <select
              id="element-quick-select"
              className="form-select inspector-select"
              value={activeIdx}
              onChange={(e) => setSelectedElement(Number(e.target.value))}
              aria-label="Directly jump to element"
            >
              {Array.from({ length: numElements }, (_, i) => (
                <option key={i} value={i}>
                  Element #{i + 1}
                </option>
              ))}
            </select>
          </div>

          {/* Quick Jump Pills */}
          <div className="inspector-quick-pills">
            <button
              type="button"
              className={`pill-btn ${activeIdx === 0 ? 'active' : ''}`}
              onClick={() => setSelectedElement(0)}
              title="Jump to element 1 (aperture edge)"
            >
              #1
            </button>
            <button
              type="button"
              className={`pill-btn ${activeIdx === 31 ? 'active' : ''}`}
              onClick={() => setSelectedElement(31)}
              title="Jump to element 32 (aperture center)"
            >
              #32
            </button>
            <button
              type="button"
              className={`pill-btn ${activeIdx === 63 ? 'active' : ''}`}
              onClick={() => setSelectedElement(63)}
              title="Jump to element 64 (aperture edge)"
            >
              #64
            </button>
          </div>
        </div>
      </div>

      {/* 2. FOUR COMPARATIVE STATE CARDS (Requirements 5, 6, 7, 8, 10, 22) */}
      <div className="inspector-states-grid">
        {/* CARD 01 — IDEAL EXCITATION (REFERENCE) */}
        <div className="state-detail-card card-ideal">
          <div className="state-card-header">
            <div className="state-card-num-title">
              <span className="state-mini-badge state-badge-ideal">01</span>
              <h4 className="state-card-title">IDEAL EXCITATION</h4>
            </div>
            <span className="state-role-tag role-reference">REFERENCE</span>
          </div>

          <p className="state-card-desc">
            Theoretical unit amplitude excitation with progressive steering delay.
          </p>

          <div className="state-metric-rows">
            <div className="metric-row">
              <span className="metric-label">Phase (ϕ_ideal):</span>
              <span className="inspector-metric-val metric-blue">
                {elem.ideal_phase_deg > 0 ? `+${elem.ideal_phase_deg.toFixed(2)}°` : `${elem.ideal_phase_deg.toFixed(2)}°`}
              </span>
            </div>
            <div className="metric-row">
              <span className="metric-label">Gain / Magnitude:</span>
              <span className="inspector-metric-val metric-blue">
                0.00 dB <span className="val-sub">(1.000)</span>
              </span>
            </div>
            <div className="metric-row sub-row">
              <span className="metric-label">Complex Excitation:</span>
              <span className="metric-val-mono">
                {elem.ideal_real.toFixed(3)} + {elem.ideal_imag >= 0 ? '+' : ''}{elem.ideal_imag.toFixed(3)}j
              </span>
            </div>
          </div>
        </div>

        {/* CARD 02 — OBSERVED / IMPAIRED */}
        <div className={`state-detail-card card-hardware ${isSevereImpaired ? 'border-alert' : ''}`}>
          <div className="state-card-header">
            <div className="state-card-num-title">
              <span className="state-mini-badge state-badge-hardware">02</span>
              <h4 className="state-card-title">OBSERVED / IMPAIRED</h4>
            </div>
            <span className={`state-role-tag ${isSevereImpaired ? 'role-alert' : 'role-impaired'}`}>
              {isSevereImpaired ? 'SEVERE IMPAIRMENT' : 'HARDWARE STATE'}
            </span>
          </div>

          <p className="state-card-desc">
            Transceiver state disturbed by reciprocal mutual coupling and RF error.
          </p>

          <div className="state-metric-rows">
            <div className="metric-row">
              <span className="metric-label">Phase Impairment:</span>
              <span className={`inspector-metric-val ${Math.abs(obsPhaseError) > 10 ? 'metric-red' : 'metric-amber'}`}>
                {obsPhaseError > 0 ? `+${obsPhaseError.toFixed(2)}°` : `${obsPhaseError.toFixed(2)}°`}
              </span>
            </div>
            <div className="metric-row">
              <span className="metric-label">Gain Impairment:</span>
              <span className={`inspector-metric-val ${Math.abs(obsGainError) > 1.0 ? 'metric-red' : 'metric-amber'}`}>
                {obsGainError > 0 ? `+${obsGainError.toFixed(2)} dB` : `${obsGainError.toFixed(2)} dB`}
              </span>
            </div>
            <div className="metric-row sub-row">
              <span className="metric-label">Observed Current (I_obs):</span>
              <span className="metric-val-mono">
                {elem.observed_real.toFixed(4)} + {elem.observed_imag >= 0 ? '+' : ''}{elem.observed_imag.toFixed(4)}j
              </span>
            </div>
          </div>
        </div>

        {/* CARD 03 — V4 ESTIMATED COMPENSATION */}
        <div className="state-detail-card card-pinn">
          <div className="state-card-header">
            <div className="state-card-num-title">
              <span className="state-mini-badge state-badge-pinn">03</span>
              <h4 className="state-card-title">V4 ESTIMATED COMPENSATION</h4>
            </div>
            <span className="state-role-tag role-pinn">
              {isPinnAvailable ? 'V4 ESTIMATE' : 'ANALYTICAL BASELINE'}
            </span>
          </div>

          <p className="state-card-desc">
            Complex compensation weights ĉ = c_prior + Δc estimated by PINN-Cal V4.
          </p>

          <div className="state-metric-rows">
            <div className="metric-row">
              <span className="metric-label">Estimated Phase (∠ĉ):</span>
              <span className="inspector-metric-val metric-purple">
                {cPhase > 0 ? `+${cPhase.toFixed(2)}°` : `${cPhase.toFixed(2)}°`}
              </span>
            </div>
            <div className="metric-row">
              <span className="metric-label">Estimated Gain (|ĉ|):</span>
              <span className="inspector-metric-val metric-purple">
                {cGainDb > 0 ? `+${cGainDb.toFixed(2)} dB` : `${cGainDb.toFixed(2)} dB`}{' '}
                <span className="val-sub">({cMag.toFixed(3)})</span>
              </span>
            </div>
            <div className="metric-row sub-row">
              <span className="metric-label">Weight Vector (ĉ_n):</span>
              <span className="metric-val-mono">
                {cpRe.toFixed(4)} + {cpIm >= 0 ? '+' : ''}{cpIm.toFixed(4)}j
              </span>
            </div>
          </div>
        </div>

        {/* CARD 04 — POST-COMPENSATION STATE */}
        <div className="state-detail-card card-calibrated">
          <div className="state-card-header">
            <div className="state-card-num-title">
              <span className="state-mini-badge state-badge-calibrated">04</span>
              <h4 className="state-card-title">POST-COMPENSATION STATE</h4>
            </div>
            <span className="state-role-tag role-restored">RESTORED STATE</span>
          </div>

          <p className="state-card-desc">
            Restored array excitation: V_post,n = V_hw,n · ĉ_n.
          </p>

          <div className="state-metric-rows">
            <div className="metric-row">
              <span className="metric-label">Restored Phase:</span>
              <span className="inspector-metric-val metric-green">
                {corrPhase > 0 ? `+${corrPhase.toFixed(2)}°` : `${corrPhase.toFixed(2)}°`}
              </span>
            </div>
            <div className="metric-row">
              <span className="metric-label">Restored Gain:</span>
              <span className="inspector-metric-val metric-green">
                {corrGainDb > 0 ? `+${corrGainDb.toFixed(2)} dB` : `${corrGainDb.toFixed(2)} dB`}{' '}
                <span className="val-sub">({corrMag.toFixed(3)})</span>
              </span>
            </div>
            <div className="metric-row sub-row">
              <span className="metric-label">Residual Phase Error:</span>
              <span className="metric-val-mono highlight-green">
                {displayPhaseRes > 0 ? `+${displayPhaseRes.toFixed(2)}°` : `${displayPhaseRes.toFixed(2)}°`}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 3. SIMULATION BENCHMARK REFERENCE STRIP (Requirement 22) */}
      <div className="inspector-validation-strip" role="note" aria-label="Simulation Benchmark Reference">
        <div className="validation-strip-left">
          <span className="strip-info-icon">🔬</span>
          <span className="strip-title">SIMULATION BENCHMARK REFERENCE:</span>
          <span className="strip-note">Ground truth values from simulation engine — closed to physical neural inference.</span>
        </div>
        <div className="validation-strip-metrics">
          <div className="val-chip">
            <span className="val-k">True Phase Error:</span>
            <span className="val-v">{truePhase > 0 ? `+${truePhase.toFixed(2)}°` : `${truePhase.toFixed(2)}°`}</span>
          </div>
          <div className="val-chip">
            <span className="val-k">True Gain Error:</span>
            <span className="val-v">{trueGain > 0 ? `+${trueGain.toFixed(2)} dB` : `${trueGain.toFixed(2)} dB`}</span>
          </div>
          <div className="val-chip highlight-res">
            <span className="val-k">Phase Residual:</span>
            <span className="val-v">{displayPhaseRes > 0 ? `+${displayPhaseRes.toFixed(2)}°` : `${displayPhaseRes.toFixed(2)}°`}</span>
          </div>
          <div className="val-chip highlight-res">
            <span className="val-k">Gain Residual:</span>
            <span className="val-v">{displayGainRes > 0 ? `+${displayGainRes.toFixed(2)} dB` : `${displayGainRes.toFixed(2)} dB`}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
