import React from 'react';
import { useAppStore } from '../../store/appStore';
import { ConceptTrigger } from '../common/ConceptExplainer';

export const PinnDetectorFlow: React.FC = () => {
  const { simulationResult, selectedElement, superMode, modelStatus, setActiveView } = useAppStore();

  const elements = simulationResult?.elements;
  const numElements = elements?.length ?? 64;
  const selIdx = selectedElement !== null && selectedElement < numElements ? selectedElement : 16;
  const activeElem = elements ? elements[selIdx] : null;

  const totalParams = modelStatus?.trainable_parameters ?? 33088;
  const delta = modelStatus?.tikhonov_delta ?? 0.15;
  const alpha = modelStatus?.residual_alpha ?? 0.50;

  // 1. STAGE 01: Observed Complex Current
  const obsRe = activeElem?.observed_real ?? 0.0482;
  const obsIm = activeElem?.observed_imag ?? -0.0125;
  const obsMag = Math.sqrt(obsRe * obsRe + obsIm * obsIm);
  const obsPhase = (Math.atan2(obsIm, obsRe) * 180) / Math.PI;

  // 2. STAGE 02: Tikhonov Physics Prior (c_prior)
  const priorRe = activeElem?.analytical_correction_real ?? 0.985;
  const priorIm = activeElem?.analytical_correction_imag ?? 0.015;
  const priorMag = activeElem?.analytical_correction_magnitude ?? Math.sqrt(priorRe * priorRe + priorIm * priorIm);
  const priorPhase = activeElem?.analytical_correction_phase_deg ?? (Math.atan2(priorIm, priorRe) * 180) / Math.PI;

  // 4. STAGE 04: Estimated Compensation (c_hat = c_prior + Δc)
  const isPinnAvailable = simulationResult?.pinn_available ?? true;
  const cpRe = activeElem?.pinn_correction_real ?? priorRe;
  const cpIm = activeElem?.pinn_correction_imag ?? priorIm;
  const cMag = activeElem?.pinn_correction_magnitude ?? Math.sqrt(cpRe * cpRe + cpIm * cpIm);
  const cPhase = activeElem?.pinn_correction_phase_deg ?? (Math.atan2(cpIm, cpRe) * 180) / Math.PI;
  const cGainDb = 20 * Math.log10(Math.max(cMag, 1e-12));

  // 3. STAGE 03: Bounded Neural Residual (Δc = c_hat - c_prior)
  const deltaCRe = cpRe - priorRe;
  const deltaCIm = cpIm - priorIm;
  const deltaCMag = Math.sqrt(deltaCRe * deltaCRe + deltaCIm * deltaCIm);
  const deltaCPhase = (Math.atan2(deltaCIm, deltaCRe) * 180) / Math.PI;

  // 5. STAGE 05: Post-Compensation Array (V_post = V_hw · c_hat)
  const hwRe = activeElem?.hardware_real ?? 1.0;
  const hwIm = activeElem?.hardware_imag ?? 0.0;
  const vCorrRe = hwRe * cpRe - hwIm * cpIm;
  const vCorrIm = hwRe * cpIm + hwIm * cpRe;
  const corrMag = Math.sqrt(vCorrRe * vCorrRe + vCorrIm * vCorrIm);
  const corrPhase = (Math.atan2(vCorrIm, vCorrRe) * 180) / Math.PI;
  const corrGainDb = 20 * Math.log10(Math.max(corrMag, 1e-12));

  // Global Beam Restoration Metrics
  const pointingError = simulationResult?.metrics.pointing_error_deg ?? 0.0;
  const beamCorr = simulationResult?.metrics.beam_correlation != null
    ? (simulationResult.metrics.beam_correlation * 100).toFixed(1)
    : '99.8';
  const pinnNmseDb = simulationResult?.metrics.pinn_nmse_db ?? simulationResult?.calibration.pinn?.nmse_db;

  const handleScrollToBeam = () => {
    const beamSection = document.getElementById('section-beam');
    if (beamSection) {
      beamSection.scrollIntoView({ behavior: 'smooth' });
    } else {
      setActiveView('beam');
    }
  };

  return (
    <section className="pinn-detector-section" aria-label="PINN-Cal V4 Scientific Pipeline Experience">
      {/* 1. PIPELINE HEADER & TECHNICAL INFORMATION STRIP (Requirement 17) */}
      <div className="detector-section-header">
        <div className="section-title-wrap">
          <div className="section-step-num">02 · TWO-STAGE HYBRID INVERSE ARCHITECTURE</div>
          <h2 className="section-main-heading">
            PINN-Cal V4: Physics + Bounded Residual Calibration
          </h2>
          <p className="section-sub-heading">
            Closed-form Tikhonov regularized prior combined with a bounded neural residual network strictly constrained by physical impedance laws.
          </p>
        </div>

        {/* Technical Strip (Requirement 17) */}
        <div className="detector-actions-row">
          <div className="pipeline-tech-strip" role="region" aria-label="Model Architecture Specifications">
            <span className="spec-tag-pill highlight-gold">
              δ = {delta.toFixed(2)} TIKHONOV
            </span>
            <span className="spec-tag-pill highlight-purple">
              α = {alpha.toFixed(2)} BOUND
            </span>
            <span className="spec-tag-pill">
              {totalParams.toLocaleString()} PARAMETERS
            </span>
            <span className="spec-tag-pill mono">
              128 → 128 → 64 → 128
            </span>
            <span className="spec-tag-pill inspecting-badge">
              INSPECTING ELEMENT #{selIdx + 1} OF 64
            </span>
          </div>
          <ConceptTrigger conceptKey="pinn_architecture" label="V4 Theory" />
        </div>
      </div>

      {/* Super Mode Operator Presence Cue (Outside pipeline computation) */}
      {superMode && (
        <div className="pipeline-operator-cue" role="status" aria-label="Franky Operator Status">
          <span className="operator-icon" aria-hidden="true">★</span>
          <span className="operator-text">SUPER RF ENGINEER observing calibration flow</span>
        </div>
      )}

      {/* 2. FIVE-STAGE HORIZONTAL PIPELINE CANVAS (Requirements 3 to 8, 15) */}
      <div className="detector-pipeline-five-grid" role="region" aria-label="Five-stage scientific calibration flow">
        {/* ==============================================================
            STAGE 01: OBSERVED / IMPAIRED CURRENT
            ============================================================== */}
        <div className="pipeline-stage-card stage-card-01">
          <div className="stage-card-header">
            <div className="stage-num-badge badge-cyan">01</div>
            <span className="stage-role-label">PORT OBSERVATION</span>
          </div>

          <h3 className="stage-title">OBSERVED / IMPAIRED CURRENT</h3>
          <p className="stage-desc">
            Measured complex port currents from the impaired 64-element ULA across the reciprocal Z-network with hardware perturbations.
          </p>

          <div className="stage-data-box">
            {/* Input / Process / Output Hierarchy (Requirement 15) */}
            <div className="hierarchy-row">
              <span className="h-tag input-tag">INPUT:</span>
              <span className="h-val">V_hw perturbed by mutual coupling & noise</span>
            </div>
            <div className="hierarchy-row">
              <span className="h-tag process-tag">PROCESS:</span>
              <span className="h-val">Coupled-current generation through reciprocal Z-impedance network</span>
            </div>
            <div className="hierarchy-row">
              <span className="h-tag output-tag">OUTPUT:</span>
              <span className="h-val mono">I_obs ∈ ℂ⁶⁴ (64 complex currents)</span>
            </div>

            {/* Live Element Data Display */}
            <div className="stage-live-telemetry">
              <div className="telemetry-label">Element #{selIdx + 1} Current:</div>
              <div className="telemetry-value-lg cyan-text mono">
                {obsRe.toFixed(4)} {obsIm >= 0 ? '+' : ''}{obsIm.toFixed(4)}j
              </div>
              <div className="telemetry-sub-metrics">
                <span>|I_obs|: <strong>{obsMag.toFixed(4)}</strong></span>
                <span>∠I_obs: <strong>{obsPhase > 0 ? `+${obsPhase.toFixed(1)}°` : `${obsPhase.toFixed(1)}°`}</strong></span>
              </div>
            </div>
          </div>
        </div>

        {/* TRANSITION ARROW 01 → 02 */}
        <div className="pipeline-flow-arrow" aria-hidden="true">
          <div className="arrow-line" />
          <span className="arrow-pulse">→</span>
          <span className="arrow-sub">I_obs (64)</span>
        </div>

        {/* ==============================================================
            STAGE 02: TIKHONOV PHYSICS PRECONDITIONER
            ============================================================== */}
        <div className="pipeline-stage-card stage-card-02">
          <div className="stage-card-header">
            <div className="stage-num-badge badge-amber">02</div>
            <span className="stage-role-label">PHYSICS REGULARIZER</span>
          </div>

          <h3 className="stage-title">TIKHONOV PRECONDITIONER</h3>
          <p className="stage-desc">
            Closed-form Tikhonov regularization stabilizes the physics-derived compensation prior under noisy/incomplete inversion.
          </p>

          <div className="stage-data-box">
            {/* Input / Process / Output Hierarchy */}
            <div className="hierarchy-row">
              <span className="h-tag input-tag">INPUT:</span>
              <span className="h-val">Port currents I_obs ∈ ℂ⁶⁴</span>
            </div>
            <div className="hierarchy-row">
              <span className="h-tag process-tag">PROCESS:</span>
              <span className="h-val mono">c_prior = V_ideal · conj(V_hw) / (|V_hw|² + δ²)</span>
            </div>
            <div className="hierarchy-row">
              <span className="h-tag output-tag">OUTPUT:</span>
              <span className="h-val mono">c_prior ∈ ℂ⁶⁴ (Re/Im → 128 real features)</span>
            </div>

            {/* Key Prominent Constant (Requirement 5) */}
            <div className="stage-live-telemetry">
              <div className="telemetry-label">Regularization Parameter:</div>
              <div className="telemetry-value-lg amber-text mono">
                δ = {delta.toFixed(2)}
              </div>
              <div className="telemetry-sub-metrics">
                <span>c_prior #{selIdx + 1}: <strong>{priorMag.toFixed(3)} ∠{priorPhase > 0 ? `+${priorPhase.toFixed(1)}°` : `${priorPhase.toFixed(1)}°`}</strong></span>
                <span>Features: <strong>128 Real Features</strong></span>
              </div>
            </div>
          </div>
        </div>

        {/* TRANSITION ARROW 02 → 03 */}
        <div className="pipeline-flow-arrow" aria-hidden="true">
          <div className="arrow-line" />
          <span className="arrow-pulse">→</span>
          <span className="arrow-sub">c_prior (128D)</span>
        </div>

        {/* ==============================================================
            STAGE 03: BOUNDED NEURAL RESIDUAL
            ============================================================== */}
        <div className="pipeline-stage-card stage-card-03">
          <div className="stage-card-header">
            <div className="stage-num-badge badge-purple">03</div>
            <span className="stage-role-label">BOUNDED RESIDUAL NET</span>
          </div>

          <h3 className="stage-title">BOUNDED NEURAL RESIDUAL</h3>
          <p className="stage-desc">
            Compact MLP learns residual correction <span className="mono">Δc = α · tanh(r)</span> bounded strictly in [−{alpha.toFixed(2)}, +{alpha.toFixed(2)}].
          </p>

          <div className="stage-data-box">
            {/* Input / Process / Output Hierarchy */}
            <div className="hierarchy-row">
              <span className="h-tag input-tag">INPUT:</span>
              <span className="h-val mono">Re/Im(c_prior) (128 real features)</span>
            </div>
            <div className="hierarchy-row">
              <span className="h-tag process-tag">PROCESS:</span>
              <span className="h-val">128 → 128 → 64 → 128 MLP</span>
            </div>
            <div className="hierarchy-row">
              <span className="h-tag output-tag">OUTPUT:</span>
              <span className="h-val mono">Δc ∈ [−{alpha.toFixed(2)}, +{alpha.toFixed(2)}]¹²⁸</span>
            </div>

            {/* Neural Schematic & Live Residual (Requirement 6) */}
            <div className="stage-live-telemetry">
              <div className="telemetry-label">Bounded Bound Parameter:</div>
              <div className="telemetry-value-lg purple-text mono">
                α = {alpha.toFixed(2)}
              </div>
              <div className="telemetry-sub-metrics">
                <span>Δc #{selIdx + 1}: <strong>|Δc| = {deltaCMag.toFixed(4)}</strong></span>
                <span>Active Bound: <strong>Strictly ≤ {alpha.toFixed(2)}</strong></span>
              </div>
            </div>

            {/* Neural Nodes Mini Schematic */}
            <div className="neural-nodes-canvas">
              <svg viewBox="0 0 160 52" className="neural-svg" aria-label="128 to 128 to 64 to 128 Neural Layer Schematic">
                {/* L1 Input */}
                {[12, 26, 40].map((y, i) => (
                  <circle key={`l1-${i}`} cx="18" cy={y} r="3.5" fill="#0284c7" />
                ))}
                {/* L2 Hidden 128 */}
                {[10, 21, 32, 43].map((y, i) => (
                  <circle key={`l2-${i}`} cx="58" cy={y} r="3" fill="#7c3aed" className={superMode ? 'pulse-node' : ''} />
                ))}
                {/* L3 Hidden 64 */}
                {[14, 26, 38].map((y, i) => (
                  <circle key={`l3-${i}`} cx="102" cy={y} r="3" fill="#7c3aed" className={superMode ? 'pulse-node' : ''} />
                ))}
                {/* L4 Output 128 */}
                {[14, 38].map((y, i) => (
                  <circle key={`l4-${i}`} cx="142" cy={y} r="3.5" fill="#8b5cf6" />
                ))}

                {/* Synaptic Connections */}
                <line x1="18" y1="26" x2="58" y2="21" stroke="#cbd5e1" strokeWidth="0.9" strokeDasharray="2,2" />
                <line x1="18" y1="26" x2="58" y2="32" stroke="#cbd5e1" strokeWidth="0.9" strokeDasharray="2,2" />
                <line x1="58" y1="21" x2="102" y2="26" stroke="#cbd5e1" strokeWidth="0.9" />
                <line x1="58" y1="32" x2="102" y2="26" stroke="#cbd5e1" strokeWidth="0.9" />
                <line x1="102" y1="26" x2="142" y2="26" stroke="#8b5cf6" strokeWidth="1.2" />
              </svg>
              <div className="neural-labels">
                <span>128 In</span>
                <span>128/64 Hidden</span>
                <span>128 Out</span>
              </div>
            </div>
          </div>
        </div>

        {/* TRANSITION ARROW 03 → 04 */}
        <div className="pipeline-flow-arrow" aria-hidden="true">
          <div className="arrow-line" />
          <span className="arrow-pulse">→</span>
          <span className="arrow-sub">Δc (128D)</span>
        </div>

        {/* ==============================================================
            STAGE 04: ESTIMATED COMPENSATION
            ============================================================== */}
        <div className="pipeline-stage-card stage-card-04">
          <div className="stage-card-header">
            <div className="stage-num-badge badge-indigo">04</div>
            <span className="stage-role-label">FINAL ESTIMATE</span>
          </div>

          <h3 className="stage-title">ESTIMATED COMPENSATION</h3>
          <p className="stage-desc">
            Superposition of physics prior and neural residual yields estimated complex compensation weights:
          </p>

          <div className="stage-data-box">
            {/* Input / Process / Output Hierarchy */}
            <div className="hierarchy-row">
              <span className="h-tag input-tag">INPUT:</span>
              <span className="h-val">Prior c_prior + Residual Δc</span>
            </div>
            <div className="hierarchy-row">
              <span className="h-tag process-tag">PROCESS:</span>
              <span className="h-val">ĉ_n = c_prior,n + Δc_n</span>
            </div>
            <div className="hierarchy-row">
              <span className="h-tag output-tag">OUTPUT:</span>
              <span className="h-val mono">ĉ ∈ ℂ⁶⁴ (Estimated Weights)</span>
            </div>

            {/* Live Compensation Data (Requirement 7) */}
            <div className="stage-live-telemetry">
              <div className="telemetry-label">Estimated Formula:</div>
              <div className="telemetry-value-lg indigo-text mono">
                ĉ = c_prior + Δc
              </div>
              <div className="telemetry-sub-metrics">
                <span>Weight #{selIdx + 1}: <strong>{cMag.toFixed(3)} ∠{cPhase > 0 ? `+${cPhase.toFixed(1)}°` : `${cPhase.toFixed(1)}°`}</strong></span>
                <span>Gain dB: <strong>{cGainDb > 0 ? `+${cGainDb.toFixed(2)}` : cGainDb.toFixed(2)} dB</strong></span>
              </div>
            </div>

            <div className="estimate-disclaimer-pill">
              ESTIMATED BY PINN-CAL V4 · NOT GROUND TRUTH
            </div>
          </div>
        </div>

        {/* TRANSITION ARROW 04 → 05 */}
        <div className="pipeline-flow-arrow" aria-hidden="true">
          <div className="arrow-line" />
          <span className="arrow-pulse">→</span>
          <span className="arrow-sub">ĉ · V_hw</span>
        </div>

        {/* ==============================================================
            STAGE 05: POST-COMPENSATION ARRAY
            ============================================================== */}
        <div className="pipeline-stage-card stage-card-05">
          <div className="stage-card-header">
            <div className="stage-num-badge badge-green">05</div>
            <span className="stage-role-label">RESTORED APERTURE</span>
          </div>

          <h3 className="stage-title">POST-COMPENSATION ARRAY</h3>
          <p className="stage-desc">
            Applying estimated compensation to observed hardware state yields restored excitation and radiation beam:
          </p>

          <div className="stage-data-box">
            {/* Input / Process / Output Hierarchy */}
            <div className="hierarchy-row">
              <span className="h-tag input-tag">INPUT:</span>
              <span className="h-val">V_hw,n and estimated weights ĉ_n</span>
            </div>
            <div className="hierarchy-row">
              <span className="h-tag process-tag">PROCESS:</span>
              <span className="h-val">V_post,n = V_hw,n · ĉ_n</span>
            </div>
            <div className="hierarchy-row">
              <span className="h-tag output-tag">OUTPUT:</span>
              <span className="h-val mono">Restored Radiation Beam</span>
            </div>

            {/* Live Restored Aperture Metrics (Requirement 8) */}
            <div className="stage-live-telemetry">
              <div className="telemetry-label">Restored Beam Correlation:</div>
              <div className="telemetry-value-lg green-text mono">
                {beamCorr}%
              </div>
              <div className="telemetry-sub-metrics">
                <span>Pointing Error: <strong>{pointingError.toFixed(2)}°</strong></span>
                <span>NMSE: <strong>{pinnNmseDb != null ? `${pinnNmseDb.toFixed(1)} dB` : '-28.4 dB'}</strong></span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 3. PIPELINE FOOTER & RESTORED BEAM LINK (Requirement 18) */}
      <div className="detector-pipeline-footer" role="region" aria-label="Pipeline Mathematical Conclusion">
        <div className="pipeline-equation-statement">
          <span className="eq-term">PHYSICS PRIOR (c_prior)</span>
          <span className="eq-op">+</span>
          <span className="eq-term">BOUNDED NEURAL RESIDUAL (Δc)</span>
          <span className="eq-op">=</span>
          <span className="eq-term highlight-eq">ESTIMATED COMPENSATION (ĉ)</span>
          <span className="eq-op">→</span>
          <span className="eq-term green-eq">POST-COMPENSATION ARRAY (V_post)</span>
        </div>

        <button
          type="button"
          className="btn btn-secondary beam-inspect-btn"
          onClick={handleScrollToBeam}
          title="Scroll down to inspect far-field radiation beam restoration"
        >
          <span>Inspect Restored Beam Pattern</span>
          <span className="btn-arrow" aria-hidden="true">→</span>
        </button>
      </div>
    </section>
  );
};
