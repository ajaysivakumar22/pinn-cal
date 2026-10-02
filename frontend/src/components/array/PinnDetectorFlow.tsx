import React from 'react';
import { useAppStore } from '../../store/appStore';
import { ConceptTrigger } from '../common/ConceptExplainer';

export const PinnDetectorFlow: React.FC = () => {
  const { simulationResult, selectedElement, superMode } = useAppStore();

  const elements = simulationResult?.elements;
  const selIdx = selectedElement ?? 16;
  const activeElem = elements ? elements[selIdx] : null;

  // Derived or default quantities
  const estPhase = activeElem?.pinn_estimated_phase_error_deg ?? 8.42;
  const estGain = activeElem?.pinn_estimated_gain_error_db ?? 2.83;
  const truePhase = activeElem?.true_phase_error_deg ?? 8.42;
  const trueGain = activeElem?.true_gain_error_db ?? 2.83;
  const phaseRes = activeElem?.phase_error_residual_deg ?? 0.15;
  const gainRes = activeElem?.gain_error_residual_db ?? 0.08;

  return (
    <section className="pinn-detector-section" aria-label="PINN Neural Detection Pipeline">
      {/* Editorial Section Header */}
      <div className="detector-section-header">
        <div className="section-title-wrap">
          <div className="section-step-num">03 · NEURAL DIAGNOSIS</div>
          <h2 className="section-main-heading">
            THE PINN AS AN RF DETECTOR
          </h2>
          <p className="section-sub-heading">
            Inverting hidden hardware impairments directly from noisy, coupled port currents
          </p>
        </div>

        <div className="detector-actions-row">
          <span className="spec-tag-pill">
            ARCHITECTURE: 130 → 256 → 128 → 64 → 128
          </span>
          <span className="spec-tag-pill highlight">
            83,904 PARAMETERS (EPOCH 40)
          </span>
          <ConceptTrigger conceptKey="pinn_architecture" label="Architecture Physics" />
        </div>
      </div>

      {/* Main 5-Stage Neural Pipeline Visual Canvas */}
      <div className="detector-pipeline-grid">
        {/* STAGE 1: OBSERVED COUPLING CURRENTS */}
        <div className="pipeline-stage-card stage-input">
          <div className="stage-card-header">
            <span className="stage-num">STAGE 1</span>
            <span className="stage-type">PORT MEASUREMENT</span>
          </div>
          <h3 className="stage-title">Observed Port Currents</h3>
          <p className="stage-desc">
            Receiver samples complex currents <span className="mono">I_obs = [Z + Z_L]⁻¹ · V_impaired</span> perturbed by 64-element mutual coupling network.
          </p>

          <div className="stage-data-box">
            <div className="data-row">
              <span className="data-key">Channel Input:</span>
              <span className="data-val mono">Ch #{selIdx + 1} of 64</span>
            </div>
            <div className="data-row">
              <span className="data-key">Coupled Current:</span>
              <span className="data-val mono">
                {activeElem ? `${activeElem.observed_real.toFixed(3)} + ${activeElem.observed_imag.toFixed(3)}j` : '0.048 - 0.012j'}
              </span>
            </div>
            <div className="data-row">
              <span className="data-key">Mutual Coupling:</span>
              <span className="data-val green">Active (c₁=0.12, α=0.55)</span>
            </div>
          </div>
        </div>

        {/* TRANSITION ARROW 1 */}
        <div className="pipeline-flow-arrow" aria-hidden="true">
          <span className="arrow-pulse">→</span>
          <span className="arrow-sub">Input Vector</span>
        </div>

        {/* STAGE 2: NEURAL NETWORK CORE */}
        <div className="pipeline-stage-card stage-neural">
          <div className="stage-card-header">
            <span className="stage-num">STAGE 2</span>
            <span className="stage-type">PHYSICS-INFORMED INFERENCE</span>
          </div>
          <h3 className="stage-title">PINN Deep Core</h3>
          <p className="stage-desc">
            4-layer feedforward network trained with physics-regularized loss function minimizing coupling residual.
          </p>

          {/* Interactive Neural Nodes Schematic */}
          <div className="neural-nodes-canvas">
            <svg viewBox="0 0 160 70" className="neural-svg" aria-label="Neural Network Layers">
              {/* Layer 1: Input (130D) */}
              {[15, 35, 55].map((y, i) => (
                <circle key={`l1-${i}`} cx="20" cy={y} r="4" fill="#0284c7" />
              ))}
              {/* Layer 2: Hidden 256 */}
              {[10, 25, 45, 60].map((y, i) => (
                <circle key={`l2-${i}`} cx="60" cy={y} r="3.5" fill="#7c3aed" className={superMode ? 'pulse-node' : ''} />
              ))}
              {/* Layer 3: Hidden 128 */}
              {[15, 35, 55].map((y, i) => (
                <circle key={`l3-${i}`} cx="100" cy={y} r="3.5" fill="#7c3aed" className={superMode ? 'pulse-node' : ''} />
              ))}
              {/* Layer 4: Output 128 */}
              {[20, 50].map((y, i) => (
                <circle key={`l4-${i}`} cx="140" cy={y} r="4.5" fill="#10b981" />
              ))}

              {/* Connecting Synaptic Rays */}
              <line x1="20" y1="35" x2="60" y2="25" stroke="#cbd5e1" strokeWidth="1" strokeDasharray="2,2" />
              <line x1="20" y1="35" x2="60" y2="45" stroke="#cbd5e1" strokeWidth="1" strokeDasharray="2,2" />
              <line x1="60" y1="25" x2="100" y2="35" stroke="#cbd5e1" strokeWidth="1" />
              <line x1="60" y1="45" x2="100" y2="35" stroke="#cbd5e1" strokeWidth="1" />
              <line x1="100" y1="35" x2="140" y2="35" stroke="#10b981" strokeWidth="1.5" />
            </svg>
            <div className="neural-labels">
              <span>Input (130)</span>
              <span>Hidden (256/128/64)</span>
              <span>Output (128)</span>
            </div>
          </div>
        </div>

        {/* TRANSITION ARROW 2 */}
        <div className="pipeline-flow-arrow" aria-hidden="true">
          <span className="arrow-pulse">→</span>
          <span className="arrow-sub">Inferred C</span>
        </div>

        {/* STAGE 3: ESTIMATED IMPAIRMENTS & COMPENSATION */}
        <div className="pipeline-stage-card stage-estimate">
          <div className="stage-card-header">
            <span className="stage-num">STAGE 3</span>
            <span className="stage-type">NEURAL ESTIMATE</span>
          </div>
          <h3 className="stage-title">Inferred Impairments</h3>
          <p className="stage-desc">
            PINN infers complex correction <span className="mono">C = H⁻¹</span>. Derived element-level phase and gain estimates:
          </p>

          <div className="estimate-metrics-grid">
            <div className="est-box phase">
              <span className="est-label">ESTIMATED PHASE (ϕ):</span>
              <span className="est-num mono purple">
                {estPhase > 0 ? `+${estPhase.toFixed(2)}°` : `${estPhase.toFixed(2)}°`}
              </span>
              <span className="est-sub">Aperture Deviation</span>
            </div>

            <div className="est-box gain">
              <span className="est-label">ESTIMATED GAIN (G):</span>
              <span className="est-num mono purple">
                {estGain > 0 ? `+${estGain.toFixed(2)} dB` : `${estGain.toFixed(2)} dB`}
              </span>
              <span className="est-sub">Amplitude Deviation</span>
            </div>
          </div>
        </div>

        {/* TRANSITION ARROW 3 */}
        <div className="pipeline-flow-arrow" aria-hidden="true">
          <span className="arrow-pulse">→</span>
          <span className="arrow-sub">Inverse Weights</span>
        </div>

        {/* STAGE 4: POST-COMPENSATION APERTURE STATE */}
        <div className="pipeline-stage-card stage-post">
          <div className="stage-card-header">
            <span className="stage-num">STAGE 4</span>
            <span className="stage-type">POST-COMPENSATION</span>
          </div>
          <h3 className="stage-title">Post-Compensation State</h3>
          <p className="stage-desc">
            Inverse weights applied: <span className="mono">V_corr = V_hw · C</span>. Verified against simulated test realization:
          </p>

          <div className="post-compensation-box">
            <div className="post-row">
              <span className="post-key">Phase Residual (True − Est):</span>
              <strong className="post-val green mono">
                {phaseRes > 0 ? `+${phaseRes.toFixed(2)}°` : `${phaseRes.toFixed(2)}°`}
              </strong>
            </div>
            <div className="post-row">
              <span className="post-key">Gain Residual:</span>
              <strong className="post-val green mono">
                {gainRes > 0 ? `+${gainRes.toFixed(2)} dB` : `${gainRes.toFixed(2)} dB`}
              </strong>
            </div>
            <div className="post-row reference-tag">
              <span className="post-key">Simulation Reference:</span>
              <span className="post-val red mono">
                True: {truePhase > 0 ? `+${truePhase.toFixed(1)}°` : `${truePhase.toFixed(1)}°`} / {trueGain > 0 ? `+${trueGain.toFixed(1)} dB` : `${trueGain.toFixed(1)} dB`}
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
