import React from 'react';
import { PageContainer } from '../components/common/PageContainer';
import { Panel } from '../components/common/Panel';
import { MetricCard } from '../components/common/MetricCard';
import { useAppStore } from '../store/appStore';

export const OverviewPage: React.FC = () => {
  const { modelStatus, systemStatus, setActiveView } = useAppStore();

  const paramCount = modelStatus?.trainable_parameters
    ? modelStatus.trainable_parameters.toLocaleString()
    : '33,088';
  const deltaVal = modelStatus?.tikhonov_delta ?? 0.15;
  const alphaVal = modelStatus?.residual_alpha ?? 0.50;
  const gpuName = systemStatus?.gpu_name || 'NVIDIA GTX 1650';

  const handleSuperCalibrate = () => {
    setActiveView('simulation');
  };

  return (
    <PageContainer className="overview-page-container">
      {/* =========================================================================
          1. OVERVIEW HERO (Section 5, 7, 8)
          ========================================================================= */}
      <section className="overview-hero-section" aria-label="PINN-Cal V4 Main Overview Hero">
        <div className="overview-hero-badge-row">
          <span className="hero-badge-pill hero-badge-v4">
            <span className="badge-dot" aria-hidden="true" />
            PINN-CAL V4 FROZEN RELEASE
          </span>
          <span className="hero-badge-pill hero-badge-spec">
            64-ELEMENT 5G MASSIVE-MIMO
          </span>
          <span className="hero-badge-pill hero-badge-freq">
            28 GHz mmWAVE
          </span>
        </div>

        <h1 className="overview-hero-title">
          CALIBRATE THE ARRAY.<br />
          <span className="overview-hero-title-accent">RESTORE THE BEAM.</span>
        </h1>

        <p className="overview-hero-description">
          Physics-informed neural calibration for a 64-element 5G Massive-MIMO antenna array.
          Coupling closed-form Tikhonov electromagnetic preconditioning with bounded neural residual learning
          to compensate mutual coupling, phase noise, and gain drift in real time.
        </p>

        {/* Compact Technical Identity Row */}
        <div className="overview-hero-spec-strip" aria-label="Technical Specifications">
          <div className="spec-strip-chip">
            <span className="chip-key">ARRAY</span>
            <span className="chip-val">64 ELEMENTS</span>
          </div>
          <span className="spec-strip-sep" aria-hidden="true">/</span>
          <div className="spec-strip-chip">
            <span className="chip-key">CARRIER</span>
            <span className="chip-val">28 GHz</span>
          </div>
          <span className="spec-strip-sep" aria-hidden="true">/</span>
          <div className="spec-strip-chip">
            <span className="chip-key">SPACING</span>
            <span className="chip-val">0.50 λ</span>
          </div>
          <span className="spec-strip-sep" aria-hidden="true">/</span>
          <div className="spec-strip-chip">
            <span className="chip-key">MODEL</span>
            <span className="chip-val highlight">PINN-Cal V4</span>
          </div>
          <span className="spec-strip-sep" aria-hidden="true">/</span>
          <div className="spec-strip-chip">
            <span className="chip-key">WEIGHTS</span>
            <span className="chip-val">{paramCount} PARAMS</span>
          </div>
        </div>

        {/* Primary Action Button Station (Section 7) */}
        <div className="overview-hero-cta-group">
          <button
            type="button"
            className="hero-super-calibrate-btn"
            onClick={handleSuperCalibrate}
            aria-label="Super Calibrate: Launch live 64-element calibration workflow"
          >
            <span className="btn-star-icon" aria-hidden="true">★</span>
            <span className="btn-text-main">SUPER CALIBRATE</span>
            <span className="btn-arrow" aria-hidden="true">→</span>
          </button>

          <button
            type="button"
            className="hero-secondary-btn"
            onClick={() => setActiveView('research')}
            aria-label="View verified blind holdout benchmarks and research data"
          >
            <span className="btn-icon-sec">🛡️</span>
            <span>View Holdout Benchmarks</span>
          </button>
        </div>
      </section>

      {/* =========================================================================
          2. HERO VISUAL: 4-STAGE CALIBRATION STORY (Section 6)
          IDEAL → IMPAIRED → V4 CALIBRATED → RESTORED BEAM
          ========================================================================= */}
      <section className="overview-story-instrument-panel" aria-label="Aperture Calibration Progression">
        <div className="instrument-panel-header">
          <div className="instrument-title-group">
            <span className="instrument-badge">CALIBRATION PIPELINE STORY</span>
            <h2 className="instrument-main-title">Aperture Distortion & Restoration Lifecycle</h2>
          </div>
          <span className="instrument-subtitle">
            Physical impairment lifecycle across the 64-element 28 GHz antenna aperture
          </span>
        </div>

        <div className="instrument-stages-grid">
          {/* Stage 1: Ideal */}
          <div className="instrument-stage-card stage-ideal">
            <div className="stage-card-top">
              <span className="stage-step-tag">01 / REFERENCE</span>
              <span className="stage-status-dot dot-ideal" aria-hidden="true" />
            </div>
            <div className="stage-card-visual ideal-aperture-visual">
              <div className="aperture-mini-grid ideal-grid">
                {Array.from({ length: 16 }).map((_, i) => (
                  <span key={i} className="mini-patch patch-ideal" />
                ))}
              </div>
            </div>
            <h3 className="stage-card-title">1. Ideal Excitation</h3>
            <p className="stage-card-desc">
              Uniform unit amplitude <code className="tech-code">1.0∠0°</code> across all 64 elements, producing nominal boresight radiation.
            </p>
            <div className="stage-card-metric">
              <span className="metric-k">AMPLITUDE</span>
              <span className="metric-v">1.000 / 0.00°</span>
            </div>
          </div>

          <div className="stage-flow-arrow" aria-hidden="true">→</div>

          {/* Stage 2: Impaired */}
          <div className="instrument-stage-card stage-impaired">
            <div className="stage-card-top">
              <span className="stage-step-tag">02 / HARDWARE</span>
              <span className="stage-status-dot dot-impaired" aria-hidden="true" />
            </div>
            <div className="stage-card-visual impaired-aperture-visual">
              <div className="aperture-mini-grid impaired-grid">
                {Array.from({ length: 16 }).map((_, i) => (
                  <span key={i} className={`mini-patch patch-impaired patch-p${(i % 4) + 1}`} />
                ))}
              </div>
            </div>
            <h3 className="stage-card-title">2. Physical Impairments</h3>
            <p className="stage-card-desc">
              Coupling matrix <code className="tech-code">Z_A</code>, phase noise <code className="tech-code">σ=20°</code>, and gain drift <code className="tech-code">2 dB</code> distort array wavefront.
            </p>
            <div className="stage-card-metric">
              <span className="metric-k">DEGRADATION</span>
              <span className="metric-v alert">NMSE &gt; -6 dB</span>
            </div>
          </div>

          <div className="stage-flow-arrow" aria-hidden="true">→</div>

          {/* Stage 3: V4 Calibrated */}
          <div className="instrument-stage-card stage-calibrated">
            <div className="stage-card-top">
              <span className="stage-step-tag">03 / PINN-CAL V4</span>
              <span className="stage-status-dot dot-calibrated" aria-hidden="true" />
            </div>
            <div className="stage-card-visual calibrated-aperture-visual">
              <div className="neural-mini-flow">
                <span className="neural-node">TIK δ={deltaVal}</span>
                <span className="neural-plus">+</span>
                <span className="neural-node">RES α={alphaVal}</span>
              </div>
            </div>
            <h3 className="stage-card-title">3. V4 Neural Estimator</h3>
            <p className="stage-card-desc">
              Tikhonov preconditioning (<code className="tech-code">δ={deltaVal}</code>) plus bounded residual (<code className="tech-code">α={alphaVal}</code>) calculates compensation weights.
            </p>
            <div className="stage-card-metric">
              <span className="metric-k">ARCHITECTURE</span>
              <span className="metric-v highlight">{paramCount} Params</span>
            </div>
          </div>

          <div className="stage-flow-arrow" aria-hidden="true">→</div>

          {/* Stage 4: Restored Beam */}
          <div className="instrument-stage-card stage-restored">
            <div className="stage-card-top">
              <span className="stage-step-tag">04 / RESTORATION</span>
              <span className="stage-status-dot dot-restored" aria-hidden="true" />
            </div>
            <div className="stage-card-visual beam-mini-visual">
              <div className="mini-beam-lobe" />
              <div className="mini-beam-side" />
            </div>
            <h3 className="stage-card-title">4. Restored Beam</h3>
            <p className="stage-card-desc">
              Near-perfect spatial reconstruction, eliminating steering deviation and restoring sharp radiation directivity.
            </p>
            <div className="stage-card-metric">
              <span className="metric-k">CORRELATION</span>
              <span className="metric-v success">0.9998 Fidelity</span>
            </div>
          </div>
        </div>
      </section>

      {/* =========================================================================
          3. OVERVIEW METRICS "AT A GLANCE" (Section 11)
          ========================================================================= */}
      <section className="overview-at-a-glance-section" aria-label="Key Project Metrics At a Glance">
        <div className="at-a-glance-header">
          <h2 className="at-a-glance-title">System Metrics at a Glance</h2>
          <span className="at-a-glance-caption">Verified specifications from the frozen PINN-Cal V4 release</span>
        </div>

        <div className="overview-metrics-grid">
          <MetricCard
            label="ARRAY APERTURE"
            value="64"
            unit="ELEMENTS"
            description="64-Element Uniform Linear Array (ULA) @ 28 GHz"
            status="normal"
            icon="📡"
          />
          <MetricCard
            label="CARRIER FREQUENCY"
            value="28"
            unit="GHz"
            description="5G mmWave Band with 0.50 λ Inter-Element Spacing"
            status="normal"
            icon="⚡"
          />
          <MetricCard
            label="NEURAL WEIGHTS"
            value={paramCount}
            unit="PARAMS"
            description="Bounded Residual MLP (128 → 128 → 64 → 128)"
            status="accent"
            icon="🧠"
          />
          <MetricCard
            label="TIKHONOV FACTOR"
            value={`δ = ${deltaVal.toFixed(2)}`}
            unit="PHYSICS"
            description="Regularized Mutual Coupling Matrix Inverse"
            status="success"
            icon="⚖️"
          />
          <MetricCard
            label="RESIDUAL BOUND"
            value={`α = ${alphaVal.toFixed(2)}`}
            unit="SCALE"
            description="Hyperbolic Tangent Residual Ceiling"
            status="warning"
            icon="🛡️"
          />
          <MetricCard
            label="BLIND HOLDOUT"
            value="5,000"
            unit="SAMPLES"
            description="10 Verified Physical Validation Conditions"
            status="accent"
            icon="📊"
          />
        </div>
      </section>

      {/* =========================================================================
          4. OVERVIEW INFORMATION BLOCKS (Section 9)
          A. ARRAY | B. PHYSICS | C. NEURAL CALIBRATION | D. VALIDATION
          ========================================================================= */}
      <section className="overview-info-blocks-section" aria-label="Scientific Overview Blocks">
        <div className="overview-info-grid">
          {/* Block A: Array */}
          <Panel
            title="A. 64-Element Array"
            badge={<span className="panel-badge-info">Aperture</span>}
            subtitle="Uniform Linear Array configuration"
            elevated
          >
            <ul className="info-block-list">
              <li>
                <strong>64-Element Uniform Linear Array (ULA):</strong> Centered element positions designed for 5G mmWave Massive-MIMO.
              </li>
              <li>
                <strong>28 GHz Carrier Frequency:</strong> High-bandwidth operational frequency with λ = 10.71 mm wavelength.
              </li>
              <li>
                <strong>0.5 λ Inter-Element Spacing:</strong> Strict half-wavelength spatial grid (5.36 mm) avoiding grating lobes across ±60° scan angles.
              </li>
              <li>
                <strong>Reciprocal Array Physics:</strong> Passive mutual-coupling array geometry with verified electromagnetic reciprocity.
              </li>
            </ul>
          </Panel>

          {/* Block B: Physics */}
          <Panel
            title="B. Electromagnetic Physics"
            badge={<span className="panel-badge-info">Preconditioner</span>}
            subtitle="Closed-form mutual coupling prior"
            elevated
          >
            <ul className="info-block-list">
              <li>
                <strong>Mutual Coupling Matrix:</strong> Full-rank complex impedance operator <code className="tech-code">Z_A ∈ ℂ^(64×64)</code> modeling inter-element mutual coupling.
              </li>
              <li>
                <strong>Circuit-Theoretic Current Coupling:</strong> Observed currents follow <code className="tech-code">I = (Z_L + Z_A)^(-1) V_oc + n</code> under front-end noise.
              </li>
              <li>
                <strong>Tikhonov Physics Inversion:</strong> Closed-form regularized inverse with regularization parameter <code className="tech-code">δ = {deltaVal}</code>.
              </li>
              <li>
                <strong>Noise Regularization:</strong> Prevents ill-conditioned matrix inversion without requiring empirical parameter search.
              </li>
            </ul>
          </Panel>

          {/* Block C: Neural Calibration */}
          <Panel
            title="C. Bounded Neural Denoiser"
            badge={<span className="panel-badge-planning">V4 Architecture</span>}
            subtitle="Residual non-linear learning"
            elevated
          >
            <ul className="info-block-list">
              <li>
                <strong>Bounded Residual Principle:</strong> The network predicts only small correction residuals <code className="tech-code">Δc</code> bounded by <code className="tech-code">α = {alphaVal}</code>.
              </li>
              <li>
                <strong>Architecture (128 → 128 → 64 → 128):</strong> Input is 128 real features representing Re/Im(c_prior) → 128 → 64 → 128 hidden → 128 real values representing Δc, yielding <code className="tech-code">ĉ = c_prior + Δc</code>.
              </li>
              <li>
                <strong>33,088 Trainable Parameters:</strong> Lightweight model footprint executing in ~1.5 ms on NVIDIA GeForce GTX 1650.
              </li>
              <li>
                <strong>Bounded Activation:</strong> Hyperbolic tangent output stage enforces strict theoretical error bounds: <code className="tech-code">[-0.50, +0.50]</code>.
              </li>
            </ul>
          </Panel>

          {/* Block D: Validation */}
          <Panel
            title="D. Scientific Validation"
            badge={<span className="panel-badge-info">Holdout Benchmark</span>}
            subtitle="Controlled split & 5,000 blind sample evaluation"
            elevated
          >
            <ul className="info-block-list">
              <li>
                <strong>Controlled Split:</strong> 800 Train / 200 Validation / 200 Test for reproducible calibration modeling.
              </li>
              <li>
                <strong>5,000-Sample Final Blind Holdout:</strong> Separate closed final holdout across 10 conditions (closed for model selection).
              </li>
              <li>
                <strong>Normalized Mean Square Error:</strong> Achieves <code className="tech-code">-14.81 dB NMSE</code>, outperforming analytical baseline (<code className="tech-code">-14.23 dB</code>).
              </li>
              <li>
                <strong>Spatial Beam Directivity:</strong> Demonstrates <code className="tech-code">0.0253°</code> pointing error and <code className="tech-code">0.99979</code> beam correlation.
              </li>
            </ul>
          </Panel>
        </div>
      </section>

      {/* =========================================================================
          5. SCIENTIFIC STORY SECTION: HOW PINN-CAL WORKS (Section 10)
          01 Observed Current → 02 Tikhonov → 03 Neural Residual → 04 Compensation → 05 Post-Comp
          ========================================================================= */}
      <section className="overview-how-it-works-section" aria-label="How PINN-Cal Works Scientific Story">
        <Panel
          title="How PINN-Cal V4 Works: The Two-Stage Hybrid Pipeline"
          badge={<span className="panel-badge-info">End-to-End Pipeline</span>}
          subtitle="Combining electromagnetic physics preconditioning with bounded deep learning"
          elevated
        >
          <div className="pinn-pipeline-steps-container">
            {/* Step 1 */}
            <div className="pipeline-step-card">
              <div className="step-badge-col">
                <span className="step-circle">01</span>
                <span className="step-line" aria-hidden="true" />
              </div>
              <div className="step-content-col">
                <div className="step-title-row">
                  <h4 className="step-heading">OBSERVED / IMPAIRED CURRENTS</h4>
                  <span className="step-type-pill pill-hw">RF Front-End</span>
                </div>
                <p className="step-paragraph">
                  Impaired RF front-end array signals <code className="tech-code">I = (Z_L + Z_A)^(-1) V_oc + n</code> incorporate
                  inter-element mutual coupling impedance <code className="tech-code">Z_A</code>, random phase noise (<code className="tech-code">σ_θ=20°</code>),
                  gain drift (<code className="tech-code">σ_g=2 dB</code>), and thermal noise.
                </p>
              </div>
            </div>

            {/* Step 2 */}
            <div className="pipeline-step-card">
              <div className="step-badge-col">
                <span className="step-circle">02</span>
                <span className="step-line" aria-hidden="true" />
              </div>
              <div className="step-content-col">
                <div className="step-title-row">
                  <h4 className="step-heading">TIKHONOV PHYSICS PRECONDITIONER</h4>
                  <span className="step-type-pill pill-physics">Stage 1: Physics Prior</span>
                </div>
                <p className="step-paragraph">
                  A closed-form physical inverse operator stabilizes noise inversion:
                  <code className="tech-code">c_prior = (Z_A^H Z_A + δ^2 I)^(-1) Z_A^H I</code> with regularization parameter
                  <code className="tech-code">δ = {deltaVal}</code>. This resolves ~90% of mutual coupling without machine learning.
                </p>
              </div>
            </div>

            {/* Step 3 */}
            <div className="pipeline-step-card">
              <div className="step-badge-col">
                <span className="step-circle">03</span>
                <span className="step-line" aria-hidden="true" />
              </div>
              <div className="step-content-col">
                <div className="step-title-row">
                  <h4 className="step-heading">BOUNDED NEURAL RESIDUAL DENOISER</h4>
                  <span className="step-type-pill pill-neural">Stage 2: 33,088 Params</span>
                </div>
                <p className="step-paragraph">
                  A compact residual MLP (128 → 128 → 64 → 128) takes 128 real features [Re/Im(c_prior)] and predicts 128 bounded residual values:
                  <code className="tech-code">Δc = α · tanh(r_θ(Re/Im(c_prior)))</code> with ceiling <code className="tech-code">α = {alphaVal}</code>.
                </p>
              </div>
            </div>

            {/* Step 4 */}
            <div className="pipeline-step-card">
              <div className="step-badge-col">
                <span className="step-circle">04</span>
                <span className="step-line" aria-hidden="true" />
              </div>
              <div className="step-content-col">
                <div className="step-title-row">
                  <h4 className="step-heading">ESTIMATED COMPENSATION WEIGHTS</h4>
                  <span className="step-type-pill pill-comp">Estimated Weights</span>
                </div>
                <p className="step-paragraph">
                  The final per-element inverse compensation vector is assembled:
                  <code className="tech-code">c_estimated = c_prior + Δc</code>.
                  Each element receives a complex calibration coefficient <code className="tech-code">ĉ_n = |ĉ_n| · e^(j φ_n)</code>.
                </p>
              </div>
            </div>

            {/* Step 5 */}
            <div className="pipeline-step-card">
              <div className="step-badge-col">
                <span className="step-circle">05</span>
              </div>
              <div className="step-content-col">
                <div className="step-title-row">
                  <h4 className="step-heading">POST-COMPENSATION ARRAY & RESTORED BEAM</h4>
                  <span className="step-type-pill pill-restored">Aperture Restored</span>
                </div>
                <p className="step-paragraph">
                  Applying compensation weights to transmit excitations yields restored voltages:
                  <code className="tech-code">V_post = V_hw ⊙ ĉ_pinn</code>, recovering the ideal far-field radiation beam
                  with sharp boresight directivity, pointing deviation below <code className="tech-code">0.05°</code>, and 0.9998 correlation.
                </p>
              </div>
            </div>
          </div>
        </Panel>
      </section>

      {/* =========================================================================
          6. RUNTIME FOOTER STRIP
          ========================================================================= */}
      <footer className="overview-runtime-footer-strip">
        <div className="footer-runtime-info">
          <span className="footer-status-dot online" aria-hidden="true" />
          <span>Backend: <strong>FastAPI Active</strong></span>
          <span className="footer-sep">·</span>
          <span>Device: <strong>{gpuName}</strong></span>
          <span className="footer-sep">·</span>
          <span>Checkpoint: <strong>pinn_v4_tikhonov015_seed42.pth</strong> (Frozen)</span>
        </div>
        <button
          type="button"
          className="footer-quick-bench-btn"
          onClick={handleSuperCalibrate}
        >
          <span>Launch Live Array Lab</span>
          <span aria-hidden="true">→</span>
        </button>
      </footer>
    </PageContainer>
  );
};

export default OverviewPage;
