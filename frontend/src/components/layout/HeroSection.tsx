import React from 'react';
import { useAppStore } from '../../store/appStore';

export const HeroSection: React.FC = () => {
  const {
    simulationResult,
    simulateState,
    superMode,
    toggleSuperMode,
    config,
    setConfig,
    setIsHowItWorksOpen,
    workflowStage,
  } = useAppStore();

  const isComputing = simulateState === 'computing';
  const isComplete = workflowStage === 'completed' || workflowStage === 'verify';
  const targetAngle = config.target_angle_deg;

  // Real-time status text based on computation workflow state (non-interactive indicator)
  const getStatusContent = () => {
    if (simulateState === 'error') {
      return 'CALIBRATION FAILED';
    }
    if (isComputing) {
      switch (workflowStage) {
        case 'observe':
          return '01 OBSERVING CURRENTS...';
        case 'diagnose':
          return '02 DIAGNOSING ERRORS...';
        case 'calibrate':
          return '03 COMPUTING COMPENSATION...';
        case 'restore':
          return '04 RESTORING EXCITATION...';
        case 'verify':
          return '05 VERIFYING...';
        default:
          return 'CALIBRATING ARRAY...';
      }
    }
    if (isComplete || simulateState === 'success') {
      return '★ CALIBRATION COMPLETE';
    }
    return 'CALIBRATION READY';
  };

  return (
    <section className="hero-section" aria-label="PINN-Cal Hero Introduction">
      {/* Background Decorative Tech Grid & Subtle Radiation Gradients */}
      <div className="hero-backdrop">
        <div className="hero-grid-pattern" />
        <div className="hero-glow-orb hero-glow-left" />
        <div className="hero-glow-orb hero-glow-right" />
      </div>

      <div className="hero-content">
        {/* Top Badges & Operational Indicators */}
        <div className="hero-badge-row">
          <div className="hero-badge brand-badge">
            <span className="star-icon">★</span>
            <span className="brand-name">PINN-CAL</span>
            <span className="badge-divider">|</span>
            <span className="badge-sub">RF CALIBRATION LAB</span>
          </div>

          <div className="hero-badge status-badge">
            <span className="hero-live-dot" />
            <span>64-ELEMENT ULA · 28 GHz mmWAVE</span>
          </div>

          <div className="hero-badge spec-badge">
            <span>TARGET STEERING: </span>
            <strong style={{ color: '#0284c7', marginLeft: 4 }}>
              {targetAngle > 0 ? `+${targetAngle}°` : `${targetAngle}°`}
            </strong>
          </div>

          <button
            className={`hero-super-toggle ${superMode ? 'active' : ''}`}
            onClick={toggleSuperMode}
            title="Toggle Franky & Super Mode presentation layer"
            aria-label="Toggle Super Mode"
          >
            <span>★ SUPER MODE</span>
            <span className="super-status-pill">{superMode ? 'ON' : 'OFF'}</span>
          </button>
        </div>

        {/* Main Editorial Headlines with MonkCuts Energy */}
        <div className="hero-headline-group">
          <h1 className="hero-main-title">
            THE ANTENNA IS IMPAIRED.
            <br />
            <span className="hero-gradient-text">LET'S RESTORE THE BEAM.</span>
          </h1>

          <p className="hero-subtitle">
            Physics-Informed Neural Calibration for 5G Massive MIMO Antenna Arrays
          </p>

          {/* Quick Scientific Story Breadcrumb */}
          <div className="hero-story-strip">
            <span className="story-step">Observed RF Currents</span>
            <span className="story-arrow">→</span>
            <span className="story-step highlight-purple">PINN Diagnosis</span>
            <span className="story-arrow">→</span>
            <span className="story-step">Inverse Compensation</span>
            <span className="story-arrow">→</span>
            <span className="story-step highlight-green">Restored Beam</span>
          </div>

          <p className="hero-narrative">
            Fabrication tolerances and mutual coupling perturb transceiver phase and gain, deflecting the radiation main lobe.
            PINN-Cal observes the coupled port currents, infers the inverse compensation matrix <span className="code-pill">C = H⁻¹</span>, and restores spatial radiation fidelity.
          </p>
        </div>

        {/* Interactive Action Command Center with Tactile Steering Preview */}
        <div className="hero-actions-row">
          {/* Non-Interactive Real-Time Calibration Status Badge */}
          <div
            className={`hero-status-indicator ${isComputing ? 'computing' : isComplete ? 'completed' : ''}`}
            aria-live="polite"
          >
            <span className="status-indicator-dot" />
            <span className="status-indicator-text">{getStatusContent()}</span>
          </div>

          <button
            className="btn-hero-secondary"
            onClick={() => setIsHowItWorksOpen(true)}
            aria-label="Open How It Works Guide"
          >
            <span style={{ fontSize: '1.25rem' }}>📖</span>
            <span>HOW IT WORKS</span>
          </button>

          {/* Quick Steering Angle Slider inline for instant reactive motion */}
          <div className="hero-steering-quick-slider">
            <div className="slider-header-row">
              <span className="slider-tag">STEERING ANGLE (θ):</span>
              <strong className="slider-angle-val mono">
                {targetAngle > 0 ? `+${targetAngle.toFixed(1)}°` : `${targetAngle.toFixed(1)}°`}
              </strong>
            </div>
            <input
              type="range"
              min="-60"
              max="60"
              step="1"
              value={targetAngle}
              onChange={(e) => setConfig({ target_angle_deg: parseFloat(e.target.value) })}
              className="tactile-hero-range"
              aria-label="Quick Target Steering Angle Adjustment"
            />
          </div>

          {/* Real-Time Telemetry Summary */}
          {simulationResult && (
            <div className="hero-quick-stat-box">
              <div className="stat-item">
                <span className="stat-label">ARRAY APERTURE</span>
                <span className="stat-val">64 Patches</span>
              </div>
              <div className="stat-divider" />
              <div className="stat-item">
                <span className="stat-label">POINTING RESIDUAL</span>
                <span className="stat-val green">
                  {Math.abs(simulationResult.metrics.pointing_error_deg).toFixed(2)}°
                </span>
              </div>
              <div className="stat-divider" />
              <div className="stat-item">
                <span className="stat-label">BEAM CORRELATION</span>
                <span className="stat-val green">
                  {(simulationResult.metrics.beam_correlation ?? 0).toFixed(4)}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Live Radiating Wavefront Miniature Banner (Reacts to Target Angle) */}
        <div className="hero-wavefront-banner">
          <div className="wavefront-legend">
            <span className="wavefront-dot impaired-dot" />
            <span>Impaired Sidelobe Scatter</span>
            <span className="wavefront-dot calibrated-dot" />
            <span>Coherent Planar Wavefront ({targetAngle > 0 ? `+${targetAngle}°` : `${targetAngle}°`})</span>
          </div>

          <svg
            className="hero-wavefront-svg"
            viewBox="0 0 800 64"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            role="presentation"
          >
            {/* Ground Line */}
            <line x1="20" y1="56" x2="780" y2="56" stroke="#64748b" strokeWidth="2" strokeDasharray="4,4" />
            {/* 64 Tiny Radiating Elements */}
            {Array.from({ length: 64 }, (_, i) => {
              const x = 30 + (i * (740 / 63));
              return (
                <rect
                  key={i}
                  x={x - 2}
                  y={50}
                  width={4}
                  height={6}
                  rx={1}
                  fill={i % 7 === 0 ? '#ef4444' : '#0284c7'}
                  opacity={0.85}
                />
              );
            })}

            {/* Steered Coherent Waves Propagating at target heading */}
            {[1, 2, 3, 4].map((wave) => {
              const yBase = 50 - wave * 10;
              const xShift = (50 - yBase) * Math.tan((targetAngle * Math.PI) / 180);
              return (
                <path
                  key={wave}
                  d={`M ${60 + xShift} ${yBase} Q ${400 + xShift} ${yBase - 5} ${740 + xShift} ${yBase}`}
                  stroke="url(#waveGrad)"
                  strokeWidth={2}
                  strokeDasharray={wave === 2 ? '6,3' : 'none'}
                  opacity={1 - wave * 0.18}
                  className="wavefront-arc"
                />
              );
            })}

            {/* Direction Vector Arrow */}
            <g transform={`translate(${400 + 35 * Math.tan((targetAngle * Math.PI) / 180)}, 14)`}>
              <line x1="0" y1="18" x2={16 * Math.sin((targetAngle * Math.PI) / 180)} y2={-10} stroke="#10b981" strokeWidth="2" />
              <polygon points="0,-12 4,-4 -4,-4" fill="#10b981" transform={`rotate(${targetAngle})`} />
            </g>

            <defs>
              <linearGradient id="waveGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#0284c7" stopOpacity="0.2" />
                <stop offset="50%" stopColor="#10b981" stopOpacity="0.9" />
                <stop offset="100%" stopColor="#0284c7" stopOpacity="0.2" />
              </linearGradient>
            </defs>
          </svg>
        </div>
      </div>
    </section>
  );
};
