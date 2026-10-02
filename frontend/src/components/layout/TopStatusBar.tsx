import React from 'react';
import { useAppStore } from '../../store/appStore';

interface TopStatusBarProps {
  onOpenMobileNav: () => void;
}

export const TopStatusBar: React.FC<TopStatusBarProps> = ({ onOpenMobileNav }) => {
  const { config, simulationResult, systemStatus, health, modelStatus, superMode, frankyMessage } = useAppStore();

  const isBackendOk = health?.status === 'ok';
  const cudaOk = systemStatus?.cuda_available ?? false;
  const targetAngle = config.target_angle_deg;

  return (
    <header className="top-status-bar" role="banner" aria-label="System telemetry status bar">
      {/* Mobile Menu Trigger & Mini Brand */}
      <div className="top-status-left">
        <button
          className="mobile-nav-trigger-btn"
          onClick={onOpenMobileNav}
          aria-label="Open navigation menu"
        >
          <span className="hamburger-line" />
          <span className="hamburger-line" />
          <span className="hamburger-line" />
        </button>

        <div className="mobile-brand-title">
          <span className="brand-name">PINN-Cal</span>
          <span className="brand-star">★</span>
        </div>

        {/* Dynamic Aperture Geometry Badge */}
        <div className="status-pill aperture-spec-pill" title={isBackendOk ? 'Backend Online' : 'Backend Offline'}>
          <span className="pill-dot" style={{ background: isBackendOk ? '#0284c7' : '#ef4444' }} />
          <span className="spec-item"><strong>{config.num_elements}</strong> PATCHES</span>
          <span className="spec-sep">·</span>
          <span className="spec-item"><strong>{(config.frequency_hz / 1e9).toFixed(0)} GHz</strong> mmWAVE</span>
          <span className="spec-sep">·</span>
          <span className="spec-item">0.50λ</span>
        </div>

        {/* Target Heading Badge */}
        <div className="status-pill target-heading-pill">
          <span className="target-icon">🎯</span>
          <span>TARGET: <strong>{targetAngle > 0 ? `+${targetAngle.toFixed(1)}°` : `${targetAngle.toFixed(1)}°`}</strong></span>
        </div>
      </div>

      {/* Right Telemetry: CUDA / Model / Active State */}
      <div className="top-status-right">
        {/* Franky Compact Bubble in Super Mode */}
        {superMode && frankyMessage && (
          <div className="franky-mini-telemetry">
            <span className="franky-mini-avatar">★</span>
            <span className="franky-mini-text">{frankyMessage}</span>
          </div>
        )}

        <div className="status-pill model-badge-pill">
          <span className="model-icon">🧠</span>
          <span>PINN <strong>{modelStatus?.best_epoch ? `EP ${modelStatus.best_epoch}` : '83.9k'}</strong></span>
        </div>

        <div className={`status-pill hardware-pill ${cudaOk ? 'cuda' : 'cpu'}`}>
          <span className={`status-light ${cudaOk ? 'green' : 'amber'}`} />
          <span>{cudaOk ? 'CUDA ACCEL' : 'CPU MODE'}</span>
        </div>

        {simulationResult && (
          <div className="status-pill pointing-residual-pill">
            <span className="metric-tag">POINTING Δθ:</span>
            <strong className="metric-number">
              {Math.abs(simulationResult.metrics.pointing_error_deg).toFixed(2)}°
            </strong>
          </div>
        )}
      </div>
    </header>
  );
};
