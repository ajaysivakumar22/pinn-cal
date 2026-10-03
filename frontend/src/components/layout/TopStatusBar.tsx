import React from 'react';
import { useAppStore } from '../../store/appStore';
import { StatusChip } from '../common/StatusChip';
import { ThemeToggle } from '../common/ThemeToggle';

export interface TopStatusBarProps {
  onOpenMobileNav: () => void;
  className?: string;
}

export const TopStatusBar: React.FC<TopStatusBarProps> = ({
  onOpenMobileNav,
  className = '',
}) => {
  const {
    config,
    simulationResult,
    systemStatus,
    health,
    modelStatus,
  } = useAppStore();

  const isBackendOk = health?.status === 'ok';
  const cudaOk = systemStatus?.cuda_available ?? false;
  const gpuName = systemStatus?.gpu_name || 'NVIDIA GTX 1650';
  const targetAngle = config.target_angle_deg;

  // Parameter and version strings from live metadata
  const paramCount = modelStatus?.trainable_parameters
    ? modelStatus.trainable_parameters.toLocaleString()
    : '33,088';
  const versionString = modelStatus?.version
    ? `V${modelStatus.version}`
    : 'V4.0.0';

  return (
    <header
      className={`app-top-header ${className}`}
      role="banner"
      aria-label="Application Telemetry and Status Header"
    >
      {/* Left Section: Mobile Menu Trigger + Brand + Array Geometry */}
      <div className="top-header-left">
        <button
          type="button"
          className="mobile-nav-trigger-btn"
          onClick={onOpenMobileNav}
          aria-label="Open primary navigation rail"
        >
          <span className="hamburger-line" />
          <span className="hamburger-line" />
          <span className="hamburger-line" />
        </button>

        <div className="top-header-brand-mobile">
          <span className="brand-name">PINN-Cal</span>
          <span className="brand-v4-tag">V4</span>
        </div>

        {/* Array Geometry Specification Pill */}
        <div
          className="aperture-spec-pill"
          title={`Array geometry: ${config.num_elements}-element Uniform Linear Array (ULA), ${(config.frequency_hz / 1e9).toFixed(0)} GHz carrier frequency, ${config.element_spacing_lambda}λ spacing`}
        >
          <span className="spec-dot" aria-hidden="true" />
          <span className="spec-item"><strong>{config.num_elements}</strong> ELEMENTS</span>
          <span className="spec-sep" aria-hidden="true">·</span>
          <span className="spec-item"><strong>{(config.frequency_hz / 1e9).toFixed(0)} GHz</strong> mmWAVE</span>
          <span className="spec-sep" aria-hidden="true">·</span>
          <span className="spec-item">{config.element_spacing_lambda.toFixed(2)}λ</span>
        </div>

        {/* Target Heading Pill */}
        <div className="target-heading-pill" title={`Target steering angle: ${targetAngle}°`}>
          <span className="target-icon" aria-hidden="true">🎯</span>
          <span className="target-label">TARGET:</span>
          <strong className="target-val">
            {targetAngle > 0 ? `+${targetAngle.toFixed(1)}°` : `${targetAngle.toFixed(1)}°`}
          </strong>
        </div>
      </div>

      {/* Right Section: Live Telemetry Status Chips + Theme Toggle */}
      <div className="top-header-right">
        {/* Backend Status Chip */}
        <StatusChip
          label="BACKEND"
          value={isBackendOk ? 'ONLINE' : 'OFFLINE'}
          status={isBackendOk ? 'online' : 'offline'}
          title={`Backend status: ${isBackendOk ? 'Healthy' : 'Disconnected'}`}
        />

        {/* Model Architecture & Parameter Chip (from live metadata) */}
        <StatusChip
          icon={<span className="chip-icon-symbol">🧠</span>}
          label="PINN-Cal V4"
          value={`${paramCount} PARAMETERS`}
          status="info"
          title={
            modelStatus
              ? `${modelStatus.model_name} · ${paramCount} trainable parameters · δ=${modelStatus.tikhonov_delta}, α=${modelStatus.residual_alpha} · Checkpoint: ${modelStatus.checkpoint}`
              : 'PINN-Cal V4 · 33,088 parameters · δ=0.15, α=0.50 · Frozen reference'
          }
        />

        {/* Checkpoint Version Chip */}
        <StatusChip
          label="VERSION"
          value={`${versionString} FROZEN`}
          status="primary"
          title="Frozen V4 Reference Model Checkpoint (seed 42)"
        />

        {/* CUDA Hardware Acceleration Chip */}
        <StatusChip
          label="CUDA ACCEL"
          value={cudaOk ? 'ONLINE' : 'CPU MODE'}
          status={cudaOk ? 'online' : 'warning'}
          title={cudaOk ? `CUDA Hardware Acceleration Active: ${gpuName}` : 'CPU fallback execution'}
        />

        {/* Live Simulation Pointing Error (when result available) */}
        {simulationResult && (
          <div className="pointing-error-telemetry-chip">
            <span className="telemetry-label">POINTING Δθ:</span>
            <strong className="telemetry-number">
              {Math.abs(simulationResult.metrics.pointing_error_deg).toFixed(2)}°
            </strong>
          </div>
        )}

        {/* Theme Toggle (Light / Dark) */}
        <div className="top-header-theme-wrapper">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
};

export default TopStatusBar;
