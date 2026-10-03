import React from 'react';
import { useAppStore } from '../../store/appStore';
import type { ActiveView } from '../../types/api';
import frankyImg from '../../assets/franky.jpg';

export interface NavItemConfig {
  id: ActiveView;
  index: string;
  label: string;
  secondaryText: string;
  icon: string;
}

const SHELL_NAV_ITEMS: NavItemConfig[] = [
  {
    id: 'overview',
    index: '01',
    label: 'OVERVIEW',
    secondaryText: 'Massive-MIMO & V4 Architecture',
    icon: '🌐',
  },
  {
    id: 'simulation',
    index: '02',
    label: 'LIVE ARRAY',
    secondaryText: '64-Element ULA & Controls',
    icon: '📡',
  },
  {
    id: 'calibration',
    index: '03',
    label: 'CALIBRATE',
    secondaryText: 'Per-Element Diagnosis & Weights',
    icon: '🔍',
  },
  {
    id: 'beam',
    index: '04',
    label: 'BEAM RESTORATION',
    secondaryText: 'Spatial Radiation Bench & Steering',
    icon: '⚡',
  },
  {
    id: 'robustness',
    index: '05',
    label: 'ROBUSTNESS LAB',
    secondaryText: 'Monte Carlo SNR & Mismatch Sweeps',
    icon: '📊',
  },
  {
    id: 'research',
    index: '06',
    label: 'RESEARCH & DATA',
    secondaryText: 'Holdout Benchmark & Scientific Audit',
    icon: '🛡️',
  },
  {
    id: 'system',
    index: '07',
    label: 'SYSTEM',
    secondaryText: 'Hardware Runtime & Model Telemetry',
    icon: '⚙️',
  },
];

export interface SidebarProps {
  isOpenOnMobile?: boolean;
  onCloseMobile?: () => void;
  className?: string;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isOpenOnMobile = false,
  onCloseMobile,
  className = '',
}) => {
  const {
    activeView,
    setActiveView,
    systemStatus,
    health,
    superMode,
    toggleSuperMode,
  } = useAppStore();

  const handleSelectView = (view: ActiveView) => {
    setActiveView(view);
    if (onCloseMobile) {
      onCloseMobile();
    }
  };

  const isBackendOk = health?.status === 'ok';
  const cudaAvailable = systemStatus?.cuda_available ?? false;
  const gpuName = systemStatus?.gpu_name || 'NVIDIA GTX 1650';

  // Normalize view mapping for active state
  const isItemActive = (itemId: ActiveView) => {
    if (activeView === itemId) return true;
    if (itemId === 'overview' && activeView === 'dashboard') return true;
    return false;
  };

  return (
    <aside
      className={`app-sidebar-rail ${isOpenOnMobile ? 'mobile-open' : ''} ${className}`}
      aria-label="Application Primary Navigation Rail"
    >
      {/* 1. Brand Identity Area (Section 9) */}
      <div className="sidebar-brand-area">
        <div className="sidebar-brand-header">
          <div className="brand-logo-mark" aria-hidden="true">
            <span className="brand-logo-char">P</span>
            <span className="brand-star-accent">★</span>
          </div>
          <div className="brand-title-group">
            <div className="brand-main-title">
              <span className="brand-name">PINN-Cal</span>
              <span className="brand-v4-badge">V4</span>
            </div>
            <div className="brand-subtitles">
              <span className="brand-subtitle-line">Physics-Informed Neural Calibration</span>
              <span className="brand-subtitle-line secondary">for 5G Massive-MIMO Arrays</span>
            </div>
          </div>
        </div>

        {/* Mobile Close Button */}
        {onCloseMobile && (
          <button
            type="button"
            className="sidebar-mobile-close-btn"
            onClick={onCloseMobile}
            aria-label="Close navigation rail"
          >
            ✕
          </button>
        )}
      </div>

      {/* 2. Navigation Items (Section 8) */}
      <nav className="sidebar-nav-container" role="navigation" aria-label="Main Navigation">
        <div className="sidebar-nav-label">MODULES</div>
        <ul className="sidebar-nav-list">
          {SHELL_NAV_ITEMS.map((item) => {
            const active = isItemActive(item.id);
            return (
              <li key={item.id} className="sidebar-nav-item">
                <button
                  type="button"
                  className={`sidebar-nav-btn ${active ? 'active' : ''}`}
                  onClick={() => handleSelectView(item.id)}
                  aria-current={active ? 'page' : undefined}
                >
                  <span className="nav-active-indicator" aria-hidden="true" />
                  <span className="nav-item-index" aria-hidden="true">{item.index}</span>
                  <span className="nav-item-icon" aria-hidden="true">{item.icon}</span>
                  <div className="nav-item-text">
                    <span className="nav-item-primary">{item.label}</span>
                    <span className="nav-item-secondary">{item.secondaryText}</span>
                  </div>
                </button>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* 3. Lower Station: Franky Reserved Area (Section 17) + Hardware Status Strip (Section 8) */}
      <div className="sidebar-footer-station">
        {/* Franky RF Engineer Mascot Card (Step 75I) */}
        <div
          className={`sidebar-franky-card ${superMode ? 'super-on' : 'super-off'}`}
          id="franky-slot-reservation"
          role="region"
          aria-label="Franky RF Engineer Mascot"
        >
          <div className="franky-card-header">
            <div className="franky-header-left">
              <span className="franky-star-icon" aria-hidden="true">★</span>
              <span className="franky-engineer-title">FRANKY RF ENGINEER</span>
            </div>
            <span className={`franky-status-pill ${superMode ? 'active' : ''}`}>
              {superMode ? 'SUPER ON' : 'STANDBY'}
            </span>
          </div>

          <div className="franky-card-body">
            <div className="franky-avatar-container" title="Franky from One Piece">
              <img
                src={frankyImg}
                alt="Franky from One Piece"
                className={`franky-portrait-img ${superMode ? 'super-glow' : ''}`}
                width="48"
                height="48"
                loading="lazy"
              />
            </div>

            <div className="franky-quote-box">
              <span className="franky-quote-text">
                “CALIBRATE IT.<br />
                RESTORE IT.<br />
                MAKE IT SUPER.”
              </span>
            </div>
          </div>

          <div className="franky-card-action">
            <span className="franky-action-label">SUPER MODE</span>
            <button
              type="button"
              className={`franky-action-toggle ${superMode ? 'active' : ''}`}
              onClick={toggleSuperMode}
              aria-pressed={superMode}
              aria-label="Toggle Super Presentation Mode"
            >
              <span className="toggle-dot" aria-hidden="true" />
              <span>{superMode ? 'ON' : 'OFF'}</span>
            </button>
          </div>
        </div>

        {/* Compact System Status Area (Section 8) */}
        <div className="sidebar-status-strip" aria-label="Runtime telemetry status">
          <div className="status-strip-header">SYSTEM RUNTIME</div>
          <div className="status-strip-items">
            <div className="status-strip-row" title={`Backend status: ${isBackendOk ? 'Online' : 'Offline'}`}>
              <span className={`status-strip-dot ${isBackendOk ? 'online' : 'offline'}`} aria-hidden="true" />
              <span className="status-strip-label">BACKEND</span>
              <span className="status-strip-value">{isBackendOk ? 'ONLINE' : 'OFFLINE'}</span>
            </div>
            <div className="status-strip-row" title={`Acceleration: ${cudaAvailable ? 'CUDA Enabled' : 'CPU Mode'}`}>
              <span className={`status-strip-dot ${cudaAvailable ? 'online' : 'warning'}`} aria-hidden="true" />
              <span className="status-strip-label">CUDA</span>
              <span className="status-strip-value">{cudaAvailable ? 'ONLINE' : 'CPU MODE'}</span>
            </div>
            <div className="status-strip-row" title={`Compute Device: ${gpuName}`}>
              <span className="status-strip-icon" aria-hidden="true">⚡</span>
              <span className="status-strip-label">DEVICE</span>
              <span className="status-strip-value highlight">{gpuName.replace('NVIDIA GeForce ', '')}</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
};
