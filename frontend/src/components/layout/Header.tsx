import React, { useEffect } from 'react';
import { useAppStore } from '../../store/appStore';
import type { ActiveView } from '../../types/api';
import { HowItWorksModal } from '../common/HowItWorksModal';

const VIEWS: { id: ActiveView; label: string }[] = [
  { id: 'simulation', label: 'Simulation' },
  { id: 'calibration', label: 'Calibration Explorer' },
  { id: 'beam', label: 'Beam Lab' },
  { id: 'robustness', label: 'Robustness Lab' },
  { id: 'research', label: 'Research Status' },
];

export const Header: React.FC = () => {
  const {
    health,
    systemStatus,
    modelStatus,
    activeView,
    setActiveView,
    loadSystemStatus,
    config,
    superMode,
    toggleSuperMode,
    isHowItWorksOpen,
    setIsHowItWorksOpen,
  } = useAppStore();

  useEffect(() => {
    loadSystemStatus();
    const interval = setInterval(loadSystemStatus, 30_000);
    return () => clearInterval(interval);
  }, []);

  const backendOnline = health?.status === 'ok';
  const cudaOk = systemStatus?.cuda_available ?? false;
  const gpuName = systemStatus?.gpu_name || (cudaOk ? 'CUDA Device' : 'None (CPU)');
  const modelLoaded = health?.pinn_model_loaded ?? false;

  // Map backward compatible aliases if any
  const normalizedActiveView =
    activeView === 'dashboard' ? 'simulation' : activeView === 'system' ? 'research' : activeView;

  return (
    <>
      <header className="app-header">
        {/* Brand & Mission Statement */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
            <h1 style={{ fontSize: '1.5rem', margin: 0, fontWeight: 800, letterSpacing: '-0.02em', color: '#0f172a', display: 'flex', alignItems: 'center', gap: 4 }}>
              <span>PINN-Cal</span>
              <span style={{ color: '#eab308', fontSize: '1.25rem' }}>★</span>
            </h1>
            <span style={{ fontSize: '0.9375rem', color: 'var(--color-primary)', fontWeight: 600 }}>
              Physics-Informed Neural Calibration
            </span>
          </div>
          <span style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)', fontWeight: 500 }}>
            Estimate RF impairment → compute compensation → restore the beam
          </span>
        </div>

        {/* Real-time Hardware & Model Telemetry */}
        <div className="header-telemetry" style={{ display: 'flex', gap: 14, alignItems: 'center', marginLeft: 20 }}>
          <StatusChip label="Backend" value={backendOnline ? 'ONLINE' : 'OFFLINE'} ok={backendOnline} />
          <StatusChip label="CUDA" value={cudaOk ? 'AVAILABLE' : 'CPU ONLY'} ok={cudaOk} />
          <StatusChip label="GPU" value={gpuName} ok={cudaOk} />
          <StatusChip
            label="PINN"
            value={modelLoaded ? `LOADED (ep ${modelStatus?.best_epoch ?? 40})` : 'UNAVAILABLE'}
            ok={modelLoaded}
          />
          <div style={{ height: 24, width: 1, background: '#cbd5e1' }} />
          <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.15 }}>
            <span style={{ fontSize: '0.6875rem', color: 'var(--color-text-tertiary)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Array State
            </span>
            <span style={{ fontSize: '0.8125rem', color: '#0f172a', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
              {config.num_elements}E · {(config.frequency_hz / 1e9).toFixed(0)} GHz · TARGET {config.target_angle_deg > 0 ? `+${config.target_angle_deg}°` : `${config.target_angle_deg}°`}
            </span>
          </div>
        </div>

        {/* Navigation Tabs + How It Works + Super Mode Toggle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginLeft: 'auto' }}>
          <nav className="nav-tabs" role="navigation" aria-label="Main navigation">
            {VIEWS.map((v) => (
              <button
                key={v.id}
                className={`nav-tab ${normalizedActiveView === v.id ? 'active' : ''}`}
                onClick={() => setActiveView(v.id)}
                aria-current={normalizedActiveView === v.id ? 'page' : undefined}
                style={{ fontSize: '0.875rem', padding: '8px 14px', fontWeight: 600 }}
              >
                {v.label}
              </button>
            ))}
          </nav>

          <button
            className="btn btn-secondary"
            onClick={() => setIsHowItWorksOpen(true)}
            style={{ fontSize: '0.8125rem', padding: '6px 12px', display: 'flex', alignItems: 'center', gap: 6, fontWeight: 600 }}
            title="Overview of the 8-step RF impairment to beam restoration lifecycle"
          >
            <span>ℹ</span>
            <span>How it Works</span>
          </button>

          {/* Super Mode Toggle */}
          <button
            className={`btn super-mode-toggle ${superMode ? 'active' : ''}`}
            onClick={toggleSuperMode}
            title={superMode ? 'Disable Super Mode' : 'Enable Super Mode: Virtual Engineer Franky & Star Celebration'}
            style={{
              fontSize: '0.8125rem',
              padding: '6px 12px',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              fontWeight: 700,
              borderRadius: 6,
              border: superMode ? '1px solid #facc15' : '1px solid #cbd5e1',
              background: superMode ? 'linear-gradient(180deg, #fef08a 0%, #fde047 100%)' : '#ffffff',
              color: superMode ? '#854d0e' : 'var(--color-text-secondary)',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              boxShadow: superMode ? '0 0 10px rgba(250, 204, 21, 0.4)' : 'none',
            }}
          >
            <span style={{ color: superMode ? '#ca8a04' : '#94a3b8', fontSize: '1rem' }}>★</span>
            <span>SUPER MODE</span>
          </button>
        </div>
      </header>

      {/* Explainer Modal */}
      <HowItWorksModal isOpen={isHowItWorksOpen} onClose={() => setIsHowItWorksOpen(false)} />
    </>
  );
};

const StatusChip: React.FC<{ label: string; value: string; ok: boolean }> = ({
  label,
  value,
  ok,
}) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
    <div className={`status-dot ${ok ? 'online' : 'offline'}`} aria-hidden="true" />
    <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.1 }}>
      <span style={{ fontSize: '0.5625rem', color: 'var(--color-text-tertiary)', fontWeight: 600, textTransform: 'uppercase' }}>
        {label}
      </span>
      <span style={{ fontSize: '0.6875rem', color: 'var(--color-text-secondary)', fontWeight: 600 }}>
        {value}
      </span>
    </div>
  </div>
);
