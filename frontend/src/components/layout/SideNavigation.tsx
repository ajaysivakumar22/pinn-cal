import React from 'react';
import { useAppStore } from '../../store/appStore';
import type { ActiveView } from '../../types/api';

interface NavItem {
  id: ActiveView;
  index: string;
  label: string;
  descriptor: string;
  icon: string;
}

const NAV_ITEMS: NavItem[] = [
  {
    id: 'simulation',
    index: '01',
    label: 'Simulation',
    descriptor: 'Live 64E antenna & beam experiment',
    icon: '📡',
  },
  {
    id: 'calibration',
    index: '02',
    label: 'Calibration Explorer',
    descriptor: 'Per-element 64-channel diagnosis',
    icon: '🔍',
  },
  {
    id: 'beam',
    index: '03',
    label: 'Beam Lab',
    descriptor: 'Spatial radiation bench & steering',
    icon: '⚡',
  },
  {
    id: 'robustness',
    index: '04',
    label: 'Robustness Lab',
    descriptor: 'Monte Carlo SNR & mismatch sweeps',
    icon: '📊',
  },
  {
    id: 'research',
    index: '05',
    label: 'Research Status',
    descriptor: 'Scientific audit & core validation',
    icon: '🛡️',
  },
];

interface SideNavigationProps {
  isOpenOnMobile?: boolean;
  onCloseMobile?: () => void;
}

export const SideNavigation: React.FC<SideNavigationProps> = ({
  isOpenOnMobile = false,
  onCloseMobile,
}) => {
  const {
    activeView,
    setActiveView,
    superMode,
    toggleSuperMode,
    setIsHowItWorksOpen,
    systemStatus,
    health,
  } = useAppStore();

  const [scrolledSection, setScrolledSection] = React.useState<ActiveView | null>(null);

  // Scroll spy: observe visible section when in simulation / dashboard view
  React.useEffect(() => {
    if (activeView !== 'simulation' && activeView !== 'dashboard') {
      setScrolledSection(null);
      return;
    }

    const sectionIds: { id: string; view: ActiveView }[] = [
      { id: 'section-simulation', view: 'simulation' },
      { id: 'section-calibration', view: 'calibration' },
      { id: 'section-beam', view: 'beam' },
      { id: 'section-robustness', view: 'robustness' },
    ];

    const elements = sectionIds
      .map((s) => ({ el: document.getElementById(s.id), view: s.view }))
      .filter((item): item is { el: HTMLElement; view: ActiveView } => item.el !== null);

    if (elements.length === 0 || typeof window === 'undefined' || !('IntersectionObserver' in window)) return;

    const observer = new window.IntersectionObserver(
      (entries) => {
        // Find the visible section with highest intersection ratio or topmost visible
        const visible = entries.filter((e) => e.isIntersecting);
        if (visible.length > 0) {
          // Sort by intersectionRatio descending
          visible.sort((a, b) => b.intersectionRatio - a.intersectionRatio);
          const matched = sectionIds.find((s) => s.id === visible[0].target.id);
          if (matched) {
            setScrolledSection(matched.view);
          }
        }
      },
      {
        root: null,
        rootMargin: '-15% 0px -40% 0px',
        threshold: [0.1, 0.25, 0.5],
      }
    );

    elements.forEach(({ el }) => observer.observe(el));

    return () => {
      observer.disconnect();
    };
  }, [activeView]);

  const handleSelectView = (view: ActiveView) => {
    setActiveView(view);
    // If selecting a section in simulation view, smoothly scroll into view
    if (view === 'simulation' || view === 'calibration' || view === 'beam' || view === 'robustness') {
      const targetEl = document.getElementById(`section-${view}`);
      if (targetEl) {
        targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
    if (onCloseMobile) {
      onCloseMobile();
    }
  };

  const currentActive = scrolledSection ?? activeView;
  const normalizedActiveView =
    currentActive === 'dashboard' ? 'simulation' : currentActive === 'system' ? 'research' : currentActive;

  const cudaAvailable = systemStatus?.cuda_available ?? false;
  const isBackendOk = health?.status === 'ok';

  return (
    <aside
      className={`app-side-rail ${isOpenOnMobile ? 'mobile-open' : ''}`}
      aria-label="Application Primary Viewing Rail"
    >
      {/* 1. Brand Identity Header */}
      <div className="side-rail-header">
        <div className="side-rail-brand">
          <div className="brand-logo-mark">
            <span className="brand-logo-char">P</span>
            <span className="brand-star-accent">★</span>
          </div>
          <div className="brand-text-block">
            <div className="brand-title-row">
              <span className="brand-name">PINN-Cal</span>
              <span className="brand-tag">RF-LAB</span>
            </div>
            <span className="brand-caption">Physics-Informed Neural Calibration</span>
          </div>
        </div>

        {/* Mobile Close Button */}
        {onCloseMobile && (
          <button
            className="mobile-rail-close-btn"
            onClick={onCloseMobile}
            aria-label="Close navigation"
          >
            ✕
          </button>
        )}
      </div>

      {/* 2. Navigation Items (01 to 05) */}
      <nav className="side-rail-nav" role="navigation" aria-label="Main Sections">
        <div className="rail-section-label">EXPERIMENT MODULES</div>
        <ul className="side-rail-list">
          {NAV_ITEMS.map((item) => {
            const isActive = normalizedActiveView === item.id;
            return (
              <li key={item.id} className="side-rail-item">
                <button
                  className={`side-rail-btn ${isActive ? 'active' : ''}`}
                  onClick={() => handleSelectView(item.id)}
                  aria-current={isActive ? 'page' : undefined}
                >
                  <span className="nav-item-indicator" aria-hidden="true" />
                  <div className="nav-item-index">{item.index}</div>
                  <div className="nav-item-body">
                    <div className="nav-item-title-row">
                      <span className="nav-item-icon">{item.icon}</span>
                      <span className="nav-item-label">{item.label}</span>
                    </div>
                    <span className="nav-item-desc">{item.descriptor}</span>
                  </div>
                </button>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* 3. Lower Control Station: Super Mode + How It Works + Telemetry */}
      <div className="side-rail-footer">
        {/* Super Mode Toggle Station */}
        <div className={`rail-super-mode-card ${superMode ? 'active' : ''}`}>
          <div className="super-card-top">
            <div className="super-card-title">
              <span className="super-star">★</span>
              <span>SUPER MODE</span>
            </div>
            <button
              className={`super-toggle-pill ${superMode ? 'on' : 'off'}`}
              onClick={toggleSuperMode}
              aria-label="Toggle Super Presentation Mode"
            >
              {superMode ? 'ENABLED' : 'OFF'}
            </button>
          </div>
          <p className="super-card-desc">
            {superMode
              ? 'Franky mascot & dynamic RF motion active'
              : 'Switch on virtual RF workshop engineer & motion'}
          </p>
        </div>

        {/* How It Works Button */}
        <button
          className="side-rail-action-btn"
          onClick={() => {
            setIsHowItWorksOpen(true);
            if (onCloseMobile) onCloseMobile();
          }}
          aria-label="Open 8-step lifecycle guide"
        >
          <span className="action-btn-icon">📖</span>
          <span className="action-btn-text">How It Works Guide</span>
          <span className="action-btn-arrow">→</span>
        </button>

        {/* Hardware Status Strip */}
        <div className="side-rail-hardware-status">
          <div className="hw-status-item">
            <span className={`hw-dot ${isBackendOk ? 'green' : 'red'}`} />
            <span>BACKEND {isBackendOk ? 'ONLINE' : 'OFFLINE'}</span>
          </div>
          <div className="hw-status-item">
            <span className={`hw-dot ${cudaAvailable ? 'green' : 'amber'}`} />
            <span>{cudaAvailable ? 'CUDA GTX 1650' : 'CPU FALLBACK'}</span>
          </div>
        </div>
      </div>
    </aside>
  );
};
