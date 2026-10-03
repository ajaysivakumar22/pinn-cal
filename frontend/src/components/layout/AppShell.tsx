import React, { useState, useEffect } from 'react';
import { Sidebar } from './Sidebar';
import { TopStatusBar } from './TopStatusBar';
import { ConceptModal } from '../common/ConceptExplainer';
import { HowItWorksModal } from '../common/HowItWorksModal';
import { useAppStore } from '../../store/appStore';

export interface AppShellProps {
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({ children }) => {
  const {
    activeView,
    isHowItWorksOpen,
    setIsHowItWorksOpen,
    loadSystemStatus,
    simulateState,
    simulateError,
    theme,
    superMode,
  } = useAppStore();

  const [isMobileNavOpen, setIsMobileNavOpen] = useState(false);

  // Poll system and model status every 15s
  useEffect(() => {
    loadSystemStatus();
    const interval = setInterval(loadSystemStatus, 15_000);
    return () => clearInterval(interval);
  }, [loadSystemStatus]);

  // Ensure document data-theme matches store
  useEffect(() => {
    if (typeof document !== 'undefined') {
      document.documentElement.setAttribute('data-theme', theme);
    }
  }, [theme]);

  // Handle Escape key to close mobile drawer
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isMobileNavOpen) {
        setIsMobileNavOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isMobileNavOpen]);

  return (
    <div className={`app-layout-shell ${superMode ? 'super-mode-active' : ''}`}>
      {/* Mobile Drawer Backdrop */}
      {isMobileNavOpen && (
        <div
          className="sidebar-backdrop-overlay"
          onClick={() => setIsMobileNavOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* 1. Left Navigation Rail */}
      <Sidebar
        isOpenOnMobile={isMobileNavOpen}
        onCloseMobile={() => setIsMobileNavOpen(false)}
      />

      {/* 2. Main Viewport Stage */}
      <div className="app-main-viewport">
        {/* Top Header */}
        <TopStatusBar onOpenMobileNav={() => setIsMobileNavOpen(true)} />

        {/* Global Simulation Error Alert (when outside Dashboard) */}
        {activeView !== 'simulation' && activeView !== 'dashboard' && simulateState === 'error' && simulateError && (
          <div className="simulation-error-banner" role="alert">
            <span className="error-banner-icon" aria-hidden="true">⚠️</span>
            <div className="error-banner-text">
              <strong>Simulation Computation Error:</strong> {simulateError}
            </div>
          </div>
        )}

        {/* Main Content Viewport */}
        <main className="app-page-content" role="main">
          {children}
        </main>
      </div>

      {/* Global Modals */}
      <ConceptModal />
      <HowItWorksModal
        isOpen={isHowItWorksOpen}
        onClose={() => setIsHowItWorksOpen(false)}
      />
    </div>
  );
};

export default AppShell;
