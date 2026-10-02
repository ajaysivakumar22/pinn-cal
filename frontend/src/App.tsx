import React, { useState, useEffect } from 'react';
import { SideNavigation } from './components/layout/SideNavigation';
import { TopStatusBar } from './components/layout/TopStatusBar';
import { Dashboard } from './pages/Dashboard';
import { CalibrationExplorer } from './pages/CalibrationExplorer';
import { BeamLab } from './pages/BeamLab';
import { RobustnessLab } from './pages/RobustnessLab';
import { ResearchStatus } from './pages/ResearchStatus';
import { ConceptModal } from './components/common/ConceptExplainer';
import { HowItWorksModal } from './components/common/HowItWorksModal';
import { useAppStore } from './store/appStore';

export const App: React.FC = () => {
  const { activeView, isHowItWorksOpen, setIsHowItWorksOpen, loadSystemStatus } = useAppStore();
  const [isMobileNavOpen, setIsMobileNavOpen] = useState(false);

  useEffect(() => {
    loadSystemStatus();
    const interval = setInterval(loadSystemStatus, 15_000);
    return () => clearInterval(interval);
  }, [loadSystemStatus]);

  const isSimulation = activeView === 'simulation' || activeView === 'dashboard';
  const isCalibration = activeView === 'calibration';
  const isBeam = activeView === 'beam';
  const isRobustness = activeView === 'robustness';
  const isResearch = activeView === 'research' || activeView === 'system';

  return (
    <div className="app-layout-shell">
      {/* 1. Left Side Viewing Rail */}
      <SideNavigation
        isOpenOnMobile={isMobileNavOpen}
        onCloseMobile={() => setIsMobileNavOpen(false)}
      />

      {/* 2. Main Application Viewport */}
      <div className="app-main-viewport">
        {/* Compact Top Telemetry & Status Bar */}
        <TopStatusBar onOpenMobileNav={() => setIsMobileNavOpen(true)} />

        {/* Dynamic Page Stage */}
        <main className="app-page-content" role="main">
          {isSimulation && <Dashboard />}
          {isCalibration && <CalibrationExplorer />}
          {isBeam && <BeamLab />}
          {isRobustness && <RobustnessLab />}
          {isResearch && <ResearchStatus />}
        </main>
      </div>

      {/* Global Explainer Modals */}
      <ConceptModal />
      <HowItWorksModal isOpen={isHowItWorksOpen} onClose={() => setIsHowItWorksOpen(false)} />
    </div>
  );
};

export default App;
