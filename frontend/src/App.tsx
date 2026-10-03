import React from 'react';
import { AppShell } from './components/layout/AppShell';
import { OverviewPage } from './pages/OverviewPage';
import { Dashboard } from './pages/Dashboard';
import { CalibrationExplorer } from './pages/CalibrationExplorer';
import { BeamLab } from './pages/BeamLab';
import { RobustnessLab } from './pages/RobustnessLab';
import { ResearchStatus } from './pages/ResearchStatus';
import { SystemStatusPage } from './pages/SystemStatusPage';
import { useAppStore } from './store/appStore';

export const App: React.FC = () => {
  const { activeView } = useAppStore();

  const isOverview = activeView === 'overview';
  const isSimulation = activeView === 'simulation' || activeView === 'dashboard';
  const isCalibration = activeView === 'calibration';
  const isBeam = activeView === 'beam';
  const isRobustness = activeView === 'robustness';
  const isResearch = activeView === 'research';
  const isSystem = activeView === 'system';

  return (
    <AppShell>
      {isOverview && <OverviewPage />}
      {isSimulation && <Dashboard />}
      {isCalibration && <CalibrationExplorer />}
      {isBeam && <BeamLab />}
      {isRobustness && <RobustnessLab />}
      {isResearch && <ResearchStatus />}
      {isSystem && <SystemStatusPage />}
    </AppShell>
  );
};

export default App;
