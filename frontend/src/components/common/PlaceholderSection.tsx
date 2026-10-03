import React from 'react';
import { Panel } from './Panel';

export interface PlannedModule {
  id: string;
  title: string;
  description: string;
  icon: string;
}

export interface PlaceholderSectionProps {
  sectionId: string;
  title: string;
  subtitle: string;
  plannedStage: string;
  summary: string;
  modules: PlannedModule[];
  liveLabComponent?: React.ReactNode;
  defaultShowLiveLab?: boolean;
}

export const PlaceholderSection: React.FC<PlaceholderSectionProps> = ({
  sectionId,
  title,
  subtitle,
  plannedStage,
  summary,
  modules,
  liveLabComponent,
  defaultShowLiveLab = false,
}) => {
  const [showLiveLab, setShowLiveLab] = React.useState(defaultShowLiveLab);

  return (
    <div className="placeholder-section-wrapper" id={`section-wrapper-${sectionId}`}>
      {/* Stage Banner */}
      <div className="placeholder-stage-banner">
        <div className="stage-banner-left">
          <span className="stage-pill">{plannedStage}</span>
          <span className="stage-title">{title}</span>
          <span className="stage-sep">·</span>
          <span className="stage-desc">{subtitle}</span>
        </div>
        {liveLabComponent && (
          <div className="stage-banner-actions">
            <button
              type="button"
              className={`stage-view-toggle-btn ${showLiveLab ? 'active' : ''}`}
              onClick={() => setShowLiveLab(!showLiveLab)}
              aria-label={showLiveLab ? 'Switch to planned architecture preview' : 'Switch to live interactive lab'}
            >
              <span className="btn-icon">{showLiveLab ? '📐' : '⚡'}</span>
              <span>{showLiveLab ? 'View Architecture Plan' : 'View Live Interactive Lab'}</span>
            </button>
          </div>
        )}
      </div>

      {showLiveLab && liveLabComponent ? (
        <div className="placeholder-live-lab-slot">
          {liveLabComponent}
        </div>
      ) : (
        <div className="placeholder-plan-container">
          {/* Overview Panel */}
          <Panel
            title="Module Architecture & Scope"
            badge={<span className="panel-badge-planning">Stage Target</span>}
            subtitle={summary}
            elevated
          >
            <div className="planned-modules-grid">
              {modules.map((mod, idx) => (
                <div key={mod.id} className="planned-module-card">
                  <div className="module-card-header">
                    <span className="module-idx">{(idx + 1).toString().padStart(2, '0')}</span>
                    <span className="module-icon" aria-hidden="true">{mod.icon}</span>
                    <h4 className="module-title">{mod.title}</h4>
                  </div>
                  <p className="module-desc">{mod.description}</p>
                  <div className="module-status-chip">Scheduled for Implementation</div>
                </div>
              ))}
            </div>
          </Panel>

          {/* Scientific Credibility Notice */}
          <div className="placeholder-scientific-notice">
            <span className="notice-icon">🛡️</span>
            <div className="notice-text">
              <strong>PINN-Cal V4 Frozen Scientific Reference:</strong> All calculations, calibrations, and beam-forming metrics
              consume the verified backend model (33,088 parameters, δ=0.15, α=0.50). This layout shell establishes the design tokens
              and navigation frame for subsequent scientific integration stages.
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
