import React from 'react';
import { useAppStore, type WorkflowStage } from '../../store/appStore';

interface StageMeta {
  id: WorkflowStage;
  num: string;
  title: string;
  tagline: string;
  activeDescription: string;
}

const STAGES: StageMeta[] = [
  {
    id: 'observe',
    num: '01',
    title: 'OBSERVE',
    tagline: 'Measure RF currents',
    activeDescription: 'Sampling coupled antenna currents under hidden mutual impedance & noise...',
  },
  {
    id: 'diagnose',
    num: '02',
    title: 'DIAGNOSE',
    tagline: 'Estimate RF errors',
    activeDescription: 'Evaluating PINN neural inference to estimate per-element phase & gain errors...',
  },
  {
    id: 'calibrate',
    num: '03',
    title: 'CALIBRATE',
    tagline: 'Compute compensation',
    activeDescription: 'Forming complex inverse compensation weights C = H^-1...',
  },
  {
    id: 'restore',
    num: '04',
    title: 'RESTORE',
    tagline: 'Correct excitation',
    activeDescription: 'Synthesizing corrected array excitation V_corr = C ⊙ V_hw...',
  },
  {
    id: 'verify',
    num: '05',
    title: 'VERIFY',
    tagline: 'Validate beam pattern',
    activeDescription: 'Computing far-field radiation pattern, correlation, HPBW, and PSLL...',
  },
];

export const WorkflowPipeline: React.FC = () => {
  const { workflowStage, simulateState, simulationResult } = useAppStore();

  const isSimulating = simulateState === 'computing';
  const isDone = simulateState === 'success' && simulationResult !== null;

  return (
    <div className="workflow-pipeline-card card" style={{ marginBottom: 16 }}>
      <div
        className="card-header"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '8px 16px',
          background: '#fafaf9',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span
            style={{
              fontSize: '0.6875rem',
              fontWeight: 700,
              letterSpacing: '0.08em',
              color: 'var(--color-primary)',
              textTransform: 'uppercase',
            }}
          >
            Physical Calibration Pipeline
          </span>
          <span style={{ fontSize: '0.6875rem', color: 'var(--color-text-tertiary)' }}>
            · 5-Stage RF Restoration Flow
          </span>
        </div>
        <div>
          {isSimulating && (
            <span
              className="badge"
              style={{
                background: '#e0f2fe',
                color: '#0369a1',
                fontWeight: 600,
                fontSize: '0.6875rem',
              }}
            >
              Computing physics in real time...
            </span>
          )}
          {isDone && !isSimulating && (
            <span
              className="badge"
              style={{
                background: '#dcfce7',
                color: '#15803d',
                fontWeight: 600,
                fontSize: '0.6875rem',
              }}
            >
              Pipeline Complete · Array Calibrated
            </span>
          )}
        </div>
      </div>

      <div
        className="workflow-stages-grid"
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(5, 1fr)',
          gap: 1,
          background: 'var(--color-border)',
        }}
      >
        {STAGES.map((s, idx) => {
          let stageStatus: 'pending' | 'active' | 'completed' = 'pending';

          if (isDone && !isSimulating) {
            stageStatus = 'completed';
          } else if (isSimulating) {
            const currentIdx = STAGES.findIndex((st) => st.id === workflowStage);
            if (idx < currentIdx) stageStatus = 'completed';
            else if (idx === currentIdx) stageStatus = 'active';
            else stageStatus = 'pending';
          }

          const isActive = stageStatus === 'active';
          const isCompleted = stageStatus === 'completed';

          return (
            <div
              key={s.id}
              className={`workflow-stage-cell ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}
              style={{
                background: isActive ? '#eff6ff' : isCompleted ? '#ffffff' : '#f8fafc',
                padding: '12px 14px',
                display: 'flex',
                flexDirection: 'column',
                gap: 4,
                position: 'relative',
                transition: 'all 0.2s ease',
                borderBottom: isActive ? '3px solid var(--color-primary)' : isCompleted ? '3px solid #16a34a' : '3px solid transparent',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.625rem',
                    fontWeight: 700,
                    color: isActive ? 'var(--color-primary)' : isCompleted ? '#16a34a' : 'var(--color-text-tertiary)',
                  }}
                >
                  {s.num}
                </span>
                {isCompleted && (
                  <span style={{ fontSize: '0.6875rem', color: '#16a34a', fontWeight: 700 }}>✓</span>
                )}
                {isActive && (
                  <span
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: '50%',
                      background: 'var(--color-primary)',
                      animation: 'pulse 1s infinite',
                    }}
                  />
                )}
              </div>

              <div
                style={{
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  letterSpacing: '0.04em',
                  color: isActive ? 'var(--color-primary)' : 'var(--color-text-primary)',
                }}
              >
                {s.title}
              </div>

              <div
                style={{
                  fontSize: '0.6875rem',
                  color: 'var(--color-text-secondary)',
                  lineHeight: 1.3,
                }}
              >
                {isActive ? s.activeDescription : s.tagline}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
