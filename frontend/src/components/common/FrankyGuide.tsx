import React from 'react';
import { useAppStore } from '../../store/appStore';

export const FrankyGuide: React.FC = () => {
  const { superMode, frankyMessage, workflowStage, simulateState } = useAppStore();

  if (!superMode) return null;

  // Default context messages if no transient message is set
  const getContextMessage = () => {
    if (frankyMessage) return frankyMessage;
    switch (workflowStage) {
      case 'observe':
        return "Reading coupled antenna currents across all 64 elements...";
      case 'diagnose':
        return "Diagnosing RF impairment: phase skews and amplifier gain variations detected!";
      case 'calibrate':
        return "Computing compensation weights: applying the neural physics core!";
      case 'restore':
        return "Restoring array excitation: realigning phases across the aperture!";
      case 'verify':
      case 'completed':
        return "SUPER! Calibration verified! Check the restored beam pattern!";
      default:
        return simulateState === 'idle'
          ? "Welcome to the RF Workshop! Pick your parameters and click RUN CALIBRATION!"
          : "RF laboratory ready for next experiment.";
    }
  };

  return (
    <div className="franky-guide-banner" role="complementary" aria-label="Franky Virtual RF Engineer Guide">
      <div className="franky-avatar-container">
        {/* Illustrated Vector Mascot — Franky */}
        <svg
          width="48"
          height="48"
          viewBox="0 0 48 48"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className="franky-svg-avatar"
        >
          {/* Hair / Headgear */}
          <path d="M12 18C12 11.37 17.37 6 24 6C30.63 6 36 11.37 36 18V24H12V18Z" fill="#0284c7" />
          <path d="M8 14L16 8L24 14L32 8L40 14V20H8V14Z" fill="#0369a1" />

          {/* Face base */}
          <rect x="14" y="16" width="20" height="18" rx="4" fill="#fde047" stroke="#0f172a" strokeWidth="1.5" />

          {/* Star Emblem on Forehead/Headband */}
          <path
            d="M24 8L25.3 11.4H28.9L26 13.5L27.1 16.9L24 14.8L20.9 16.9L22 13.5L19.1 11.4H22.7L24 8Z"
            fill="#facc15"
            stroke="#b45309"
            strokeWidth="0.8"
          />

          {/* Engineer Visor / Safety Glasses */}
          <rect x="16" y="20" width="16" height="7" rx="2" fill="#0284c7" opacity="0.85" stroke="#0f172a" strokeWidth="1.2" />
          <line x1="17" y1="23" x2="31" y2="23" stroke="#e0f2fe" strokeWidth="1.2" strokeLinecap="round" />

          {/* Smile / Mouth */}
          {simulateState === 'computing' ? (
            <ellipse cx="24" cy="30" rx="3" ry="2" fill="#991b1b" />
          ) : (
            <path d="M20 29C20 31.2 21.8 32.5 24 32.5C26.2 32.5 28 31.2 28 29" stroke="#0f172a" strokeWidth="1.5" strokeLinecap="round" />
          )}

          {/* Torso / Lab Uniform */}
          <path d="M10 42C10 36 16 34 24 34C32 34 38 36 38 42V46H10V42Z" fill="#0284c7" stroke="#0f172a" strokeWidth="1.5" />
          {/* Star Badge on chest */}
          <path
            d="M24 36L24.9 38.2H27.3L25.3 39.6L26.1 41.8L24 40.5L21.9 41.8L22.7 39.6L20.7 38.2H23.1L24 36Z"
            fill="#fde047"
          />
        </svg>
        <span className="franky-tag">FRANKY ★ RF</span>
      </div>

      <div className="franky-speech-bubble">
        <div className="franky-speech-header">
          <span className="franky-name">Franky — Virtual RF Workshop Engineer</span>
          <span className="franky-badge">SUPER MODE ACTIVE</span>
        </div>
        <p className="franky-speech-text">{getContextMessage()}</p>
      </div>
    </div>
  );
};
