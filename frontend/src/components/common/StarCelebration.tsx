import React, { useEffect } from 'react';
import { useAppStore } from '../../store/appStore';

export const StarCelebration: React.FC = () => {
  const { showStarCelebration, setShowStarCelebration, simulationResult } = useAppStore();

  useEffect(() => {
    if (showStarCelebration) {
      const timer = setTimeout(() => {
        setShowStarCelebration(false);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [showStarCelebration, setShowStarCelebration]);

  if (!showStarCelebration || !simulationResult) return null;

  const peak = simulationResult.metrics.beam_peak_angle_deg;
  const target = simulationResult.request.target_angle_deg;
  const correlation = simulationResult.metrics.beam_correlation ?? 1;

  return (
    <div className="star-celebration-toast" role="status" aria-live="polite">
      <div className="star-celebration-stars">
        <span className="star-burst">★</span>
        <span className="star-burst small">★</span>
        <span className="star-burst gold">★</span>
      </div>
      <div className="star-celebration-body">
        <div className="star-celebration-title">★ SUPER!! CALIBRATION VERIFIED!</div>
        <div className="star-celebration-subtitle">
          Aperture restored · Target {target > 0 ? `+${target}°` : `${target}°`} · Peak {peak > 0 ? `+${peak.toFixed(1)}°` : `${peak.toFixed(1)}°`} · Corr {(correlation * 100).toFixed(1)}%
        </div>
      </div>
      <button
        className="star-celebration-close"
        onClick={() => setShowStarCelebration(false)}
        aria-label="Dismiss completion celebration"
      >
        ✕
      </button>
    </div>
  );
};
