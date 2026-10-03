import React, { useState } from 'react';
import { useAppStore } from '../../store/appStore';
import type { ElementState, ArrayDisplayMode } from '../../types/api';
import { ConceptTrigger } from '../common/ConceptExplainer';

export interface ArrayModeConfig {
  id: ArrayDisplayMode;
  stateNumber: string;
  label: string;
  shortLabel: string;
  badge: string;
  description: string;
}

export const ARRAY_MODES: ArrayModeConfig[] = [
  {
    id: 'ideal',
    stateNumber: '01',
    label: '01 — IDEAL EXCITATION',
    shortLabel: 'IDEAL',
    badge: 'REFERENCE STATE',
    description: 'Theoretical excitation reference without mutual coupling or hardware impairments (uniform unit amplitude, progressive steering delay).',
  },
  {
    id: 'hardware',
    stateNumber: '02',
    label: '02 — OBSERVED / IMPAIRED',
    shortLabel: 'OBSERVED / IMPAIRED',
    badge: 'HARDWARE IMPAIRMENT',
    description: 'Actual hardware transceiver state perturbed by reciprocal mutual coupling (Z-matrix) and RF gain/phase noise errors.',
  },
  {
    id: 'pinn_diagnosis',
    stateNumber: '03',
    label: '03 — V4 ESTIMATED COMPENSATION',
    shortLabel: 'V4 ESTIMATED COMPENSATION',
    badge: 'PINN-CAL V4 ESTIMATE',
    description: 'Complex compensation coefficients ĉ = c_prior + Δc estimated by PINN-Cal V4 (Tikhonov δ=0.15, bounded residual α=0.50).',
  },
  {
    id: 'calibrated',
    stateNumber: '04',
    label: '04 — POST-COMPENSATION STATE',
    shortLabel: 'POST-COMPENSATION',
    badge: 'RESTORED STATE',
    description: 'Restored excitation state resulting from applying estimated V4 compensation to observed hardware state: V_post,n = V_hw,n · ĉ_n.',
  },
];

export function phaseToHue(phaseDeg: number): number {
  return ((phaseDeg % 360) + 360) % 360;
}

export interface ElementVisualState {
  hue: number;
  heightScale: number;
  opacity: number;
  isImpaired: boolean;
  phaseVal: number;
  gainVal: number;
  magnitudeVal: number;
  reVal: number;
  imVal: number;
}

export function getElementVisuals(elem: ElementState, mode: ArrayDisplayMode): ElementVisualState {
  const isSevereImpaired =
    Math.abs(elem.gain_error_db) > 1.2 || Math.abs(elem.phase_error_deg) > 12;

  switch (mode) {
    case 'ideal':
      return {
        hue: phaseToHue(elem.ideal_phase_deg),
        heightScale: 1.0, // nominal baseline
        opacity: 0.95,
        isImpaired: false,
        phaseVal: elem.ideal_phase_deg,
        gainVal: 0.0,
        magnitudeVal: elem.ideal_magnitude,
        reVal: elem.ideal_real,
        imVal: elem.ideal_imag,
      };

    case 'hardware':
      return {
        hue: phaseToHue(elem.hardware_phase_deg),
        heightScale: Math.min(1.45, Math.max(0.45, elem.hardware_magnitude)),
        opacity: 0.95,
        isImpaired: isSevereImpaired,
        phaseVal: elem.hardware_phase_deg,
        gainVal: elem.gain_error_db,
        magnitudeVal: elem.hardware_magnitude,
        reVal: elem.hardware_real,
        imVal: elem.hardware_imag,
      };

    case 'pinn_diagnosis': {
      // V4 Estimated Compensation Coefficients c_hat_n
      const cpRe = elem.pinn_correction_real ?? elem.analytical_correction_real ?? 1.0;
      const cpIm = elem.pinn_correction_imag ?? elem.analytical_correction_imag ?? 0.0;
      let cMag = elem.pinn_correction_magnitude;
      let cPhase = elem.pinn_correction_phase_deg;

      if (cMag == null) {
        cMag = Math.sqrt(cpRe * cpRe + cpIm * cpIm);
      }
      if (cPhase == null) {
        cPhase = (Math.atan2(cpIm, cpRe) * 180) / Math.PI;
      }
      const estGainDb = 20 * Math.log10(Math.max(cMag, 1e-12));

      return {
        hue: phaseToHue(cPhase),
        heightScale: Math.min(1.45, Math.max(0.45, cMag)),
        opacity: 0.95,
        isImpaired: Math.abs(cPhase) > 10 || Math.abs(estGainDb) > 1.2,
        phaseVal: cPhase,
        gainVal: estGainDb,
        magnitudeVal: cMag,
        reVal: cpRe,
        imVal: cpIm,
      };
    }

    case 'calibrated':
    default: {
      // Post-Compensation State: V_post,n = V_hw,n · c_hat,n
      const cpRe = elem.pinn_correction_real ?? elem.analytical_correction_real ?? 1.0;
      const cpIm = elem.pinn_correction_imag ?? elem.analytical_correction_imag ?? 0.0;
      const vCorrRe = elem.hardware_real * cpRe - elem.hardware_imag * cpIm;
      const vCorrIm = elem.hardware_real * cpIm + elem.hardware_imag * cpRe;
      const corrPhase = (Math.atan2(vCorrIm, vCorrRe) * 180) / Math.PI;
      const corrMag = Math.sqrt(vCorrRe * vCorrRe + vCorrIm * vCorrIm);
      const corrGainDb = 20 * Math.log10(Math.max(corrMag, 1e-12));

      return {
        hue: phaseToHue(corrPhase),
        heightScale: Math.min(1.45, Math.max(0.45, corrMag)),
        opacity: 0.95,
        isImpaired: false,
        phaseVal: corrPhase,
        gainVal: corrGainDb,
        magnitudeVal: corrMag,
        reVal: vCorrRe,
        imVal: vCorrIm,
      };
    }
  }
}

export const AntennaArray: React.FC = () => {
  const {
    simulationResult,
    simulateState,
    arrayDisplayMode,
    setArrayDisplayMode,
    selectedElement,
    setSelectedElement,
    config,
    superMode,
  } = useAppStore();

  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const elements = simulationResult?.elements;
  const numElements = elements?.length ?? 64;
  const isPinnAvailable = simulationResult?.pinn_available ?? true;
  const targetAngle = config.target_angle_deg;

  // Selected element index normalized (default to 16 if unselected)
  const activeSelected = selectedElement ?? 16;

  // Canvas Geometry Dimensions: Dominant 64-Element Horizontal Linear Aperture
  const SVG_WIDTH = 1200;
  const SVG_HEIGHT = 290;
  const MARGIN_X = 40;
  const GROUND_Y = 222;
  const SUBSTRATE_Y = 208;
  const FEED_LENGTH = 38;
  const PATCH_BASE_H = 60;
  const PATCH_W = 12.8;
  const TOTAL_APERTURE_W = SVG_WIDTH - MARGIN_X * 2 - PATCH_W;
  const STEP_X = TOTAL_APERTURE_W / Math.max(numElements - 1, 1);

  const activeMode = arrayDisplayMode;
  const currentModeConfig = ARRAY_MODES.find((m) => m.id === activeMode) ?? ARRAY_MODES[1];

  // Steering Heading Vector Coordinates (Broadside normal is vertical pointing up)
  const apertureCenterX = SVG_WIDTH / 2;
  const apertureCenterY = 96;
  const headingLength = 76;
  const targetAngleRad = (targetAngle * Math.PI) / 180;
  const headingEndX = apertureCenterX + headingLength * Math.sin(targetAngleRad);
  const headingEndY = apertureCenterY - headingLength * Math.cos(targetAngleRad);

  return (
    <div className="hero-antenna-container" aria-label="Live 64-Element Uniform Linear Array Experience">
      {/* 1. PRIMARY PRESENTATION HEADER */}
      <div className="antenna-header-bar">
        <div className="antenna-title-group">
          <div className="antenna-title-row">
            <h2 className="antenna-title">
              LIVE 64-ELEMENT UNIFORM LINEAR ARRAY (ULA)
            </h2>
            <span className="antenna-tag-28ghz">28 GHz mmWAVE</span>
            <span className="antenna-tag-badge">{currentModeConfig.badge}</span>
            <ConceptTrigger conceptKey="mutual_coupling" label="Coupling Physics" />
          </div>
          <p className="antenna-subtitle">
            Primary Research Instrument · 64-Element Uniform Linear Array · 0.5 λ Inter-Element Spacing (5.35 mm) · Reciprocal Z-Matrix
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          {superMode && (
            <div className="antenna-super-tag" title="Super Presentation Mode Active">
              <span className="super-star">★</span>
              <span>SUPER ARRAY</span>
            </div>
          )}

          {/* Dynamic Steering Status Badge */}
          <div className="antenna-steering-pill" title="Array broadside steering target angle">
            <span className="steering-pill-label">STEERING TARGET:</span>
            <span className="steering-pill-value">
              θ₀ = {targetAngle > 0 ? `+${targetAngle.toFixed(1)}°` : `${targetAngle.toFixed(1)}°`}
            </span>
          </div>
        </div>
      </div>

      {/* 2. COMPACT TECHNICAL INFORMATION STRIP (Requirement 14) */}
      <div className="antenna-tech-strip" role="region" aria-label="Array Technical Parameters">
        <div className="tech-strip-item">
          <span className="strip-key">APERTURE:</span>
          <span className="strip-val">64 ELEMENTS</span>
        </div>
        <span className="strip-sep">·</span>
        <div className="tech-strip-item">
          <span className="strip-key">FREQUENCY:</span>
          <span className="strip-val">28 GHz</span>
        </div>
        <span className="strip-sep">·</span>
        <div className="tech-strip-item">
          <span className="strip-key">INTER-ELEMENT SPACING:</span>
          <span className="strip-val">0.5 λ (5.35 mm)</span>
        </div>
        <span className="strip-sep">·</span>
        <div className="tech-strip-item">
          <span className="strip-key">GEOMETRY:</span>
          <span className="strip-val">UNIFORM LINEAR ARRAY (ULA)</span>
        </div>
        <span className="strip-sep">·</span>
        <div className="tech-strip-item highlight">
          <span className="strip-key">TARGET ANGLE:</span>
          <span className="strip-val">θ₀ = {targetAngle > 0 ? `+${targetAngle}°` : `${targetAngle}°`}</span>
        </div>
        <span className="strip-sep">·</span>
        <div className="tech-strip-item">
          <span className="strip-key">V4 PRECONDITIONER:</span>
          <span className="strip-val">δ = 0.15 · α = 0.50</span>
        </div>
      </div>

      {/* 3. FOUR ARRAY STATES SWITCHER (Requirements 4, 5, 6, 7, 8) */}
      <div className="mode-tabs-wrapper" role="region" aria-label="Array Semantic States">
        <div className="mode-tabs" role="tablist" aria-label="Four semantic array excitation states">
          {ARRAY_MODES.map((m) => {
            if (m.id === 'pinn_diagnosis' && !isPinnAvailable) return null;
            const isSelected = activeMode === m.id;
            return (
              <button
                key={m.id}
                role="tab"
                id={`array-tab-${m.id}`}
                aria-selected={isSelected}
                aria-controls="array-canvas-view"
                className={`mode-tab ${isSelected ? 'active' : ''}`}
                onClick={() => setArrayDisplayMode(m.id)}
                title={m.description}
              >
                <span className="mode-tab-title">{m.label}</span>
                <span className="mode-tab-sub">{m.shortLabel}</span>
              </button>
            );
          })}
        </div>
        <div className="active-mode-description-bar">
          <span className="active-mode-desc-icon">ℹ</span>
          <span className="active-mode-desc-text">{currentModeConfig.description}</span>
        </div>
      </div>

      {/* 4. DOMINANT 64-ELEMENT ULA APERTURE SVG CANVAS */}
      <div className="antenna-canvas-wrapper" id="array-canvas-view">
        {simulateState === 'computing' && (
          <div className="antenna-computing-overlay" aria-live="polite">
            <div className="computing-spinner-bar">
              <div className="computing-spinner-glow" />
            </div>
            <span className="computing-text">
              Solving 64-port mutual impedance network & evaluating V4 neural residual…
            </span>
          </div>
        )}

        <svg
          viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
          className="antenna-svg-canvas"
          role="img"
          aria-label={`Interactive 64-element uniform linear antenna array in ${currentModeConfig.label}`}
        >
          <defs>
            {/* Gradients */}
            <linearGradient id="patchCopperGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.9" />
              <stop offset="40%" stopColor="#d97706" stopOpacity="0.95" />
              <stop offset="100%" stopColor="#b45309" stopOpacity="1" />
            </linearGradient>

            <linearGradient id="substrateGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#1e293b" />
              <stop offset="100%" stopColor="#0f172a" />
            </linearGradient>

            {/* Selection Halo Filter */}
            <filter id="activeHalo" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="5" floodColor="#0284c7" floodOpacity="0.9" />
            </filter>
            <filter id="patchShadow" x="-20%" y="-10%" width="140%" height="130%">
              <feDropShadow dx="0" dy="2" stdDeviation="1.5" floodColor="#0f172a" floodOpacity="0.25" />
            </filter>
          </defs>

          {/* A. FREE-SPACE RADIATED WAVEFIELD (Aperture Radiated Wavefronts) */}
          <g className="radiated-wavefront-group" opacity={superMode ? 0.95 : 0.75}>
            {/* Wavefront Arcs propagating toward target steering direction */}
            {[20, 42, 64, 86].map((distY, wIdx) => {
              const yLevel = 100 - distY;
              const xOffset = distY * Math.tan(targetAngleRad);
              const isCalibratedOrIdeal = activeMode === 'calibrated' || activeMode === 'ideal';
              const strokeColor =
                activeMode === 'ideal'
                  ? '#0284c7'
                  : activeMode === 'calibrated'
                  ? '#10b981'
                  : activeMode === 'hardware'
                  ? '#ef4444'
                  : '#8b5cf6';

              return (
                <path
                  key={wIdx}
                  d={`M ${MARGIN_X + 16 + xOffset} ${yLevel} Q ${apertureCenterX + xOffset} ${yLevel - (isCalibratedOrIdeal ? 7 : 16)} ${SVG_WIDTH - MARGIN_X - 16 + xOffset} ${yLevel}`}
                  stroke={strokeColor}
                  strokeWidth={isCalibratedOrIdeal ? 2 : 1.5}
                  strokeDasharray={activeMode === 'hardware' ? '5,4' : 'none'}
                  fill="none"
                  opacity={0.85 - wIdx * 0.18}
                  className={superMode ? 'animated-wave-arc' : ''}
                />
              );
            })}

            {/* B. APERTURE STEERING DIRECTION INDICATOR (Requirement 13) */}
            <g className="steering-vector-group">
              {/* Broadside Reference Line (dashed normal) */}
              <line
                x1={apertureCenterX}
                y1={apertureCenterY}
                x2={apertureCenterX}
                y2={apertureCenterY - 65}
                stroke="#94a3b8"
                strokeWidth={1}
                strokeDasharray="2,3"
                opacity={0.6}
              />

              {/* Dynamic Target Steering Heading Ray */}
              <line
                x1={apertureCenterX}
                y1={apertureCenterY}
                x2={headingEndX}
                y2={headingEndY}
                stroke="#0284c7"
                strokeWidth={2.5}
              />

              {/* Arrow Head */}
              <circle cx={headingEndX} cy={headingEndY} r={4.5} fill="#0284c7" stroke="#ffffff" strokeWidth={1.5} />

              {/* Aperture Origin Center Pivot */}
              <circle cx={apertureCenterX} cy={apertureCenterY} r={3.5} fill="#0f172a" />

              {/* Dynamic Target Heading Badge */}
              <g transform={`translate(${headingEndX + (targetAngle >= 0 ? 12 : -110)}, ${headingEndY - 8})`}>
                <rect x={-4} y={-14} width={105} height={18} rx={4} fill="#0f172a" fillOpacity={0.88} />
                <text
                  x={48}
                  y={-2}
                  textAnchor="middle"
                  fill="#38bdf8"
                  fontSize="10"
                  fontWeight="800"
                  fontFamily="var(--font-mono)"
                >
                  TARGET: {targetAngle > 0 ? `+${targetAngle}°` : `${targetAngle}°`}
                </text>
              </g>
            </g>
          </g>

          {/* C. RECIPROCAL MUTUAL COUPLING LINKS (Active in Hardware Mode) */}
          <g className="coupling-links-group" opacity={activeMode === 'hardware' ? 0.7 : 0.25}>
            {Array.from({ length: numElements - 1 }, (_, i) => {
              const x1 = MARGIN_X + i * STEP_X + PATCH_W / 2;
              const x2 = MARGIN_X + (i + 1) * STEP_X + PATCH_W / 2;
              const y = SUBSTRATE_Y - FEED_LENGTH - 8;
              return (
                <path
                  key={i}
                  d={`M ${x1} ${y} Q ${(x1 + x2) / 2} ${y + 7} ${x2} ${y}`}
                  stroke={activeMode === 'hardware' ? '#ef4444' : '#64748b'}
                  strokeWidth={activeMode === 'hardware' ? 1.2 : 0.8}
                  strokeDasharray="2,2"
                  fill="none"
                />
              );
            })}
          </g>

          {/* D. DIELECTRIC SUBSTRATE LAYER (ε_r = 2.2 Microstrip Bed) */}
          <rect
            x={MARGIN_X - 16}
            y={SUBSTRATE_Y}
            width={TOTAL_APERTURE_W + PATCH_W + 32}
            height={14}
            rx={3}
            fill="url(#substrateGrad)"
            stroke="#334155"
            strokeWidth={1.5}
          />
          <text
            x={MARGIN_X}
            y={SUBSTRATE_Y + 10}
            fill="#94a3b8"
            fontSize="8.5"
            fontWeight="700"
            fontFamily="var(--font-mono)"
            letterSpacing="0.06em"
          >
            DIELECTRIC SUBSTRATE (ε_r = 2.2) · 28 GHz mmWAVE HIGH-FREQUENCY MICROSTRIP APERTURE
          </text>

          {/* E. CONTINUOUS RF GROUND REFERENCE PLANE */}
          <rect
            x={MARGIN_X - 16}
            y={GROUND_Y}
            width={TOTAL_APERTURE_W + PATCH_W + 32}
            height={16}
            rx={3}
            fill="#0f172a"
            stroke="#020617"
            strokeWidth={1.5}
          />
          <line
            x1={MARGIN_X - 12}
            y1={GROUND_Y + 2}
            x2={SVG_WIDTH - MARGIN_X + 12}
            y2={GROUND_Y + 2}
            stroke="#f59e0b"
            strokeWidth={2}
          />
          <text
            x={MARGIN_X}
            y={GROUND_Y + 12}
            fill="#cbd5e1"
            fontSize="8.5"
            fontWeight="700"
            fontFamily="var(--font-mono)"
          >
            CONTINUOUS RF GROUND PLANE · RECIPROCAL Z-IMPEDANCE NETWORK (Z_0 = 50 Ω)
          </text>

          {/* F. 64 ENGINEERED MICROSTRIP PATCH ELEMENTS (Requirements 3 & 4) */}
          {elements ? (
            elements.map((elem, idx) => {
              const x = MARGIN_X + idx * STEP_X;
              const isSelected = activeSelected === idx;
              const isHovered = hoveredIdx === idx;
              const vis = getElementVisuals(elem, activeMode);

              const patchH = PATCH_BASE_H * vis.heightScale;
              const patchY = SUBSTRATE_Y - FEED_LENGTH - patchH;
              const feedY = SUBSTRATE_Y - FEED_LENGTH;
              const patchColor = `hsl(${vis.hue}, 86%, 48%)`;

              return (
                <g
                  key={idx}
                  id={`element-patch-${idx + 1}`}
                  className="engineered-patch-element"
                  onClick={() => setSelectedElement(idx)}
                  onMouseEnter={() => setHoveredIdx(idx)}
                  onMouseLeave={() => setHoveredIdx(null)}
                  style={{ cursor: 'pointer' }}
                  role="button"
                  tabIndex={0}
                  aria-label={`Element ${idx + 1} of 64: Phase ${vis.phaseVal.toFixed(1)}°, Gain ${vis.gainVal.toFixed(2)} dB`}
                >
                  {/* Selected Element Focus Ray & Halo (Requirement 18) */}
                  {isSelected && (
                    <g filter="url(#activeHalo)">
                      {/* Vertical Inspection Laser Line */}
                      <line
                        x1={x + PATCH_W / 2}
                        y1={12}
                        x2={x + PATCH_W / 2}
                        y2={patchY}
                        stroke="#0284c7"
                        strokeWidth={2}
                        strokeDasharray="3,2"
                      />
                      <circle cx={x + PATCH_W / 2} cy={16} r={5} fill="#0284c7" stroke="#ffffff" strokeWidth={1.5} />
                      {/* Enclosing Focus Rect */}
                      <rect
                        x={x - 3}
                        y={patchY - 4}
                        width={PATCH_W + 6}
                        height={patchH + FEED_LENGTH + 7}
                        rx={5}
                        fill="none"
                        stroke="#0284c7"
                        strokeWidth={2.5}
                      />
                    </g>
                  )}

                  {/* 2.5D Pedestal Base Under Patch */}
                  <rect
                    x={x - 1}
                    y={patchY + 2}
                    width={PATCH_W + 2}
                    height={patchH + 3}
                    rx={3}
                    fill="#1e293b"
                    opacity={0.3}
                  />

                  {/* Microstrip Coaxial Feed Pin */}
                  <line
                    x1={x + PATCH_W / 2}
                    y1={SUBSTRATE_Y}
                    x2={x + PATCH_W / 2}
                    y2={feedY}
                    stroke={isSelected ? '#0284c7' : '#475569'}
                    strokeWidth={isSelected ? 3 : 2}
                  />

                  {/* Solder Via Ring */}
                  <circle
                    cx={x + PATCH_W / 2}
                    cy={SUBSTRATE_Y}
                    r={3.2}
                    fill="#0f172a"
                    stroke="#475569"
                    strokeWidth={0.8}
                  />
                  <circle
                    cx={x + PATCH_W / 2}
                    cy={SUBSTRATE_Y}
                    r={1.6}
                    fill={isSelected ? '#38bdf8' : '#f59e0b'}
                  />

                  {/* Microstrip Patch Radiator (Height = Magnitude, Color = Phase) */}
                  <rect
                    x={x}
                    y={patchY}
                    width={PATCH_W}
                    height={patchH}
                    rx={2.5}
                    fill={patchColor}
                    fillOpacity={vis.opacity}
                    stroke={isSelected ? '#0f172a' : isHovered ? '#0284c7' : '#334155'}
                    strokeWidth={isSelected ? 2.5 : 1.2}
                    filter="url(#patchShadow)"
                    style={{
                      transition: 'height 0.2s ease, y 0.2s ease, fill 0.2s ease',
                      transformOrigin: `${x + PATCH_W / 2}px ${patchY + patchH}px`,
                    }}
                  />

                  {/* Top Specular Rim */}
                  <line
                    x1={x + 1}
                    y1={patchY + 1}
                    x2={x + PATCH_W - 1}
                    y2={patchY + 1}
                    stroke="#ffffff"
                    strokeWidth={1}
                    strokeOpacity={0.85}
                  />

                  {/* Phase Direction Needle Inside Patch */}
                  <g transform={`translate(${x + PATCH_W / 2}, ${patchY + patchH / 2})`}>
                    <line
                      x1={0}
                      y1={0}
                      x2={((PATCH_W - 3) / 2) * Math.cos((vis.phaseVal * Math.PI) / 180)}
                      y2={-((PATCH_W - 3) / 2) * Math.sin((vis.phaseVal * Math.PI) / 180)}
                      stroke="#ffffff"
                      strokeWidth={1.5}
                      strokeLinecap="round"
                    />
                  </g>

                  {/* Severe Hardware Impairment Alert Marker (Mode 02) */}
                  {activeMode === 'hardware' && vis.isImpaired && (
                    <circle
                      cx={x + PATCH_W / 2}
                      cy={patchY - 6}
                      r={3}
                      fill="#ef4444"
                      stroke="#ffffff"
                      strokeWidth={1}
                    />
                  )}

                  {/* V4 Neural Estimate Marker (Mode 03) */}
                  {activeMode === 'pinn_diagnosis' && (
                    <circle
                      cx={x + PATCH_W / 2}
                      cy={patchY - 5}
                      r={3}
                      fill="#8b5cf6"
                      stroke="#ffffff"
                      strokeWidth={1}
                    />
                  )}

                  {/* Post-Compensation Alignment Marker (Mode 04) */}
                  {activeMode === 'calibrated' && (
                    <circle
                      cx={x + PATCH_W / 2}
                      cy={patchY - 5}
                      r={2.5}
                      fill="#10b981"
                      stroke="#ffffff"
                      strokeWidth={0.8}
                    />
                  )}

                  {/* G. NUMBERED 1–64 ELEMENT NUMBER LABELS (Requirement 3) */}
                  <g className="element-number-label">
                    {/* Tick line from ground plane to number */}
                    <line
                      x1={x + PATCH_W / 2}
                      y1={GROUND_Y + 16}
                      x2={x + PATCH_W / 2}
                      y2={GROUND_Y + 22}
                      stroke={isSelected ? '#0284c7' : '#94a3b8'}
                      strokeWidth={isSelected ? 1.5 : 0.8}
                    />

                    {/* Active Element Number Badge */}
                    {isSelected && (
                      <rect
                        x={x + PATCH_W / 2 - 8}
                        y={GROUND_Y + 24}
                        width={16}
                        height={14}
                        rx={3}
                        fill="#0284c7"
                      />
                    )}

                    <text
                      x={x + PATCH_W / 2}
                      y={GROUND_Y + 35}
                      textAnchor="middle"
                      fill={isSelected ? '#ffffff' : (idx + 1) % 8 === 0 || idx === 0 || idx === 63 ? '#0f172a' : '#64748b'}
                      fontSize={isSelected ? '9' : '8'}
                      fontFamily="var(--font-mono)"
                      fontWeight={isSelected ? '900' : (idx + 1) % 8 === 0 ? '700' : '500'}
                    >
                      {idx + 1}
                    </text>
                  </g>
                </g>
              );
            })
          ) : (
            // Awaiting Simulation Loading State
            Array.from({ length: 64 }, (_, i) => {
              const x = MARGIN_X + i * STEP_X;
              return (
                <g key={i} opacity={0.3}>
                  <line x1={x + PATCH_W / 2} y1={SUBSTRATE_Y} x2={x + PATCH_W / 2} y2={SUBSTRATE_Y - FEED_LENGTH} stroke="#94a3b8" strokeWidth={1.5} />
                  <rect x={x} y={SUBSTRATE_Y - FEED_LENGTH - PATCH_BASE_H} width={PATCH_W} height={PATCH_BASE_H} rx={2} fill="#cbd5e1" stroke="#94a3b8" strokeWidth={0.8} />
                </g>
              );
            })
          )}
        </svg>

        {/* Hover Tooltip Overlay */}
        {hoveredIdx !== null && elements && elements[hoveredIdx] && (
          <div
            className="antenna-hover-tooltip"
            style={{
              left: `${Math.min(88, Math.max(12, (hoveredIdx / 63) * 100))}%`,
            }}
          >
            <div className="tooltip-title">
              <span>PATCH ELEMENT #{hoveredIdx + 1}</span>
              <span className="tooltip-badge">PORT #{hoveredIdx + 1}</span>
            </div>
            <div className="tooltip-metrics">
              <div className="t-row">
                <span>Phase Error:</span>
                <strong style={{ color: '#ef4444' }}>
                  {elements[hoveredIdx].phase_error_deg > 0 ? '+' : ''}
                  {elements[hoveredIdx].phase_error_deg.toFixed(1)}°
                </strong>
              </div>
              <div className="t-row">
                <span>Gain Error:</span>
                <strong style={{ color: '#f59e0b' }}>
                  {elements[hoveredIdx].gain_error_db > 0 ? '+' : ''}
                  {elements[hoveredIdx].gain_error_db.toFixed(1)} dB
                </strong>
              </div>
              <div className="t-row">
                <span>V4 Est Comp (ĉ):</span>
                <strong style={{ color: '#8b5cf6' }}>
                  {elements[hoveredIdx].pinn_correction_phase_deg != null
                    ? `${elements[hoveredIdx].pinn_correction_phase_deg.toFixed(1)}°`
                    : 'Active'}
                </strong>
              </div>
            </div>
          </div>
        )}

        {/* 5. ARRAY VISUAL LEGEND (Requirement 12) */}
        <div className="antenna-legend-bar" role="region" aria-label="Array Visualization Legend">
          {/* Phase Hue Ribbon */}
          <div className="legend-section">
            <div className="legend-phase-wheel">
              <span className="legend-title">EXCITATION PHASE (ϕ):</span>
              <div className="phase-wheel-gradient" />
              <span className="legend-mono-range">−180° ↔ 0° ↔ +180°</span>
            </div>
          </div>

          {/* Magnitude Bar Scale */}
          <div className="legend-section">
            <div className="legend-mag-scale">
              <span className="legend-title">MAGNITUDE ENCODING:</span>
              <div className="mag-scale-visual">
                <span className="mag-bar h-low" title="Low magnitude (0.5x)" />
                <span className="mag-bar h-norm" title="Nominal magnitude (1.0x)" />
                <span className="mag-bar h-high" title="High magnitude (1.5x)" />
              </div>
              <span className="legend-mono-range">Height = |V| or |ĉ|</span>
            </div>
          </div>

          {/* Semantic Role Markers */}
          <div className="legend-section">
            <div className="legend-symbols-group">
              <div className="legend-item" title="Ideal unperturbed baseline">
                <span className="sym-dot ideal-blue" />
                <span>Ideal Reference</span>
              </div>
              <div className="legend-item" title="Hardware impairment perturbation">
                <span className="sym-dot alert-red" />
                <span>Observed / Impaired</span>
              </div>
              <div className="legend-item" title="PINN-Cal V4 estimated complex compensation">
                <span className="sym-dot pinn-purple" />
                <span>V4 Est. Compensation</span>
              </div>
              <div className="legend-item" title="Restored array state after compensation">
                <span className="sym-dot post-green" />
                <span>Post-Compensation</span>
              </div>
              <div className="legend-item" title="Currently selected element for inspection">
                <span className="sym-ring halo-blue" />
                <span>Selected Element</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
