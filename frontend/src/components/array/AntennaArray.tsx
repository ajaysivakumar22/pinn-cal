import React, { useState } from 'react';
import { useAppStore } from '../../store/appStore';
import type { ElementState, ArrayDisplayMode } from '../../types/api';
import { ConceptTrigger } from '../common/ConceptExplainer';

const MODES: { id: ArrayDisplayMode; label: string; description: string }[] = [
  { id: 'ideal', label: '1. Ideal Signal', description: 'Theoretical excitation without hardware impairments or mutual coupling.' },
  { id: 'hardware', label: '2. Measured (Impaired)', description: 'Impaired transceiver state with phase/gain errors and coupling perturbation.' },
  { id: 'pinn_diagnosis', label: '3. PINN Diagnosis', description: 'Neural network estimated per-element impairment profile.' },
  { id: 'calibrated', label: '4. Calibrated', description: 'Array excitation after applying inverse compensation weights C = H⁻¹.' },
];

function phaseToHue(phaseDeg: number): number {
  return ((phaseDeg % 360) + 360) % 360;
}

function getElementVisuals(elem: ElementState, mode: ArrayDisplayMode): {
  hue: number;
  heightScale: number;
  opacity: number;
  isImpaired: boolean;
  phaseVal: number;
  gainVal: number;
} {
  const isImpaired = Math.abs(elem.gain_error_db) > 1.5 || Math.abs(elem.phase_error_deg) > 15;

  switch (mode) {
    case 'ideal':
      return {
        hue: phaseToHue(elem.ideal_phase_deg),
        heightScale: Math.min(elem.ideal_magnitude, 1.4),
        opacity: 0.95,
        isImpaired: false,
        phaseVal: elem.ideal_phase_deg,
        gainVal: 0,
      };

    case 'hardware':
      return {
        hue: phaseToHue(elem.hardware_phase_deg),
        heightScale: Math.min(elem.hardware_magnitude, 1.4),
        opacity: 0.95,
        isImpaired,
        phaseVal: elem.hardware_phase_deg,
        gainVal: elem.gain_error_db,
      };

    case 'pinn_diagnosis':
      const estPhase = elem.pinn_estimated_phase_error_deg ?? elem.phase_error_deg;
      const estGain = elem.pinn_estimated_gain_error_db ?? elem.gain_error_db;
      return {
        hue: phaseToHue(estPhase),
        heightScale: 1.0,
        opacity: 0.9,
        isImpaired: Math.abs(estPhase) > 10,
        phaseVal: estPhase,
        gainVal: estGain,
      };

    case 'calibrated':
    default:
      // Corrected state: V_corr = V_hw * C
      const caRe = elem.analytical_correction_real;
      const caIm = elem.analytical_correction_imag;
      const vCorrRe = elem.hardware_real * caRe - elem.hardware_imag * caIm;
      const vCorrIm = elem.hardware_real * caIm + elem.hardware_imag * caRe;
      const corrPhase = (Math.atan2(vCorrIm, vCorrRe) * 180) / Math.PI;
      const corrMag = Math.sqrt(vCorrRe * vCorrRe + vCorrIm * vCorrIm);
      return {
        hue: phaseToHue(corrPhase),
        heightScale: Math.min(corrMag, 1.4),
        opacity: 0.95,
        isImpaired: false,
        phaseVal: corrPhase,
        gainVal: 0,
      };
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

  // Canvas Geometry Dimensions (Enhanced 2.5D Scale)
  const SVG_WIDTH = 1040;
  const SVG_HEIGHT = 260;
  const MARGIN_X = 36;
  const GROUND_Y = 205;
  const SUBSTRATE_Y = 195;
  const FEED_LENGTH = 38;
  const PATCH_BASE_H = 54;
  const PATCH_W = Math.max(9, Math.min(12, (SVG_WIDTH - MARGIN_X * 2) / numElements - 3.8));
  const STEP_X = (SVG_WIDTH - MARGIN_X * 2 - PATCH_W) / Math.max(numElements - 1, 1);

  const activeMode = arrayDisplayMode;

  return (
    <div className="hero-antenna-container">
      {/* Top Header & Mode Switcher Bar */}
      <div className="antenna-header-bar">
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <h2 className="antenna-title">
              LIVE 64-ELEMENT MICROSTRIP PATCH ARRAY
            </h2>
            <span className="antenna-tag-28ghz">28 GHz mmWAVE</span>
            <ConceptTrigger conceptKey="mutual_coupling" label="Coupling Physics" />
          </div>
          <p className="antenna-subtitle">
            Uniform Linear Array (ULA) · 0.5λ Inter-element Spacing (5.35 mm) · Reciprocal Z-Impedance Network
          </p>
        </div>

        {/* Array State Mode Tabs */}
        <div className="mode-tabs-wrapper">
          <div className="mode-tabs" role="tablist" aria-label="Antenna array operational display mode">
            {MODES.map((m) => {
              if (m.id === 'pinn_diagnosis' && !isPinnAvailable) return null;
              const isSelected = activeMode === m.id;
              return (
                <button
                  key={m.id}
                  role="tab"
                  aria-selected={isSelected}
                  className={`mode-tab ${isSelected ? 'active' : ''}`}
                  onClick={() => setArrayDisplayMode(m.id)}
                  title={m.description}
                >
                  <span className="mode-tab-title">{m.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* SVG 2.5D Engineered Antenna System Canvas */}
      <div className="antenna-canvas-wrapper">
        {simulateState === 'computing' && (
          <div className="antenna-computing-overlay">
            <div className="computing-spinner-bar">
              <div className="computing-spinner-glow" />
            </div>
            <span className="computing-text">
              Computing physics & solving mutual impedance network…
            </span>
          </div>
        )}

        <svg
          viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
          className="antenna-svg-canvas"
          role="img"
          aria-label="Interactive 64-element microstrip patch antenna array"
        >
          <defs>
            {/* Patch Metallic Surface Gradients */}
            <linearGradient id="patchCopperGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.9" />
              <stop offset="40%" stopColor="#d97706" stopOpacity="0.95" />
              <stop offset="100%" stopColor="#b45309" stopOpacity="1" />
            </linearGradient>

            <linearGradient id="patchGoldGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#fef08a" />
              <stop offset="60%" stopColor="#eab308" />
              <stop offset="100%" stopColor="#ca8a04" />
            </linearGradient>

            {/* Substrate Dielectric Bed Gradient */}
            <linearGradient id="substrateGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#1e293b" />
              <stop offset="100%" stopColor="#0f172a" />
            </linearGradient>

            {/* RF Wavefield Radial Emission */}
            <radialGradient id="rfEmissionGrad" cx="50%" cy="100%" r="90%">
              <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.4" />
              <stop offset="60%" stopColor="#0284c7" stopOpacity="0.15" />
              <stop offset="100%" stopColor="#0284c7" stopOpacity="0" />
            </radialGradient>

            {/* Drop Shadows */}
            <filter id="patchShadow" x="-20%" y="-10%" width="140%" height="130%">
              <feDropShadow dx="0" dy="3" stdDeviation="2" floodColor="#0f172a" floodOpacity="0.25" />
            </filter>
            <filter id="activeHalo" x="-40%" y="-40%" width="180%" height="180%">
              <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor="#0284c7" floodOpacity="0.8" />
            </filter>
          </defs>

          {/* 1. Free-Space Radiated Wavefront Field (Aperture Rays) */}
          <g className="radiated-wavefront-group" opacity={superMode ? 0.9 : 0.65}>
            {/* Steered RF Wavefront Arcs */}
            {[25, 45, 65, 85].map((distY, wIdx) => {
              const yLevel = 100 - distY;
              const xOffset = distY * Math.tan((targetAngle * Math.PI) / 180);
              const isCalibratedOrIdeal = activeMode === 'calibrated' || activeMode === 'ideal';
              const strokeColor = isCalibratedOrIdeal ? '#10b981' : activeMode === 'hardware' ? '#ef4444' : '#8b5cf6';

              return (
                <path
                  key={wIdx}
                  d={`M ${MARGIN_X + 20 + xOffset} ${yLevel} Q ${SVG_WIDTH / 2 + xOffset} ${yLevel - (isCalibratedOrIdeal ? 8 : 16)} ${SVG_WIDTH - MARGIN_X - 20 + xOffset} ${yLevel}`}
                  stroke={strokeColor}
                  strokeWidth={isCalibratedOrIdeal ? 2 : 1.5}
                  strokeDasharray={activeMode === 'hardware' ? '6,4' : 'none'}
                  fill="none"
                  opacity={0.8 - wIdx * 0.16}
                  className={superMode ? 'animated-wave-arc' : ''}
                />
              );
            })}

            {/* Steering Target Heading Vector */}
            <g transform={`translate(${SVG_WIDTH / 2 + 55 * Math.tan((targetAngle * Math.PI) / 180)}, 36)`}>
              <line
                x1={0}
                y1={30}
                x2={35 * Math.sin((targetAngle * Math.PI) / 180)}
                y2={-20}
                stroke="#10b981"
                strokeWidth={2}
                strokeDasharray="4,3"
              />
              <circle cx={35 * Math.sin((targetAngle * Math.PI) / 180)} cy={-20} r={4} fill="#10b981" />
              <text
                x={45 * Math.sin((targetAngle * Math.PI) / 180)}
                y={-22}
                fill="#166534"
                fontSize="11"
                fontWeight="800"
                fontFamily="var(--font-mono)"
              >
                θ = {targetAngle > 0 ? `+${targetAngle}°` : `${targetAngle}°`}
              </text>
            </g>
          </g>

          {/* 2. Mutual Coupling Links between Adjacent Radiators */}
          <g className="coupling-links-group" opacity={activeMode === 'hardware' ? 0.75 : 0.35}>
            {Array.from({ length: numElements - 1 }, (_, i) => {
              const x1 = MARGIN_X + i * STEP_X + PATCH_W / 2;
              const x2 = MARGIN_X + (i + 1) * STEP_X + PATCH_W / 2;
              const y = SUBSTRATE_Y - FEED_LENGTH - 10;
              return (
                <path
                  key={i}
                  d={`M ${x1} ${y} Q ${(x1 + x2) / 2} ${y + 8} ${x2} ${y}`}
                  stroke="#64748b"
                  strokeWidth={1}
                  strokeDasharray="2,2"
                  fill="none"
                />
              );
            })}
          </g>

          {/* 3. Substrate PCB Bed (Dielectric Layer) */}
          <rect
            x={MARGIN_X - 14}
            y={SUBSTRATE_Y}
            width={SVG_WIDTH - MARGIN_X * 2 + 28}
            height={16}
            rx={4}
            fill="url(#substrateGrad)"
            stroke="#334155"
            strokeWidth={1.5}
          />
          <text
            x={MARGIN_X}
            y={SUBSTRATE_Y + 11}
            fill="#94a3b8"
            fontSize="8.5"
            fontWeight="700"
            fontFamily="var(--font-mono)"
            letterSpacing="0.08em"
          >
            DIELECTRIC SUBSTRATE LAYER (ε_r = 2.2) · 28 GHz HIGH-FREQUENCY MICROSTRIP
          </text>

          {/* 4. Common RF Ground Reference Plane */}
          <rect
            x={MARGIN_X - 14}
            y={GROUND_Y}
            width={SVG_WIDTH - MARGIN_X * 2 + 28}
            height={18}
            rx={3}
            fill="#0f172a"
            stroke="#020617"
            strokeWidth={1.5}
          />
          <line
            x1={MARGIN_X - 10}
            y1={GROUND_Y + 2}
            x2={SVG_WIDTH - MARGIN_X + 10}
            y2={GROUND_Y + 2}
            stroke="#f59e0b"
            strokeWidth={2}
          />
          <text
            x={MARGIN_X}
            y={GROUND_Y + 13}
            fill="#cbd5e1"
            fontSize="8.5"
            fontWeight="700"
            fontFamily="var(--font-mono)"
          >
            COMMON RF GROUND REFERENCE · CONTINUOUS GROUND PLANE · RECIPROCAL Z-NETWORK
          </text>

          {/* 5. 64 Engineered Microstrip Radiator Elements */}
          {elements ? (
            elements.map((elem, idx) => {
              const x = MARGIN_X + idx * STEP_X;
              const isSelected = selectedElement === idx;
              const isHovered = hoveredIdx === idx;
              const vis = getElementVisuals(elem, activeMode);

              const patchH = PATCH_BASE_H * vis.heightScale;
              const patchY = SUBSTRATE_Y - FEED_LENGTH - patchH;
              const feedY = SUBSTRATE_Y - FEED_LENGTH;
              const phaseHue = vis.hue;
              const patchColor = `hsl(${phaseHue}, 85%, 48%)`;

              return (
                <g
                  key={idx}
                  className="engineered-patch-element"
                  onClick={() => setSelectedElement(idx)}
                  onMouseEnter={() => setHoveredIdx(idx)}
                  onMouseLeave={() => setHoveredIdx(null)}
                  style={{ cursor: 'pointer' }}
                  role="button"
                  tabIndex={0}
                  aria-label={`Element ${idx + 1}: Phase ${elem.hardware_phase_deg.toFixed(1)}°`}
                >
                  {/* Selected Inspection Halo & Focus Ray */}
                  {isSelected && (
                    <g filter="url(#activeHalo)">
                      <line
                        x1={x + PATCH_W / 2}
                        y1={10}
                        x2={x + PATCH_W / 2}
                        y2={patchY}
                        stroke="#0284c7"
                        strokeWidth={2}
                        strokeDasharray="3,3"
                      />
                      <circle
                        cx={x + PATCH_W / 2}
                        cy={14}
                        r={5}
                        fill="#0284c7"
                      />
                      <rect
                        x={x - 3}
                        y={patchY - 4}
                        width={PATCH_W + 6}
                        height={patchH + FEED_LENGTH + 8}
                        rx={5}
                        fill="none"
                        stroke="#0284c7"
                        strokeWidth={2.5}
                      />
                    </g>
                  )}

                  {/* Dielectric Substrate Base Pedestal (2.5D layer under patch) */}
                  <rect
                    x={x - 1}
                    y={patchY + 2}
                    width={PATCH_W + 2}
                    height={patchH + 4}
                    rx={3}
                    fill="#1e293b"
                    opacity={0.35}
                  />

                  {/* Microstrip feed pin (Coaxial center conductor) */}
                  <line
                    x1={x + PATCH_W / 2}
                    y1={SUBSTRATE_Y}
                    x2={x + PATCH_W / 2}
                    y2={feedY}
                    stroke={isSelected ? '#0284c7' : '#475569'}
                    strokeWidth={isSelected ? 3 : 2}
                  />

                  {/* Feed soldering via with ground ring */}
                  <circle
                    cx={x + PATCH_W / 2}
                    cy={SUBSTRATE_Y}
                    r={3.5}
                    fill="#0f172a"
                    stroke="#475569"
                    strokeWidth={0.8}
                  />
                  <circle
                    cx={x + PATCH_W / 2}
                    cy={SUBSTRATE_Y}
                    r={1.8}
                    fill={isSelected ? '#38bdf8' : '#f59e0b'}
                  />

                  {/* Microstrip Patch Radiator (Metallic Beveled Surface) */}
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
                      transition: 'all 0.25s ease',
                      transformOrigin: `${x + PATCH_W / 2}px ${patchY + patchH}px`,
                    }}
                  />

                  {/* Microstrip Inset Feed Notch Indication */}
                  <line
                    x1={x + PATCH_W / 2}
                    y1={patchY + patchH - 3}
                    x2={x + PATCH_W / 2}
                    y2={patchY + patchH}
                    stroke="#0f172a"
                    strokeWidth={2}
                    strokeOpacity={0.6}
                  />

                  {/* Top Bevel Metallic Specular Rim */}
                  <line
                    x1={x + 1}
                    y1={patchY + 1}
                    x2={x + PATCH_W - 1}
                    y2={patchY + 1}
                    stroke="#ffffff"
                    strokeWidth={1.2}
                    strokeOpacity={0.85}
                  />

                  {/* Phase Angle Direction Needle inside patch */}
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

                  {/* Severe Hardware Impairment Alert Marker */}
                  {activeMode === 'hardware' && vis.isImpaired && (
                    <g>
                      <circle
                        cx={x + PATCH_W / 2}
                        cy={patchY - 6}
                        r={3.5}
                        fill="#ef4444"
                        stroke="#ffffff"
                        strokeWidth={1}
                      />
                      <polygon
                        points={`${x + PATCH_W / 2},${patchY - 8} ${x + PATCH_W / 2 - 2},${patchY - 4} ${x + PATCH_W / 2 + 2},${patchY - 4}`}
                        fill="#ffffff"
                      />
                    </g>
                  )}

                  {/* PINN Estimated Impairment Pointer */}
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

                  {/* Element Index Numbers */}
                  {(idx === 0 || (idx + 1) % 8 === 0 || idx === 63) && (
                    <text
                      x={x + PATCH_W / 2}
                      y={GROUND_Y + 34}
                      textAnchor="middle"
                      fill={isSelected ? '#0284c7' : '#64748b'}
                      fontSize="10"
                      fontFamily="var(--font-mono)"
                      fontWeight={isSelected ? '800' : '600'}
                    >
                      {idx + 1}
                    </text>
                  )}
                </g>
              );
            })
          ) : (
            // Awaiting Simulation Loading State
            Array.from({ length: 64 }, (_, i) => {
              const x = MARGIN_X + i * STEP_X;
              return (
                <g key={i} opacity={0.35}>
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
              left: `${Math.min(85, Math.max(15, (hoveredIdx / 63) * 100))}%`,
            }}
          >
            <div className="tooltip-title">
              <span>PATCH ELEMENT #{hoveredIdx + 1}</span>
              <span className="tooltip-badge">CH-{hoveredIdx + 1}</span>
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
                <span>PINN Est:</span>
                <strong style={{ color: '#8b5cf6' }}>
                  {(elements[hoveredIdx].pinn_estimated_phase_error_deg ?? elements[hoveredIdx].phase_error_deg).toFixed(1)}°
                </strong>
              </div>
            </div>
          </div>
        )}

        {/* Comprehensive Color Wheel & Semantic Legend */}
        <div className="antenna-legend-bar">
          <div className="legend-phase-wheel">
            <span className="legend-title">EXCITATION PHASE (ϕ):</span>
            <div className="phase-wheel-gradient" />
            <span className="legend-mono-range">−180° ↔ 0° ↔ +180°</span>
          </div>

          <div className="legend-symbols-group">
            <div className="legend-item">
              <span className="sym-box pcb-ground" />
              <span>Ground Reference</span>
            </div>
            <div className="legend-item">
              <span className="sym-dot alert-red" />
              <span>Impaired Channel</span>
            </div>
            <div className="legend-item">
              <span className="sym-dot pinn-purple" />
              <span>PINN Diagnosis</span>
            </div>
            <div className="legend-item">
              <span className="sym-ring halo-blue" />
              <span>Inspected Transceiver</span>
            </div>
            <div className="legend-item">
              <span className="sym-wave green-wave" />
              <span>Restored Wavefront</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
