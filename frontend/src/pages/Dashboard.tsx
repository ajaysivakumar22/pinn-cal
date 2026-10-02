import React, { useEffect } from 'react';
import { HeroSection } from '../components/layout/HeroSection';
import { WorkflowPipeline } from '../components/workflow/WorkflowPipeline';
import { AntennaArray } from '../components/array/AntennaArray';
import { ControlPanel } from '../components/controls/ControlPanel';
import { PinnDetectorFlow } from '../components/array/PinnDetectorFlow';
import { CalibrationFlow } from '../components/array/CalibrationFlow';
import { ElementInspector } from '../components/array/ElementInspector';
import { BeforeAfterView } from '../components/charts/BeforeAfterView';
import { CalibrationPanel } from '../components/charts/CalibrationPanel';
import { ElementAnalysisChart } from '../components/charts/ElementAnalysisChart';
import { SimulationStory } from '../components/common/SimulationStory';
import { FrankyGuide } from '../components/common/FrankyGuide';
import { StarCelebration } from '../components/common/StarCelebration';
import { useAppStore } from '../store/appStore';

export const Dashboard: React.FC = () => {
  const { simulationResult, simulateState, simulateError, runSimulation } = useAppStore();

  useEffect(() => {
    if (!simulationResult && simulateState === 'idle') {
      runSimulation();
    }
  }, []);

  return (
    <div className="science-story-experience">
      {/* 0. CINEMATIC HERO OPENING */}
      <HeroSection />

      {/* 1. WORKFLOW PIPELINE PROGRESSION */}
      <WorkflowPipeline />

      {/* 2. FRANKY VIRTUAL RF ENGINEER GUIDE (Active in Super Mode) */}
      <FrankyGuide />

      {/* 3. STAR CELEBRATION TOAST */}
      <StarCelebration />

      {/* ERROR NOTICE IF BACKEND FAILS */}
      {simulateError && (
        <div className="simulation-error-banner" role="alert">
          <strong>Simulation Error:</strong> {simulateError}
        </div>
      )}

      {/* ====================================================================
          STORY STAGE 01: WHAT IS WRONG?
          The physical antenna array experience + experiment controls
          ==================================================================== */}
      <section id="section-simulation" className="story-stage-zone stage-array-zone" aria-label="Antenna Array and Hardware Impairment">
        <div className="stage-zone-header">
          <div className="stage-num-badge">01</div>
          <div>
            <h2 className="stage-zone-title">THE PHYSICAL IMPAIRMENT</h2>
            <p className="stage-zone-desc">
              Hardware variations and mutual coupling disturb the antenna excitation, deflecting the beam.
            </p>
          </div>
        </div>

        {/* 2.5D Microstrip Patch Array */}
        <div className="stage-visual-container">
          <AntennaArray />
        </div>

        {/* Tactile Experiment Controls Console */}
        <div className="stage-controls-container">
          <div className="stage-sub-header">
            <span className="sub-icon">⚙️</span>
            <h3 className="sub-title">EXPERIMENT BENCH CONTROLS</h3>
            <span className="sub-desc">Adjust steering heading, array specs, hardware errors, and noise</span>
          </div>
          <ControlPanel />
        </div>
      </section>

      {/* ====================================================================
          STORY STAGE 02 & 03: WHAT DOES PINN SEE & ESTIMATE?
          The Neural Detector Architecture
          ==================================================================== */}
      <PinnDetectorFlow />

      {/* ====================================================================
          STORY STAGE 04: WHAT DO WE APPLY?
          Transceiver-Level Inverse Compensation & Deep Inspection
          ==================================================================== */}
      <section id="section-calibration" className="story-stage-zone stage-compensation-zone" aria-label="Transceiver Inverse Compensation">
        <div className="stage-zone-header">
          <div className="stage-num-badge">04</div>
          <div>
            <h2 className="stage-zone-title">TRANSCEIVER INVERSE COMPENSATION</h2>
            <p className="stage-zone-desc">
              Applying complex compensation weights C = H⁻¹ to cancel element-level phase skews and gain errors.
            </p>
          </div>
        </div>

        {/* Transformation Flow Bar */}
        <CalibrationFlow />

        {/* Deep Transceiver Inspector */}
        <div className="stage-inspector-wrapper">
          <ElementInspector />
        </div>
      </section>

      {/* ====================================================================
          STORY STAGE 05: WHAT CHANGED?
          Signature Beam Restoration (Before vs After Far-Field Pattern)
          ==================================================================== */}
      <section id="section-beam" className="story-stage-zone stage-beam-zone" aria-label="Beam Restoration">
        <div className="stage-zone-header">
          <div className="stage-num-badge">05</div>
          <div>
            <h2 className="stage-zone-title">FAR-FIELD BEAM RESTORATION</h2>
            <p className="stage-zone-desc">
              Evaluating how closely the calibrated radiation main lobe is restored toward the commanded target direction.
            </p>
          </div>
        </div>

        <BeforeAfterView />
      </section>

      {/* ====================================================================
          STORY STAGE 06: HOW DO WE VERIFY IT?
          Residual Analysis, Neutral Benchmark Comparison & Simulation Story
          ==================================================================== */}
      <section id="section-robustness" className="story-stage-zone stage-verify-zone" aria-label="Verification and Scientific Analysis">
        <div className="stage-zone-header">
          <div className="stage-num-badge">06</div>
          <div>
            <h2 className="stage-zone-title">PERFORMANCE VERIFICATION</h2>
            <p className="stage-zone-desc">
              Rigorous evaluation across population residuals, method comparisons, and physical findings.
            </p>
          </div>
        </div>

        {/* 2-Column Comparison & Population Analysis */}
        <div className="verification-grid">
          <div className="verify-col-left">
            <CalibrationPanel />
          </div>
          <div className="verify-col-right">
            <ElementAnalysisChart />
          </div>
        </div>

        {/* Factual Simulation Story Debriefing */}
        <SimulationStory />
      </section>
    </div>
  );
};
