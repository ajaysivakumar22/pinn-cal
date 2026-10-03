import React, { useMemo } from 'react';
import { AsyncPlot as Plot } from '../components/common/AsyncPlot';
import { useAppStore } from '../store/appStore';
import { ConceptTrigger } from '../components/common/ConceptExplainer';

export const BeamLab: React.FC = () => {
  const { simulationResult, config, setConfig, runSimulation, simulateState, simulateError, setActiveView, theme, superMode } =
    useAppStore();

  const beam = simulationResult?.beam;
  const metrics = simulationResult?.metrics;
  const targetAngle = config.target_angle_deg;
  const isComputing = simulateState === 'computing';
  const hasResult = !!simulationResult && !!beam;
  const hasError = simulateState === 'error' || !!simulateError;

  const isDark = theme === 'dark';

  // Primary Far-Field Radiation Pattern Traces
  const chartData = useMemo(() => {
    if (!beam) return [];

    const defs = [
      {
        key: 'ideal',
        label: 'Ideal Target Pattern (Synthesized)',
        trace: beam.ideal,
        color: '#16a34a',
        width: 2,
        dash: 'dash',
      },
      {
        key: 'uncalibrated',
        label: 'Uncalibrated / Impaired (Distorted)',
        trace: beam.uncalibrated,
        color: '#dc2626',
        width: 2.2,
        dash: 'solid',
      },
      {
        key: 'analytical',
        label: 'Analytical Baseline (Compensated)',
        trace: beam.analytical,
        color: '#0891b2',
        width: 2.2,
        dash: 'solid',
      },
    ];

    if (beam.pinn && simulationResult?.pinn_available) {
      defs.push({
        key: 'pinn',
        label: 'PINN-Cal V4 (Restored Pattern)',
        trace: beam.pinn,
        color: '#ea580c',
        width: 3.0,
        dash: 'solid',
      });
    }

    return defs
      .filter((d) => d.trace !== null && d.trace !== undefined)
      .map((d) => ({
        x: d.trace!.angles_deg,
        y: d.trace!.power_db,
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: d.label,
        line: { color: d.color, width: d.width, dash: d.dash as any },
        hovertemplate: `<b>%{data.name}</b><br>Angle θ: %{x:.1f}°<br>Radiation Power: %{y:.2f} dB<extra></extra>`,
      }));
  }, [beam, simulationResult?.pinn_available]);

  // Target angle marker & Computed Peak marker
  const shapes = useMemo(() => {
    const s: any[] = [
      // Commanded Target Angle
      {
        type: 'line' as const,
        x0: targetAngle,
        x1: targetAngle,
        y0: -50,
        y1: 4,
        line: { color: '#0284c7', width: 2, dash: 'dot' as const },
      },
    ];

    // PINN-Cal V4 Peak Marker if available
    if (beam?.pinn) {
      s.push({
        type: 'line' as const,
        x0: beam.pinn.peak_angle_deg,
        x1: beam.pinn.peak_angle_deg,
        y0: -50,
        y1: 4,
        line: { color: '#ea580c', width: 1.5, dash: 'dash' as const },
      });
    }

    return s;
  }, [targetAngle, beam?.pinn]);

  const annotations = useMemo(() => {
    if (!beam) return [];

    const ann: any[] = [
      {
        x: targetAngle,
        y: 2,
        xref: 'x' as const,
        yref: 'y' as const,
        text: `Target: ${targetAngle > 0 ? `+${targetAngle.toFixed(1)}°` : `${targetAngle.toFixed(1)}°`}`,
        showarrow: true,
        arrowhead: 2,
        arrowsize: 1,
        arrowwidth: 1.5,
        arrowcolor: '#0284c7',
        font: { family: "'JetBrains Mono', monospace", size: 10, color: '#0369a1' },
        bgcolor: isDark ? '#082f49' : '#e0f2fe',
        bordercolor: '#38bdf8',
        borderwidth: 1,
        borderpad: 3,
      },
    ];

    if (beam.pinn) {
      ann.push({
        x: beam.pinn.peak_angle_deg,
        y: -1.5,
        xref: 'x' as const,
        yref: 'y' as const,
        text: `V4 Peak: ${beam.pinn.peak_angle_deg > 0 ? `+${beam.pinn.peak_angle_deg.toFixed(2)}°` : `${beam.pinn.peak_angle_deg.toFixed(2)}°`}`,
        showarrow: true,
        arrowhead: 2,
        arrowsize: 1,
        arrowwidth: 1.5,
        arrowcolor: '#ea580c',
        font: { family: "'JetBrains Mono', monospace", size: 10, color: '#9a3412' },
        bgcolor: isDark ? '#431407' : '#ffedd5',
        bordercolor: '#fb923c',
        borderwidth: 1,
        borderpad: 3,
      });
    }

    return ann;
  }, [beam, targetAngle, isDark]);

  const layout = useMemo(
    () => ({
      margin: { t: 30, r: 20, b: 45, l: 55 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: {
        family: "'Inter', sans-serif",
        size: 11,
        color: isDark ? '#94a3b8' : '#475569',
      },
      legend: {
        orientation: 'h' as const,
        x: 0,
        y: 1.14,
        bgcolor: isDark ? 'rgba(15, 23, 42, 0.85)' : 'rgba(255, 255, 255, 0.92)',
        bordercolor: isDark ? '#334155' : '#e2e8f0',
        borderwidth: 1,
        font: { size: 10.5 },
      },
      xaxis: {
        title: { text: 'Far-Field Spatial Radiation Angle θ (degrees)', font: { size: 11 } },
        range: [-90, 90],
        gridcolor: isDark ? '#1e293b' : '#e2e8f0',
        zerolinecolor: isDark ? '#334155' : '#cbd5e1',
        tickfont: { size: 10 },
      },
      yaxis: {
        title: { text: 'Normalized Radiation Power P(θ) (dB)', font: { size: 11 } },
        range: [-50, 4],
        gridcolor: isDark ? '#1e293b' : '#e2e8f0',
        zerolinecolor: isDark ? '#334155' : '#cbd5e1',
        tickfont: { size: 10 },
      },
      shapes,
      annotations,
      hovermode: 'x unified' as const,
    }),
    [shapes, annotations, isDark]
  );

  const pinnPeak = beam?.pinn?.peak_angle_deg ?? metrics?.beam_peak_angle_deg;
  const pinnError = beam?.pinn?.pointing_error_deg ?? metrics?.pointing_error_deg;
  const pinnCorr = beam?.pinn?.beam_correlation ?? metrics?.beam_correlation;
  const pinnHpbw = beam?.pinn?.beamwidth_3db_deg ?? metrics?.beamwidth_3db_deg;
  const pinnPsll = beam?.pinn?.psll_db ?? metrics?.psll_db;

  const anaPeak = beam?.analytical?.peak_angle_deg;
  const anaError = beam?.analytical?.pointing_error_deg;
  const anaCorr = beam?.analytical?.beam_correlation;
  const anaHpbw = beam?.analytical?.beamwidth_3db_deg;
  const anaPsll = beam?.analytical?.psll_db;

  const uncalPeak = beam?.uncalibrated?.peak_angle_deg;
  const uncalError = beam?.uncalibrated?.pointing_error_deg;
  const uncalCorr = beam?.uncalibrated?.beam_correlation;
  const uncalHpbw = beam?.uncalibrated?.beamwidth_3db_deg;
  const uncalPsll = beam?.uncalibrated?.psll_db;

  return (
    <div className="beam-lab-page">
      {/* 3. PAGE HEADER */}
      <header className="beam-header-section">
        <div className="beam-title-row">
          <div className="beam-title-group">
            <h1 className="beam-page-title">BEAM RESTORATION LAB</h1>
            <p className="beam-page-subtitle">
              Evaluate how PINN-Cal V4 restores the commanded far-field radiation pattern under hardware impairment and mutual coupling.
            </p>
          </div>
          <div className="beam-identity-strip">
            <span className="strip-item">64 ELEMENTS</span>
            <span className="strip-dot">·</span>
            <span className="strip-item">28 GHz</span>
            <span className="strip-dot">·</span>
            <span className="strip-item">0.5 λ ULA</span>
            <span className="strip-dot">·</span>
            <span className="strip-item highlight-blue">
              TARGET: {targetAngle > 0 ? `+${targetAngle.toFixed(1)}°` : `${targetAngle.toFixed(1)}°`}
            </span>
            <span className="strip-dot">·</span>
            <span className="strip-item highlight-orange">PINN-CAL V4</span>
            {superMode && hasResult && (
              <>
                <span className="strip-dot">·</span>
                <span className="strip-item highlight-orange" style={{ fontWeight: 800 }}>★ BEAM RESTORED · SUPER</span>
              </>
            )}
          </div>
        </div>
      </header>

      {/* 4. COMMAND STEERING CONTROL BAR */}
      <section className="card beam-card beam-steering-bar">
        <div className="steering-control-meta">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span className="header-icon blue-icon">🎯</span>
            <div>
              <span className="steering-label">TARGET STEERING ANGLE (θ₀)</span>
              <div className="steering-sub">Commanded main beam radiation heading relative to broadside.</div>
            </div>
          </div>
          <div className="steering-readout-wrap">
            <span className="steering-current-val">
              {targetAngle > 0 ? `+${targetAngle.toFixed(1)}°` : `${targetAngle.toFixed(1)}°`}
            </span>
          </div>
        </div>

        <div className="steering-slider-row">
          <input
            id="beam-target-angle"
            aria-label="Target Steering Angle"
            type="range"
            className="beam-steering-slider"
            min={-85}
            max={85}
            step={1}
            value={targetAngle}
            onChange={(e) => setConfig({ target_angle_deg: Number(e.target.value) })}
          />
          <div className="slider-ticks">
            <span>-85° (Extreme Left)</span>
            <span>0° (Broadside)</span>
            <span>+85° (Extreme Right)</span>
          </div>
        </div>

        <div className="steering-action-row">
          <button
            type="button"
            className={`btn-beam-resynthesize ${isComputing ? 'computing' : ''}`}
            onClick={runSimulation}
            disabled={isComputing}
            aria-busy={isComputing}
          >
            <span className="action-icon">{isComputing ? '⏳' : '⚡'}</span>
            <span>{isComputing ? 'COMPUTING FAR-FIELD APERTURE...' : 'RE-SYNTHESIZE BEAM PATTERN'}</span>
          </button>
          <ConceptTrigger conceptKey="beamforming" label="Spatial Beamforming Guide" />
        </div>
      </section>

      {/* 22 & 23. EMPTY / ERROR STATES */}
      {hasError ? (
        <div className="card beam-card beam-alert-banner error" role="alert">
          <span className="alert-icon">⚠️</span>
          <div>
            <h3 className="alert-title">BEAM DATA UNAVAILABLE</h3>
            <p className="alert-desc">{simulateError || 'Simulation encountered an unexpected error.'}</p>
          </div>
          <button type="button" className="btn-alert-retry" onClick={runSimulation}>
            Retry Simulation
          </button>
        </div>
      ) : !hasResult ? (
        <div className="card beam-card beam-alert-banner empty" role="status">
          <span className="alert-icon">📡</span>
          <div>
            <h3 className="alert-title">READY FOR BEAM ANALYSIS</h3>
            <p className="alert-desc">
              Run the calibration bench above to generate and inspect the current synthesized far-field radiation pattern.
            </p>
          </div>
          <button type="button" className="btn-alert-action" onClick={runSimulation}>
            Run Calibration Bench
          </button>
        </div>
      ) : null}

      {/* 5, 6, 7, 8: PRIMARY BEAM CHART WORKSTATION */}
      <div className="beam-workstation-grid">
        {/* Left Column: Primary Radiation Pattern Plot */}
        <section className="card beam-card beam-chart-card">
          <div className="card-header beam-card-header">
            <div>
              <h2 className="beam-section-title">FAR-FIELD RADIATION PATTERN P(θ)</h2>
              <span className="header-meta">
                Normalized spatial power distribution across 181 angular points (−90° to +90°)
              </span>
            </div>
            <div className="chart-header-badges">
              <span className="badge-tag green-tag">Ideal Target</span>
              <span className="badge-tag red-tag">Impaired</span>
              <span className="badge-tag cyan-tag">Analytical</span>
              <span className="badge-tag orange-tag">PINN-Cal V4</span>
            </div>
          </div>

          <div className="card-body beam-chart-body">
            {!beam ? (
              <div className="awaiting-panel-large">
                <span>Awaiting spatial simulation data...</span>
              </div>
            ) : (
              <div className="beam-plot-container">
                <Plot
                  data={chartData}
                  layout={layout}
                  useResizeHandler
                  style={{ width: '100%', height: '100%' }}
                  config={{ displayModeBar: true, responsive: true }}
                />
              </div>
            )}
          </div>
        </section>

        {/* Right Column: Pointing Alignment & Telemetry Inspection */}
        <section className="beam-side-column">
          {/* 8. BEAM PEAK & POINTING ALIGNMENT CARD */}
          <div className="card beam-card beam-alignment-card">
            <div className="card-header beam-card-header">
              <h3 className="beam-card-title">MAIN LOBE POINTING ALIGNMENT</h3>
            </div>
            <div className="card-body beam-alignment-body">
              <div className="alignment-item">
                <div className="alignment-label-row">
                  <span className="align-label">COMMANDED TARGET (θ₀)</span>
                  <span className="align-val blue">
                    {targetAngle > 0 ? `+${targetAngle.toFixed(1)}°` : `${targetAngle.toFixed(1)}°`}
                  </span>
                </div>
                <div className="align-desc">Desired bore-sight heading direction</div>
              </div>

              <div className="alignment-item">
                <div className="alignment-label-row">
                  <span className="align-label">UNCALIBRATED PEAK</span>
                  <span className="align-val red">
                    {uncalPeak != null ? `${uncalPeak > 0 ? `+${uncalPeak.toFixed(2)}°` : `${uncalPeak.toFixed(2)}°`}` : '—'}
                  </span>
                </div>
                <div className="align-desc">
                  Impaired defection error: {uncalError != null ? `${Math.abs(uncalError).toFixed(2)}°` : '—'}
                </div>
              </div>

              <div className="alignment-item">
                <div className="alignment-label-row">
                  <span className="align-label">PINN-CAL V4 PEAK</span>
                  <span className="align-val orange">
                    {pinnPeak != null ? `${pinnPeak > 0 ? `+${pinnPeak.toFixed(2)}°` : `${pinnPeak.toFixed(2)}°`}` : '—'}
                  </span>
                </div>
                <div className="align-desc">
                  Restored pointing error: <strong className="green-text">{pinnError != null ? `${Math.abs(pinnError).toFixed(2)}°` : '—'}</strong>
                </div>
              </div>

              <div className="alignment-summary-banner">
                <div className="banner-mini-label">POINTING RECOVERY DELTA</div>
                <div className="banner-delta-text">
                  {uncalError != null ? `${Math.abs(uncalError).toFixed(2)}°` : '—'} →{' '}
                  <span className="green-text">{pinnError != null ? `${Math.abs(pinnError).toFixed(2)}°` : '—'}</span>
                </div>
                <div className="banner-sub-text">
                  Beam peak restored within {pinnError != null ? Math.abs(pinnError).toFixed(2) : '—'}° of target heading.
                </div>
              </div>
            </div>
          </div>

          {/* 12. CALIBRATION EFFECT SUMMARY */}
          <div className="card beam-card beam-effect-card">
            <div className="card-header beam-card-header">
              <h3 className="beam-card-title">WHAT CHANGED?</h3>
            </div>
            <div className="card-body beam-effect-body">
              <p className="effect-para">
                <strong>Observed hardware impairments</strong> (transceiver gain and phase variations) perturb the array excitation, deflecting the radiation lobe and causing severe sidelobe leakage.
              </p>
              <p className="effect-para">
                <strong>PINN-Cal V4</strong> estimates complex compensation weights via its Tikhonov physics preconditioner and bounded neural residual network.
              </p>
              <p className="effect-para">
                Applying those estimated weights produces the <strong>post-compensation aperture excitation</strong>, realigning constructive spatial interference and restoring the far-field radiation beam.
              </p>
            </div>
          </div>

          {/* 13 & 14. NAVIGATION ACTION SHORTCUTS */}
          <div className="card beam-card beam-nav-card">
            <div className="card-header beam-card-header">
              <h3 className="beam-card-title">DEEP-DIVE INSPECTION</h3>
            </div>
            <div className="card-body beam-nav-body">
              <button
                type="button"
                className="btn-deep-dive"
                onClick={() => setActiveView('calibration')}
              >
                <span>INSPECT ELEMENT CALIBRATION</span>
                <span className="arrow">→</span>
              </button>
              <button
                type="button"
                className="btn-deep-dive"
                onClick={() => setActiveView('simulation')}
              >
                <span>VIEW 64-ELEMENT ARRAY</span>
                <span className="arrow">→</span>
              </button>
            </div>
          </div>
        </section>
      </div>

      {/* 9. BEFORE → AFTER STORY (COMPARISON SECTION) */}
      <section className="beam-full-width-section">
        <div className="card beam-card before-after-story-card">
          <div className="card-header beam-card-header">
            <div>
              <h2 className="beam-section-title">BEAM RESTORATION: BEFORE → AFTER CALIBRATION</h2>
              <span className="header-meta">Physical progression from impaired excitation to post-compensation radiation</span>
            </div>
            <ConceptTrigger conceptKey="beam_correlation" label="Beam Correlation Guide" />
          </div>

          <div className="card-body" style={{ padding: 18 }}>
            <div className="before-after-story-grid">
              {/* STAGE 1: BEFORE */}
              <div className="story-stage-col before-col">
                <div className="stage-header">
                  <span className="stage-number">01</span>
                  <div className="stage-title-wrap">
                    <span className="stage-label">BEFORE CALIBRATION</span>
                    <span className="stage-sub">Uncalibrated / Impaired Hardware</span>
                  </div>
                </div>
                <div className="stage-card-body">
                  <div className="stage-stat-row">
                    <span className="stat-label">Beam Peak Heading:</span>
                    <span className="stat-val red">{uncalPeak != null ? `${uncalPeak.toFixed(1)}°` : '—'}</span>
                  </div>
                  <div className="stage-stat-row">
                    <span className="stat-label">Pointing Error (Δθ):</span>
                    <span className="stat-val red">{uncalError != null ? `${Math.abs(uncalError).toFixed(2)}°` : '—'}</span>
                  </div>
                  <div className="stage-stat-row">
                    <span className="stat-label">Beam Correlation:</span>
                    <span className="stat-val">{uncalCorr != null ? `${(uncalCorr * 100).toFixed(1)}%` : '—'}</span>
                  </div>
                  <div className="stage-stat-row">
                    <span className="stat-label">Peak Sidelobe (PSLL):</span>
                    <span className="stat-val">{uncalPsll != null ? `${uncalPsll.toFixed(1)} dB` : '—'}</span>
                  </div>
                  <div className="stage-desc-box red-tint">
                    Hardware phase drift and PA gain skew disrupt aperture phase coherence, spilling radiation power away from commanded heading.
                  </div>
                </div>
              </div>

              {/* TRANSITION ARROW */}
              <div className="story-flow-connector">
                <div className="connector-pill">
                  <span className="connector-model">PINN-Cal V4</span>
                  <span className="connector-sub">ĉ = c_prior + Δc</span>
                  <span className="connector-arrow">➔</span>
                </div>
              </div>

              {/* STAGE 2: AFTER */}
              <div className="story-stage-col after-col">
                <div className="stage-header">
                  <span className="stage-number">02</span>
                  <div className="stage-title-wrap">
                    <span className="stage-label">AFTER CALIBRATION</span>
                    <span className="stage-sub">PINN-Cal V4 Restored Aperture</span>
                  </div>
                </div>
                <div className="stage-card-body">
                  <div className="stage-stat-row">
                    <span className="stat-label">Beam Peak Heading:</span>
                    <span className="stat-val green">{pinnPeak != null ? `${pinnPeak.toFixed(1)}°` : '—'}</span>
                  </div>
                  <div className="stage-stat-row">
                    <span className="stat-label">Pointing Error (Δθ):</span>
                    <span className="stat-val green">{pinnError != null ? `${Math.abs(pinnError).toFixed(2)}°` : '—'}</span>
                  </div>
                  <div className="stage-stat-row">
                    <span className="stat-label">Beam Correlation:</span>
                    <span className="stat-val green">{pinnCorr != null ? `${(pinnCorr * 100).toFixed(1)}%` : '—'}</span>
                  </div>
                  <div className="stage-stat-row">
                    <span className="stat-label">Peak Sidelobe (PSLL):</span>
                    <span className="stat-val green">{pinnPsll != null ? `${pinnPsll.toFixed(1)} dB` : '—'}</span>
                  </div>
                  <div className="stage-desc-box green-tint">
                    Complex compensation weights realign constructive wave interference, restoring main lobe bore-sight and suppressing sidelobe leakage.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 10. BEAM METRIC WALL (6 LARGE METRIC CARDS) */}
      <section className="beam-full-width-section">
        <div className="card beam-card beam-metrics-wall-card">
          <div className="card-header beam-card-header">
            <h2 className="beam-section-title">BEAM METRIC WALL</h2>
            <span className="header-meta">Key Live Far-Field Radiation Performance Indicators</span>
          </div>
          <div className="card-body" style={{ padding: 18 }}>
            <div className="beam-metric-wall-grid">
              {/* Card 1: TARGET ANGLE */}
              <div className="beam-metric-card">
                <span className="metric-tag">COMMAND</span>
                <div className="metric-val-large blue">
                  {targetAngle > 0 ? `+${targetAngle.toFixed(1)}°` : `${targetAngle.toFixed(1)}°`}
                </div>
                <div className="metric-title">TARGET ANGLE</div>
                <div className="metric-sub">Commanded beam direction relative to broadside.</div>
              </div>

              {/* Card 2: BEAM PEAK */}
              <div className="beam-metric-card">
                <span className="metric-tag">OUTPUT PEAK</span>
                <div className="metric-val-large">
                  {pinnPeak != null ? `${pinnPeak > 0 ? `+${pinnPeak.toFixed(2)}°` : `${pinnPeak.toFixed(2)}°`}` : '—'}
                </div>
                <div className="metric-title">BEAM PEAK</div>
                <div className="metric-sub">Computed peak angular location of maximum radiation power.</div>
              </div>

              {/* Card 3: POINTING ERROR */}
              <div className="beam-metric-card">
                <span className="metric-tag">ACCURACY</span>
                <div className="metric-val-large green">
                  {pinnError != null ? `${Math.abs(pinnError).toFixed(2)}°` : '—'}
                </div>
                <div className="metric-title">POINTING ERROR</div>
                <div className="metric-sub">Absolute bore-sight deviation: |θ_peak − θ_target|.</div>
              </div>

              {/* Card 4: BEAM CORRELATION */}
              <div className="beam-metric-card">
                <span className="metric-tag">FIDELITY</span>
                <div className="metric-val-large green">
                  {pinnCorr != null ? `${(pinnCorr * 100).toFixed(1)}%` : '—'}
                </div>
                <div className="metric-title">BEAM CORRELATION</div>
                <div className="metric-sub">Inner product similarity vs ideal synthesized beam.</div>
              </div>

              {/* Card 5: 3 dB BEAMWIDTH */}
              <div className="beam-metric-card">
                <span className="metric-tag">APERTURE HPBW</span>
                <div className="metric-val-large">
                  {pinnHpbw != null ? `${pinnHpbw.toFixed(2)}°` : '—'}
                </div>
                <div className="metric-title">3 dB BEAMWIDTH</div>
                <div className="metric-sub">Half-power main lobe beamwidth across 64 elements.</div>
              </div>

              {/* Card 6: PSLL */}
              <div className="beam-metric-card">
                <span className="metric-tag">SUPPRESSION</span>
                <div className="metric-val-large">
                  {pinnPsll != null ? `${pinnPsll.toFixed(1)} dB` : '—'}
                </div>
                <div className="metric-title">PEAK SIDELOBE (PSLL)</div>
                <div className="metric-sub">Peak sidelobe level relative to main lobe peak.</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 11. METHOD COMPARISON: ANALYTICAL BASELINE vs PINN-CAL V4 */}
      <section className="beam-full-width-section">
        <div className="card beam-card beam-comparison-card">
          <div className="card-header beam-card-header">
            <div>
              <h2 className="beam-section-title">METHOD COMPARISON</h2>
              <span className="header-meta">Far-field radiation metrics evaluated under identical RF conditions</span>
            </div>
            <ConceptTrigger conceptKey="analytical_baseline" label="Physics Baseline Guide" />
          </div>
          <div className="card-body" style={{ padding: 0 }}>
            <div className="comparison-description-bar">
              Neutral engineering comparison between the linear Tikhonov analytical baseline and the PINN-Cal V4 residual architecture.
            </div>
            <div className="table-responsive">
              <table className="beam-comparison-table">
                <thead>
                  <tr>
                    <th>Radiation Metric</th>
                    <th>Uncalibrated Impaired</th>
                    <th>Analytical Baseline</th>
                    <th>PINN-Cal V4</th>
                    <th>Engineering Description</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="param-cell">
                      <strong>Beam Peak Angle</strong>
                    </td>
                    <td className="uncal-val">{uncalPeak != null ? `${uncalPeak.toFixed(2)}°` : '—'}</td>
                    <td className="ana-val">{anaPeak != null ? `${anaPeak.toFixed(2)}°` : '—'}</td>
                    <td className="pinn-val">{pinnPeak != null ? `${pinnPeak.toFixed(2)}°` : '—'}</td>
                    <td className="desc-cell">Computed maximum radiation direction</td>
                  </tr>
                  <tr>
                    <td className="param-cell">
                      <strong>Pointing Error (Δθ)</strong>
                    </td>
                    <td className="uncal-val">{uncalError != null ? `${Math.abs(uncalError).toFixed(2)}°` : '—'}</td>
                    <td className="ana-val">{anaError != null ? `${Math.abs(anaError).toFixed(2)}°` : '—'}</td>
                    <td className="pinn-val">{pinnError != null ? `${Math.abs(pinnError).toFixed(2)}°` : '—'}</td>
                    <td className="desc-cell">Angular shift from commanded target heading</td>
                  </tr>
                  <tr>
                    <td className="param-cell">
                      <strong>Beam Correlation</strong>
                    </td>
                    <td className="uncal-val">{uncalCorr != null ? `${(uncalCorr * 100).toFixed(1)}%` : '—'}</td>
                    <td className="ana-val">{anaCorr != null ? `${(anaCorr * 100).toFixed(1)}%` : '—'}</td>
                    <td className="pinn-val">{pinnCorr != null ? `${(pinnCorr * 100).toFixed(1)}%` : '—'}</td>
                    <td className="desc-cell">Pattern overlap fidelity vs ideal synthesized beam</td>
                  </tr>
                  <tr>
                    <td className="param-cell">
                      <strong>3 dB Beamwidth (HPBW)</strong>
                    </td>
                    <td className="uncal-val">{uncalHpbw != null ? `${uncalHpbw.toFixed(2)}°` : '—'}</td>
                    <td className="ana-val">{anaHpbw != null ? `${anaHpbw.toFixed(2)}°` : '—'}</td>
                    <td className="pinn-val">{pinnHpbw != null ? `${pinnHpbw.toFixed(2)}°` : '—'}</td>
                    <td className="desc-cell">Main lobe angular width at -3 dB half-power point</td>
                  </tr>
                  <tr>
                    <td className="param-cell">
                      <strong>Peak Sidelobe Level (PSLL)</strong>
                    </td>
                    <td className="uncal-val">{uncalPsll != null ? `${uncalPsll.toFixed(1)} dB` : '—'}</td>
                    <td className="ana-val">{anaPsll != null ? `${anaPsll.toFixed(1)} dB` : '—'}</td>
                    <td className="pinn-val">{pinnPsll != null ? `${pinnPsll.toFixed(1)} dB` : '—'}</td>
                    <td className="desc-cell">Energy suppression ratio of first adjacent sidelobes</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
