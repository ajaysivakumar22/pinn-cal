import React, { useMemo, useState } from 'react';
import { AsyncPlot as Plot } from '../common/AsyncPlot';
import { useAppStore } from '../../store/appStore';

function wrapDeg(deg: number): number {
  return ((((deg + 180) % 360) + 360) % 360) - 180;
}

export const ElementAnalysisChart: React.FC = () => {
  const { simulationResult } = useAppStore();
  const [metricTab, setMetricTab] = useState<'phase' | 'gain'>('phase');

  const elements = simulationResult?.elements;
  const pinnAvailable = simulationResult?.pinn_available ?? false;

  const chartData = useMemo(() => {
    if (!elements || elements.length === 0) return [];

    const indices = elements.map((e) => e.element_index + 1);

    if (metricTab === 'phase') {
      const impairedPhase = elements.map((e) => e.phase_error_deg);

      const anaPhaseResidual = elements.map((e) => {
        const vHwPhase = e.hardware_phase_deg;
        const cPhase = e.analytical_correction_phase_deg;
        const correctedPhase = vHwPhase + cPhase;
        return wrapDeg(correctedPhase - e.ideal_phase_deg);
      });

      const pinnPhaseResidual = pinnAvailable
        ? elements.map((e) => {
            if (e.pinn_correction_phase_deg === null) return 0;
            const vHwPhase = e.hardware_phase_deg;
            const cPhase = e.pinn_correction_phase_deg;
            const correctedPhase = vHwPhase + cPhase;
            return wrapDeg(correctedPhase - e.ideal_phase_deg);
          })
        : null;

      const traces: any[] = [
        {
          x: indices,
          y: impairedPhase,
          type: 'bar',
          name: 'Hardware Impairment',
          marker: { color: 'rgba(220, 38, 38, 0.45)', line: { color: '#dc2626', width: 1 } },
          hovertemplate: 'Elem %{x}: %{y:.2f}°<extra>Hardware</extra>',
        },
        {
          x: indices,
          y: anaPhaseResidual,
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Analytical Residual',
          line: { color: '#0891b2', width: 1.5 },
          marker: { size: 4, color: '#0891b2' },
          hovertemplate: 'Elem %{x}: %{y:.2f}°<extra>Analytical</extra>',
        },
      ];

      if (pinnAvailable && pinnPhaseResidual) {
        traces.push({
          x: indices,
          y: pinnPhaseResidual,
          type: 'scatter',
          mode: 'lines+markers',
          name: 'PINN Residual',
          line: { color: '#7c3aed', width: 1.5 },
          marker: { size: 4, color: '#7c3aed' },
          hovertemplate: 'Elem %{x}: %{y:.2f}°<extra>PINN</extra>',
        });
      }

      return traces;
    } else {
      const impairedGain = elements.map((e) => e.gain_error_db);

      const anaGainResidual = elements.map((e) => {
        const vHwMag = e.hardware_magnitude;
        const cMag = e.analytical_correction_magnitude;
        const vCorrMag = vHwMag * cMag;
        const vIdealMag = Math.max(e.ideal_magnitude, 1e-9);
        return 20 * Math.log10(Math.max(vCorrMag / vIdealMag, 1e-9));
      });

      const pinnGainResidual = pinnAvailable
        ? elements.map((e) => {
            if (e.pinn_correction_magnitude === null) return 0;
            const vHwMag = e.hardware_magnitude;
            const cMag = e.pinn_correction_magnitude;
            const vCorrMag = vHwMag * cMag;
            const vIdealMag = Math.max(e.ideal_magnitude, 1e-9);
            return 20 * Math.log10(Math.max(vCorrMag / vIdealMag, 1e-9));
          })
        : null;

      const traces: any[] = [
        {
          x: indices,
          y: impairedGain,
          type: 'bar',
          name: 'Hardware Impairment',
          marker: { color: 'rgba(220, 38, 38, 0.45)', line: { color: '#dc2626', width: 1 } },
          hovertemplate: 'Elem %{x}: %{y:.2f} dB<extra>Hardware</extra>',
        },
        {
          x: indices,
          y: anaGainResidual,
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Analytical Residual',
          line: { color: '#0891b2', width: 1.5 },
          marker: { size: 4, color: '#0891b2' },
          hovertemplate: 'Elem %{x}: %{y:.2f} dB<extra>Analytical</extra>',
        },
      ];

      if (pinnAvailable && pinnGainResidual) {
        traces.push({
          x: indices,
          y: pinnGainResidual,
          type: 'scatter',
          mode: 'lines+markers',
          name: 'PINN Residual',
          line: { color: '#7c3aed', width: 1.5 },
          marker: { size: 4, color: '#7c3aed' },
          hovertemplate: 'Elem %{x}: %{y:.2f} dB<extra>PINN</extra>',
        });
      }

      return traces;
    }
  }, [elements, metricTab, pinnAvailable]);

  const layout = useMemo(
    () => ({
      margin: { t: 20, r: 16, b: 45, l: 45 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { family: "'Inter', sans-serif", size: 11, color: '#475569' },
      legend: {
        orientation: 'h' as const,
        x: 0,
        y: 1.15,
        bgcolor: 'rgba(255,255,255,0.85)',
        font: { size: 10 },
      },
      xaxis: {
        title: { text: 'Array Element Index', font: { size: 11 } },
        gridcolor: '#e2e6ee',
        tickfont: { size: 10 },
      },
      yaxis: {
        title: {
          text: metricTab === 'phase' ? 'Phase Error (°)' : 'Gain Error (dB)',
          font: { size: 11 },
        },
        gridcolor: '#e2e6ee',
        tickfont: { size: 10 },
      },
      barmode: 'overlay' as const,
      hovermode: 'x unified' as const,
    }),
    [metricTab]
  );

  return (
    <div className="card">
      <div className="card-header">
        <h4>Element-Level Error Analysis</h4>
        <div style={{ display: 'flex', gap: 6 }}>
          <button
            className={`btn btn-secondary ${metricTab === 'phase' ? 'active' : ''}`}
            style={{
              padding: '2px 10px',
              fontSize: '0.6875rem',
              background: metricTab === 'phase' ? 'var(--color-primary)' : undefined,
              color: metricTab === 'phase' ? '#fff' : undefined,
            }}
            onClick={() => setMetricTab('phase')}
          >
            Phase Error
          </button>
          <button
            className={`btn btn-secondary ${metricTab === 'gain' ? 'active' : ''}`}
            style={{
              padding: '2px 10px',
              fontSize: '0.6875rem',
              background: metricTab === 'gain' ? 'var(--color-primary)' : undefined,
              color: metricTab === 'gain' ? '#fff' : undefined,
            }}
            onClick={() => setMetricTab('gain')}
          >
            Gain Error
          </button>
        </div>
      </div>
      <div className="card-body" style={{ padding: 0 }}>
        {!elements ? (
          <div className="awaiting-panel" style={{ padding: 32 }}>
            <span style={{ fontSize: '0.8125rem' }}>Awaiting simulation</span>
          </div>
        ) : (
          <div style={{ width: '100%', height: 260 }}>
            <Plot
              data={chartData}
              layout={layout}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
              config={{ displayModeBar: false, responsive: true }}
            />
          </div>
        )}
      </div>
    </div>
  );
};
