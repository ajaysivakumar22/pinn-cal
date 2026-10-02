import React, { lazy, Suspense } from 'react';
import type { PlotParams } from 'react-plotly.js';
import { ChartErrorBoundary } from './ErrorBoundary';

// Lazy-load react-plotly.js so Plotly chunk is decoupled from critical initial bundle
const LazyPlotComponent = lazy(() => import('react-plotly.js'));

export const AsyncPlot: React.FC<PlotParams & { fallbackHeight?: string | number }> = ({
  fallbackHeight = 320,
  ...props
}) => {
  return (
    <ChartErrorBoundary fallbackTitle="Visualization failed to render.">
      <Suspense
        fallback={
          <div
            style={{
              height: typeof fallbackHeight === 'number' ? `${fallbackHeight}px` : fallbackHeight,
              width: '100%',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              background: '#f8fafc',
              border: '1px dashed #cbd5e1',
              borderRadius: '8px',
              color: '#64748b',
              fontSize: '0.8125rem',
              gap: '8px',
            }}
            aria-live="polite"
          >
            <div style={{ animation: 'spin 1s linear infinite', fontSize: '1.25rem' }}>⚡</div>
            <span>Loading scientific visualization engine...</span>
          </div>
        }
      >
        <LazyPlotComponent {...props} />
      </Suspense>
    </ChartErrorBoundary>
  );
};

export default AsyncPlot;
