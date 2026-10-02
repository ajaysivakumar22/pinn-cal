import React, { Component, type ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallbackTitle?: string;
  onRetry?: () => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ChartErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('[ChartErrorBoundary] Rendering failure caught:', error, errorInfo);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
    if (this.props.onRetry) {
      this.props.onRetry();
    }
  };

  render() {
    if (this.state.hasError) {
      return (
        <div
          className="chart-error-fallback"
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '32px 24px',
            background: '#fef2f2',
            border: '1px solid #fecaca',
            borderRadius: '8px',
            color: '#991b1b',
            textAlign: 'center',
            minHeight: '220px',
            gap: '12px',
          }}
          role="alert"
        >
          <div style={{ fontSize: '1.75rem' }}>⚠️</div>
          <div style={{ fontWeight: 700, fontSize: '0.9375rem' }}>
            {this.props.fallbackTitle || 'Visualization failed to render.'}
          </div>
          <p style={{ fontSize: '0.8125rem', color: '#b91c1c', maxWidth: '420px', margin: 0 }}>
            {this.state.error?.message || 'An unexpected rendering error occurred in the chart subsystem.'}
          </p>
          <button
            onClick={this.handleRetry}
            style={{
              marginTop: '8px',
              padding: '8px 18px',
              background: '#dc2626',
              color: '#ffffff',
              border: 'none',
              borderRadius: '6px',
              fontWeight: 600,
              fontSize: '0.8125rem',
              cursor: 'pointer',
              boxShadow: '0 2px 4px rgba(220, 38, 38, 0.2)',
            }}
          >
            Retry Rendering
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ChartErrorBoundary;
