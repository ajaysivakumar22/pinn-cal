import React from 'react';

export interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  delta?: string | number;
  deltaPositive?: boolean;
  description?: string;
  status?: 'normal' | 'success' | 'warning' | 'accent' | 'danger';
  icon?: React.ReactNode;
  className?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  unit,
  delta,
  deltaPositive = true,
  description,
  status = 'normal',
  icon,
  className = '',
}) => {
  return (
    <div className={`metric-card metric-card-${status} ${className}`}>
      <div className="metric-card-header">
        <span className="metric-card-label">{label}</span>
        {icon && <span className="metric-card-icon" aria-hidden="true">{icon}</span>}
      </div>
      <div className="metric-card-value-row">
        <span className="metric-card-value">{value}</span>
        {unit && <span className="metric-card-unit">{unit}</span>}
      </div>
      {(delta !== undefined || description) && (
        <div className="metric-card-footer">
          {delta !== undefined && (
            <span
              className={`metric-card-delta ${deltaPositive ? 'positive' : 'negative'}`}
            >
              {delta}
            </span>
          )}
          {description && (
            <span className="metric-card-desc">{description}</span>
          )}
        </div>
      )}
    </div>
  );
};
