import React from 'react';

export type StatusVariant =
  | 'online'
  | 'offline'
  | 'warning'
  | 'info'
  | 'primary'
  | 'accent'
  | 'default';

export interface StatusChipProps {
  label: string;
  value?: React.ReactNode;
  status?: StatusVariant;
  icon?: React.ReactNode;
  dot?: boolean;
  title?: string;
  className?: string;
  onClick?: () => void;
}

export const StatusChip: React.FC<StatusChipProps> = ({
  label,
  value,
  status = 'default',
  icon,
  dot = true,
  title,
  className = '',
  onClick,
}) => {
  const Component = onClick ? 'button' : 'div';

  return (
    <Component
      className={`status-chip status-chip-${status} ${onClick ? 'status-chip-interactive' : ''} ${className}`}
      title={title}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
    >
      {dot && <span className={`status-chip-dot status-chip-dot-${status}`} aria-hidden="true" />}
      {icon && <span className="status-chip-icon" aria-hidden="true">{icon}</span>}
      <span className="status-chip-label">{label}</span>
      {value !== undefined && value !== null && (
        <span className="status-chip-value">{value}</span>
      )}
    </Component>
  );
};
