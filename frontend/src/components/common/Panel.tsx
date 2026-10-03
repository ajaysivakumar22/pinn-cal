import React from 'react';

export interface PanelProps {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
  footer?: React.ReactNode;
  elevated?: boolean;
  className?: string;
  id?: string;
}

export const Panel: React.FC<PanelProps> = ({
  title,
  subtitle,
  badge,
  actions,
  children,
  footer,
  elevated = false,
  className = '',
  id,
}) => {
  const hasHeader = Boolean(title || subtitle || badge || actions);

  return (
    <div
      id={id}
      className={`app-panel ${elevated ? 'app-panel-elevated' : ''} ${className}`}
    >
      {hasHeader && (
        <div className="app-panel-header">
          <div className="app-panel-title-group">
            {title && <h3 className="app-panel-title">{title}</h3>}
            {badge && <div className="app-panel-badge">{badge}</div>}
            {subtitle && <p className="app-panel-subtitle">{subtitle}</p>}
          </div>
          {actions && <div className="app-panel-actions">{actions}</div>}
        </div>
      )}
      <div className="app-panel-body">{children}</div>
      {footer && <div className="app-panel-footer">{footer}</div>}
    </div>
  );
};
