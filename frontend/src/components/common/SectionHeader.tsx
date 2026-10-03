import React from 'react';

export interface SectionHeaderProps {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}

export const SectionHeader: React.FC<SectionHeaderProps> = ({
  title,
  subtitle,
  badge,
  actions,
  className = '',
}) => {
  return (
    <div className={`section-header ${className}`}>
      <div className="section-header-main">
        <div className="section-header-title-row">
          <h1 className="section-header-title">{title}</h1>
          {badge && <div className="section-header-badge">{badge}</div>}
        </div>
        {subtitle && <p className="section-header-subtitle">{subtitle}</p>}
      </div>
      {actions && <div className="section-header-actions">{actions}</div>}
    </div>
  );
};
