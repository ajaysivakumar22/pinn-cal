import React from 'react';

export interface PageContainerProps {
  children: React.ReactNode;
  className?: string;
  maxWidth?: string | number;
  role?: string;
}

export const PageContainer: React.FC<PageContainerProps> = ({
  children,
  className = '',
  maxWidth,
  role = 'region',
}) => {
  return (
    <div
      className={`page-container ${className}`}
      style={maxWidth ? { maxWidth } : undefined}
      role={role}
    >
      {children}
    </div>
  );
};
