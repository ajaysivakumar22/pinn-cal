import React from 'react';
import { Sidebar } from './Sidebar';

export interface SideNavigationProps {
  isOpenOnMobile?: boolean;
  onCloseMobile?: () => void;
  className?: string;
}

/**
 * SideNavigation Component — Delegates to Sidebar implementation.
 * Preserves existing import references across the application and test suites.
 */
export const SideNavigation: React.FC<SideNavigationProps> = (props) => {
  return <Sidebar {...props} />;
};

export default SideNavigation;
