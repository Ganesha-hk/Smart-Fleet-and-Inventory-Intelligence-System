import React from 'react';
import { LayoutDashboard, Truck, BarChart3, Bell, Boxes, ChevronLeft, ChevronRight, Settings, HelpCircle, LogOut } from 'lucide-react';
import { motion } from 'framer-motion';
import useFleetStore from '../store/useFleetStore';
import { Link, useLocation } from 'react-router-dom';

const navItems = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard, path: '/' },
  { id: 'fleet', label: 'Fleet View', icon: Truck, path: '/fleet' },
  { id: 'analytics', label: 'Analytics', icon: BarChart3, path: '/analytics' },
  { id: 'notifications', label: 'Notifications', icon: Bell, path: '/notifications' },
  { id: 'inventory', label: 'Inventory', icon: Boxes, path: '/inventory' },
];
const MotionDiv = motion.div;
const MotionSpan = motion.span;

const Sidebar = () => {
  const { isSidebarCollapsed, toggleSidebar } = useFleetStore();
  const location = useLocation();

  return (
    <MotionDiv
      initial={false}
      animate={{ width: isSidebarCollapsed ? 80 : 260 }}
      className="h-screen bg-sidebar border-r border-gray-800 flex flex-col transition-all duration-300"
    >
      {/* Brand */}
      <div className="p-6 flex items-center gap-3">
        <div className="w-8 h-8 bg-accent-blue rounded-lg flex items-center justify-center">
          <Truck className="text-white w-5 h-5" />
        </div>
        {!isSidebarCollapsed && (
          <MotionSpan
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="font-bold text-lg tracking-tight"
          >
            Sentinel <span className="text-accent-blue">Fleet</span>
          </MotionSpan>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 space-y-2 mt-4">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.id}
              to={item.path}
              className={`flex items-center gap-3 p-3 rounded-lg transition-colors group ${
                isActive ? 'bg-accent-blue/10 text-accent-blue' : 'text-gray-400 hover:bg-gray-800 hover:text-white'
              }`}
            >
              <item.icon className="w-5 h-5" />
              {!isSidebarCollapsed && (
                <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  {item.label}
                </motion.span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-gray-800 space-y-2">
        <button onClick={toggleSidebar} className="w-full flex items-center gap-3 p-3 text-gray-400 hover:bg-gray-800 rounded-lg transition-colors">
          {isSidebarCollapsed ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
          {!isSidebarCollapsed && <span>Collapse Sidebar</span>}
        </button>
      </div>
    </MotionDiv>
  );
};

export default Sidebar;
