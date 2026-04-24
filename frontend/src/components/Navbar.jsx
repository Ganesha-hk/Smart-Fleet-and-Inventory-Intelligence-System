import React, { useMemo, useState } from 'react';
import { Bell, Search, Zap } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import useFleetStore from '../store/useFleetStore';

const Navbar = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { isNotificationsOpen, toggleNotifications, alerts, clearAlerts, vehicles, setSelectedVehicle } = useFleetStore();
  const [query, setQuery] = useState('');

  const searchResults = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      return { vehicles: [], alerts: [], routes: [] };
    }
    
    const matchedVehicles = vehicles
      .filter((vehicle) => {
        const haystack = [
          vehicle.vehicle_id,
          vehicle.type,
          vehicle.region,
          vehicle.route_name,
        ].join(' ').toLowerCase();
        return haystack.includes(normalized);
      })
      .slice(0, 5);

    const matchedAlerts = alerts
      .filter((alert) => {
        const haystack = [
          alert.message,
          alert.vehicleId,
          alert.type,
        ].join(' ').toLowerCase();
        return haystack.includes(normalized);
      })
      .slice(0, 3);

    const matchedRoutes = Array.from(new Set(
      vehicles
        .map(v => v.route_name)
        .filter(name => name && name.toLowerCase().includes(normalized))
    )).slice(0, 3);

    return { vehicles: matchedVehicles, alerts: matchedAlerts, routes: matchedRoutes };
  }, [query, vehicles, alerts]);

  const handleSelectVehicle = (vehicle) => {
    setSelectedVehicle(vehicle);
    setQuery('');
    if (location.pathname !== '/fleet') {
      navigate('/fleet');
    }
  };

  const handleSelectRoute = (routeName) => {
    const firstVehicle = vehicles.find(v => v.route_name === routeName);
    if (firstVehicle) {
      handleSelectVehicle(firstVehicle);
    }
  };

  const handleSelectAlert = (alert) => {
    const vehicle = vehicles.find(v => v.vehicle_id === alert.vehicleId);
    if (vehicle) {
      handleSelectVehicle(vehicle);
    } else {
      navigate('/notifications');
      setQuery('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      if (searchResults.vehicles.length > 0) {
        handleSelectVehicle(searchResults.vehicles[0]);
      } else if (searchResults.routes.length > 0) {
        handleSelectRoute(searchResults.routes[0]);
      } else if (searchResults.alerts.length > 0) {
        handleSelectAlert(searchResults.alerts[0]);
      }
    }
  };

  const hasResults = searchResults.vehicles.length > 0 || searchResults.alerts.length > 0 || searchResults.routes.length > 0;

  return (
    <header className="h-16 border-b border-gray-800 bg-background/50 backdrop-blur-md sticky top-0 z-30 px-6 flex items-center justify-between">
      <div className="relative w-96">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
        <input 
          type="text" 
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Search vehicles, ships, or routes..."
          className="w-full bg-gray-900/50 border border-gray-800 rounded-full py-2 pl-10 pr-4 text-sm focus:outline-none focus:border-accent-blue transition-colors"
        />
        {hasResults && (
          <div className="absolute top-12 left-0 right-0 bg-gray-950/95 border border-gray-800 rounded-2xl overflow-hidden shadow-2xl backdrop-blur-xl">
            {searchResults.vehicles.length > 0 && (
              <div className="p-2">
                <div className="px-3 py-1 text-[10px] uppercase font-bold text-gray-500 tracking-wider">Vehicles</div>
                {searchResults.vehicles.map((vehicle) => (
                  <button
                    key={vehicle.vehicle_id}
                    onClick={() => handleSelectVehicle(vehicle)}
                    className="w-full px-3 py-2 text-left hover:bg-gray-800 rounded-lg transition-colors flex items-center gap-3"
                  >
                    <div className={`w-2 h-2 rounded-full ${
                      vehicle.status === 'critical' ? 'bg-red-500' : 
                      vehicle.status === 'warning' ? 'bg-yellow-500' : 'bg-green-500'
                    }`} />
                    <div>
                      <div className="text-sm font-medium text-gray-100">{vehicle.vehicle_id}</div>
                      <div className="text-[10px] text-gray-400 capitalize">
                        {vehicle.type} • {vehicle.region || vehicle.route_name} • {vehicle.risk_level}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}
            {searchResults.routes.length > 0 && (
              <div className="p-2 border-t border-gray-800">
                <div className="px-3 py-1 text-[10px] uppercase font-bold text-gray-500 tracking-wider">Routes</div>
                {searchResults.routes.map((routeName) => (
                  <button
                    key={routeName}
                    onClick={() => handleSelectRoute(routeName)}
                    className="w-full px-3 py-2 text-left hover:bg-gray-800 rounded-lg transition-colors flex items-center gap-3"
                  >
                    <div className="w-5 h-5 rounded bg-gray-800 flex items-center justify-center">
                      <Zap className="w-3 h-3 text-accent-blue" />
                    </div>
                    <div>
                      <div className="text-sm font-medium text-gray-100">{routeName}</div>
                      <div className="text-[10px] text-gray-400">Active Route</div>
                    </div>
                  </button>
                ))}
              </div>
            )}
            {searchResults.alerts.length > 0 && (
              <div className="p-2 border-t border-gray-800">
                <div className="px-3 py-1 text-[10px] uppercase font-bold text-gray-500 tracking-wider">Active Alerts</div>
                {searchResults.alerts.map((alert) => (
                  <button
                    key={alert.id}
                    onClick={() => handleSelectAlert(alert)}
                    className="w-full px-3 py-2 text-left hover:bg-gray-800 rounded-lg transition-colors flex items-start gap-3"
                  >
                    <div className={`mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 ${
                      alert.type === 'critical' ? 'bg-red-500' : 'bg-yellow-500'
                    }`} />
                    <div>
                      <div className="text-xs font-medium text-gray-200 line-clamp-1">{alert.message}</div>
                      <div className="text-[10px] text-gray-500">
                        Vehicle {alert.vehicleId} • {alert.time}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* System Status & Profile */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2 px-3 py-1 bg-green-500/10 border border-green-500/20 rounded-full">
          <Zap className="w-4 h-4 text-green-500 fill-green-500" />
          <span className="text-xs font-semibold text-green-500 uppercase tracking-wider">System Live</span>
        </div>
        
        <div className="flex items-center gap-4 border-l border-gray-800 ml-2 pl-6 relative">
          <button 
            onClick={toggleNotifications}
            className={`relative text-gray-400 hover:text-white transition-colors p-2 rounded-full hover:bg-gray-800 ${isNotificationsOpen ? 'text-white bg-gray-800' : ''}`}
          >
            <Bell className="w-5 h-5" />
            {alerts.length > 0 && (
              <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full border-2 border-background"></span>
            )}
          </button>

          <AnimatePresence>
            {isNotificationsOpen && (
              <motion.div
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 10, scale: 0.95 }}
                className="absolute right-0 top-14 w-80 clean-card overflow-hidden z-50 shadow-2xl"
              >
                <div className="p-4 border-b border-gray-800 flex justify-between items-center bg-gray-900/50">
                  <h3 className="font-bold">Notifications</h3>
                  <button onClick={clearAlerts} className="text-xs text-accent-blue hover:underline cursor-pointer">Mark all as read</button>
                </div>
                <div className="max-h-[400px] overflow-y-auto">
                  {alerts.map((alert) => (
                    <div key={alert.id} className="p-4 border-b border-gray-800 hover:bg-gray-800/50 transition-colors cursor-pointer group">
                      <div className="flex gap-3">
                        <div className={`mt-1 w-2 h-2 rounded-full shrink-0 ${
                          alert.type === 'critical' ? 'bg-red-500' : 
                          alert.type === 'warning' ? 'bg-yellow-500' : 'bg-blue-500'
                        }`} />
                        <div>
                          <p className="text-sm font-medium text-gray-200 group-hover:text-accent-blue transition-colors">
                            {alert.message}
                          </p>
                          <p className="text-xs text-gray-500 mt-1">{alert.time} • {alert.vehicleId}</p>
                          {alert.reasons?.[0] && (
                            <p className="text-xs text-gray-400 mt-1">{`→ ${alert.reasons[0]}`}</p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                  {alerts.length === 0 && (
                    <div className="p-4 text-sm text-gray-500">No active notifications.</div>
                  )}
                </div>
                <div className="p-3 text-center border-t border-gray-800">
                  <Link 
                    to="/notifications"
                    onClick={toggleNotifications}
                    className="text-xs text-gray-400 hover:text-white transition-colors cursor-pointer"
                  >
                    View all notifications
                  </Link>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
