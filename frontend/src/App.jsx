import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import FleetView from './pages/FleetView';
import Analytics from './pages/Analytics';
import Notifications from './pages/Notifications';
import Inventory from './pages/Inventory';
import useFleetStore from './store/useFleetStore';

function App() {
  const { fetchInitialData, startRealtimeUpdates, stopRealtimeUpdates } = useFleetStore();

  useEffect(() => {
    fetchInitialData();
    startRealtimeUpdates();

    return () => stopRealtimeUpdates();
  }, [fetchInitialData, startRealtimeUpdates, stopRealtimeUpdates]);

  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/fleet" element={<FleetView />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/notifications" element={<Notifications />} />
          <Route path="/inventory" element={<Inventory />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
