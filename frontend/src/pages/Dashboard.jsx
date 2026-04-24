import React from 'react';
import { motion } from 'framer-motion';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { ShieldAlert, Truck, Timer, AlertTriangle, TrendingDown } from 'lucide-react';
import useFleetStore from '../store/useFleetStore';

const MotionDiv = motion.div;

const buildRiskDistribution = (vehicles = []) => {
  const counts = {
    LOW: 0,
    MID: 0,
    HIGH: 0,
    CRITICAL: 0,
  };

  vehicles.forEach((vehicle) => {
    const riskLevel = String(vehicle.risk_level ?? 'LOW').toUpperCase();
    if (counts[riskLevel] != null) {
      counts[riskLevel] += 1;
    }
  });

  return [
    { name: 'LOW', value: counts.LOW },
    { name: 'MID', value: counts.MID },
    { name: 'HIGH', value: counts.HIGH },
    { name: 'CRITICAL', value: counts.CRITICAL },
  ];
};

const RiskTooltip = ({ active, payload, label, total }) => {
  if (!active || !payload?.length) {
    return null;
  }
  const current = payload[0].payload;
  const percentage = total > 0 ? ((current.value / total) * 100).toFixed(1) : '0.0';
  return (
    <div style={{ backgroundColor: '#1a1a1a', border: '1px solid #333', borderRadius: '8px', padding: '8px 10px' }}>
      <p style={{ color: '#e5e7eb', fontSize: '12px', fontWeight: 600 }}>{label}</p>
      <p style={{ color: '#d1d5db', fontSize: '12px' }}>{`${current.value} vehicles (${percentage}%)`}</p>
    </div>
  );
};

const Dashboard = () => {
  const { metrics, alerts, vehicles } = useFleetStore();
  const riskDistribution = buildRiskDistribution(vehicles);
  const totalVehicles = metrics.activeVehicles || vehicles.length;

  const KPI_CARDS = [
    { id: 'delay-risk', label: 'Delay Risk %', value: `${metrics.delayRisk ?? 0}%`, change: '-2.4%', icon: Timer, color: 'text-accent-blue', bg: 'bg-accent-blue/10' },
    { id: 'anomalies', label: 'Active Anomalies', value: (metrics.activeAnomalies ?? 0).toString(), change: '+1', icon: AlertTriangle, color: 'text-accent-red', bg: 'bg-accent-red/10' },
    { id: 'vehicles', label: 'Active Vehicles', value: (metrics.activeVehicles ?? 0).toString(), change: 'Stable', icon: Truck, color: 'text-accent-yellow', bg: 'bg-accent-yellow/10' },
    { id: 'avg-delay', label: 'Avg Time to Delay', value: metrics.avgTimeToDelay ?? '0h', change: '-10m', icon: TrendingDown, color: 'text-green-500', bg: 'bg-green-500/10' },
  ];

  return (
    <div className="space-y-8 pb-10">
      {/* KPI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {KPI_CARDS.map((card, index) => (
          <MotionDiv
            key={card.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="clean-card p-6"
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="text-gray-400 text-sm font-medium">{card.label}</p>
                <h3 className="text-2xl font-bold mt-1">{card.value}</h3>
                <span className={`text-xs mt-1 inline-block ${card.change.startsWith('+') ? 'text-red-400' : card.change === 'Stable' ? 'text-gray-500' : 'text-green-400'}`}>
                  {card.change} <span className="text-gray-600">vs last hour</span>
                </span>
              </div>
              <div className={`p-3 rounded-xl ${card.bg}`}>
                <card.icon className={`w-6 h-6 ${card.color}`} />
              </div>
            </div>
          </MotionDiv>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 clean-card p-6">
          <h4 className="text-lg font-semibold mb-6 flex items-center gap-2">
            <LineChart className="w-5 h-5 text-accent-blue" />
            Delay Trend (24h)
          </h4>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metrics.delayTrend}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                <XAxis dataKey="name" stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333', borderRadius: '8px' }}
                  itemStyle={{ color: '#3b82f6' }}
                />
                <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={3} dot={{ fill: '#3b82f6', r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="clean-card p-6">
          <h4 className="text-lg font-semibold mb-6 flex items-center gap-2">
            <ShieldAlert className="w-5 h-5 text-accent-yellow" />
            Risk Distribution
          </h4>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={riskDistribution}>
                <XAxis dataKey="name" stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip content={<RiskTooltip total={totalVehicles} />} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {riskDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={['#10b981', '#3b82f6', '#f59e0b', '#ef4444'][index]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Alert Feed */}
      <div className="clean-card p-6">
        <h4 className="text-lg font-semibold mb-6">Real-time Intelligence Feed</h4>
        <div className="space-y-4">
          {alerts.map((alert) => (
            <div key={alert.id} className="flex items-center justify-between p-4 bg-gray-900/50 border border-gray-800 rounded-xl hover:border-gray-700 transition-all group">
              <div className="flex items-center gap-4">
                <div className={`w-2 h-12 rounded-full ${
                  alert.type === 'critical' ? 'bg-accent-red' : 
                  alert.type === 'warning' ? 'bg-accent-yellow' : 'bg-accent-blue'
                }`} />
                <div>
                  <p className="font-medium text-gray-200">{alert.message}</p>
                  <p className="text-xs text-gray-500 mt-1">Vehicle ID: <span className="text-gray-400 font-mono">{alert.vehicleId}</span> • {alert.time}</p>
                  {alert.reasons?.length > 0 && (
                    <p className="text-xs text-gray-400 mt-2">{alert.reasons.join(' • ')}</p>
                  )}
                  {alert.solution && (
                    <div className="mt-3 p-3 bg-accent-blue/5 border border-accent-blue/10 rounded-lg">
                      <p className="text-[10px] text-accent-blue font-black uppercase tracking-widest mb-1">Recommended Solution</p>
                      <p className="text-xs text-gray-300 font-medium">{alert.solution}</p>
                    </div>
                  )}
                </div>
              </div>
              <button disabled title="Coming soon" className="text-gray-500 transition-colors opacity-50 group-hover:opacity-50 px-4 py-2 bg-gray-800 rounded-lg text-sm cursor-not-allowed">
                Investigate
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
