import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { AreaChart, Area, PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Target, Users, Shield, Zap } from 'lucide-react';
import useFleetStore from '../store/useFleetStore';

const MotionDiv = motion.div;

const Analytics = () => {
  const { metrics, vehicles } = useFleetStore();

  const survivalData = metrics.survivalCurve || [];
  const riskSegmentation = useMemo(() => {
    const counts = {
      LOW: 0,
      MID: 0,
      HIGH: 0,
      CRITICAL: 0,
    };
    vehicles.forEach((vehicle) => {
      const riskLevel = vehicle.risk_level ?? 'LOW';
      if (counts[riskLevel] != null) {
        counts[riskLevel] += 1;
      }
    });
    return Object.entries(counts).map(([name, value]) => ({ name, value }));
  }, [vehicles]);

  const featureImportance = metrics.featureImportance || [];
  return (
    <div className="space-y-8 pb-10">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Predictive Analytics</h2>
        <div className="flex gap-3">
          <button disabled title="Coming soon" className="px-4 py-2 bg-gray-800 rounded-lg text-sm opacity-50 cursor-not-allowed transition-colors">Export Report</button>
          <button disabled title="Coming soon" className="px-4 py-2 bg-accent-blue rounded-lg text-sm font-bold opacity-50 cursor-not-allowed transition-colors">Re-train Models</button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Survival Curve */}
        <MotionDiv initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-card p-6">
          <div className="flex items-center gap-2 mb-6">
            <Zap className="w-5 h-5 text-accent-blue" />
            <h4 className="text-lg font-semibold">Survival Curve (Time-to-Delay)</h4>
          </div>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={survivalData}>
                <defs>
                  <linearGradient id="colorProb" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" stroke="#666" fontSize={12} label={{ value: 'Hours', position: 'insideBottom', offset: -5, fill: '#666' }} />
                <YAxis stroke="#666" fontSize={12} label={{ value: 'Prob %', angle: -90, position: 'insideLeft', fill: '#666' }} />
                <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }} />
                <Area type="monotone" dataKey="probability" stroke="#3b82f6" fillOpacity={1} fill="url(#colorProb)" strokeWidth={3} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </MotionDiv>

        {/* Feature Importance */}
        <MotionDiv initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass-card p-6">
          <div className="flex items-center gap-2 mb-6">
            <Target className="w-5 h-5 text-accent-yellow" />
            <h4 className="text-lg font-semibold">Model Feature Importance</h4>
          </div>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={featureImportance} layout="vertical">
                <XAxis type="number" hide />
                <YAxis dataKey="name" type="category" stroke="#666" fontSize={12} width={120} />
                <Tooltip cursor={{ fill: 'transparent' }} contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }} />
                <Bar dataKey="score" fill="#eab308" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </MotionDiv>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Risk Segmentation */}
        <MotionDiv initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="glass-card p-6">
          <h4 className="text-lg font-semibold mb-6 flex items-center gap-2">
            <Shield className="w-5 h-5 text-green-500" />
            Risk Segmentation
          </h4>
          <div className="h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
              <Pie data={riskSegmentation} innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value">
                {riskSegmentation.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={['#10b981', '#3b82f6', '#f59e0b', '#ef4444'][index % 4]} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 grid grid-cols-2 gap-2">
          {riskSegmentation.map((item, index) => (
            <div key={item.name} className="flex items-center gap-2 text-xs text-gray-400">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: ['#10b981', '#3b82f6', '#f59e0b', '#ef4444'][index % 4] }} />
              <span>{item.name}: {item.value} vehicles</span>
            </div>
          ))}
        </div>
        </MotionDiv>

        {/* Intelligence Cards */}
        <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="glass-card p-6 border-l-4 border-l-accent-blue">
            <div className="flex items-center gap-3 mb-4">
              <Users className="w-6 h-6 text-accent-blue" />
              <h5 className="font-bold">Fleet Performance</h5>
            </div>
            <p className="text-sm text-gray-400 mb-4">The overall fleet efficiency is continuously monitored and updated based on predictive route optimization.</p>
            <div className="flex justify-between items-end">
              <span className="text-3xl font-bold">{metrics.fleetPerformance || 0}</span>
              <span className="text-green-400 text-sm">Live</span>
            </div>
          </div>
          <div className="glass-card p-6 border-l-4 border-l-accent-yellow">
            <div className="flex items-center gap-3 mb-4">
              <Zap className="w-6 h-6 text-accent-yellow" />
              <h5 className="font-bold">Risk Mitigation</h5>
            </div>
            <p className="text-sm text-gray-400 mb-4">Anomaly detection models have successfully identified and averted {metrics.failuresAverted || 0} potential delivery failures.</p>
            <div className="flex justify-between items-end">
              <span className="text-3xl font-bold">{metrics.mitigationConfidence || 0}%</span>
              <span className="text-accent-yellow text-sm font-medium">Confidence</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
