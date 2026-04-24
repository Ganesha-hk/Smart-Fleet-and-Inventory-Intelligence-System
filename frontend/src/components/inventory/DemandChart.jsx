import React from 'react';
import { motion } from 'framer-motion';
import { LineChart as LineChartIcon } from 'lucide-react';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

const MotionDiv = motion.div;

const DemandChart = ({ data = [] }) => {
  return (
    <MotionDiv initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="clean-card p-6">
      <h4 className="text-lg font-semibold mb-6 flex items-center gap-2">
        <LineChartIcon className="w-5 h-5 text-accent-blue" />
        Inventory Flow
      </h4>
      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
            <XAxis dataKey="name" stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
            <YAxis stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333', borderRadius: '8px' }}
              itemStyle={{ color: '#3b82f6' }}
            />
            <Line type="monotone" dataKey="demand" stroke="#eab308" strokeWidth={2} dot={{ fill: '#eab308', r: 3 }} />
            <Line type="monotone" dataKey="predicted" stroke="#3b82f6" strokeWidth={3} dot={{ fill: '#3b82f6', r: 4 }} />
            <Line type="monotone" dataKey="days_to_stockout" stroke="#ef4444" strokeWidth={2} dot={{ fill: '#ef4444', r: 2 }} />
            <Line type="monotone" dataKey="karnataka_stock" stroke="#22c55e" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="dubai_stock" stroke="#a855f7" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </MotionDiv>
  );
};

export default DemandChart;
