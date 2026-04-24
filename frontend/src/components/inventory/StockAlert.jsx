import React from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle } from 'lucide-react';

const MotionDiv = motion.div;

const StockAlert = ({ alerts = [] }) => {
  return (
    <MotionDiv initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass-card p-6 border border-gray-800/50">
      <div className="flex justify-between items-center mb-6">
        <h4 className="text-lg font-semibold flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-accent-red" />
          Stock Alerts
        </h4>
        <span className="flex h-2 w-2 rounded-full bg-red-500 animate-pulse" />
      </div>
      
      <div className="max-h-[400px] overflow-y-auto pr-2 custom-scrollbar space-y-4">
        {alerts.map((alert) => (
          <div key={alert.item_id} className="p-4 bg-gray-950/40 border border-gray-800 rounded-xl hover:bg-gray-900/40 transition-colors">
            <div className="flex justify-between items-start mb-3">
              <div>
                <h5 className="font-bold text-gray-200">{alert.product_name}</h5>
                <p className="text-[10px] text-gray-500 uppercase tracking-widest">{alert.warehouse_location}</p>
              </div>
              <div className="text-right">
                <span className="text-[10px] font-black px-2 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20 uppercase tracking-widest">
                  {alert.risk}
                </span>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4 mt-4">
              <div className="space-y-1">
                <p className="text-[10px] text-gray-600 uppercase font-bold">Stock Probability</p>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-1 bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full bg-red-500" style={{ width: `${alert.stockout_probability * 100}%` }} />
                  </div>
                  <span className="text-xs font-bold text-gray-300">{Math.round(alert.stockout_probability * 100)}%</span>
                </div>
              </div>
              <div className="space-y-1">
                <p className="text-[10px] text-gray-600 uppercase font-bold">Days Remaining</p>
                <p className="text-sm font-bold text-red-400">{alert.days_to_stockout}d</p>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-gray-800/50 flex justify-between items-center">
              <p className="text-[10px] text-gray-500 font-medium">ETA: {alert.eta_days} days</p>
              <p className="text-[10px] text-accent-blue font-bold uppercase tracking-tighter">View Details →</p>
            </div>
          </div>
        ))}
      </div>
    </MotionDiv>
  );
};

export default StockAlert;
