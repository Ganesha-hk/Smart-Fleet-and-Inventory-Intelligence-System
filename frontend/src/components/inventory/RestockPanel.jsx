import React from 'react';
import { motion } from 'framer-motion';
import { PackagePlus } from 'lucide-react';

const urgencyColor = {
  CRITICAL: 'text-accent-red',
  HIGH: 'text-accent-yellow',
  MEDIUM: 'text-accent-blue',
  LOW: 'text-green-400',
};
const MotionDiv = motion.div;

const RestockPanel = ({ recommendations = [], warehouses = [] }) => {
  return (
    <MotionDiv initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2 clean-card p-6">
        <div className="flex justify-between items-center mb-6">
          <h4 className="text-lg font-semibold flex items-center gap-2">
            <PackagePlus className="w-5 h-5 text-accent-blue" />
            Restock Recommendations
          </h4>
          <span className="text-xs text-gray-500 font-medium bg-gray-800 px-3 py-1 rounded-full">{recommendations.length} Active Suggestions</span>
        </div>
        
        <div className="max-h-[600px] overflow-y-auto pr-2 custom-scrollbar space-y-4">
          {recommendations.map((item) => (
            <div key={item.item_id} className="p-5 bg-gray-950/40 border border-gray-800 rounded-2xl group hover:border-gray-700 transition-all duration-300">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-wider ${
                      item.urgency === 'CRITICAL' ? 'bg-red-500/20 text-red-400' : 
                      item.urgency === 'HIGH' ? 'bg-yellow-500/20 text-yellow-400' : 'bg-blue-500/20 text-blue-400'
                    }`}>
                      {item.urgency}
                    </span>
                    <p className="text-xs text-gray-500 font-mono">{item.item_id}</p>
                  </div>
                  <h5 className="text-lg font-bold text-gray-100">{item.product_name}</h5>
                  <p className="text-sm text-gray-400">{item.warehouse_location}</p>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-x-8 gap-y-3 mt-4">
                    <div>
                      <p className="text-[10px] text-gray-600 uppercase font-bold tracking-tight">Current Stock</p>
                      <p className="text-sm font-medium text-gray-300">{item.stock_units.toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-gray-600 uppercase font-bold tracking-tight">Days Left</p>
                      <p className={`text-sm font-bold ${item.days_to_stockout < 2 ? 'text-red-400' : 'text-gray-300'}`}>
                        {item.days_to_stockout}d
                      </p>
                    </div>
                    <div>
                      <p className="text-[10px] text-gray-600 uppercase font-bold tracking-tight">Demand (24h)</p>
                      <p className="text-sm font-medium text-gray-300">{item.daily_demand.toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-gray-600 uppercase font-bold tracking-tight">ETA</p>
                      <p className="text-sm font-medium text-gray-300">{item.eta_days} days</p>
                    </div>
                  </div>
                  
                  <p className="text-xs text-gray-500 mt-4 leading-relaxed line-clamp-1 group-hover:line-clamp-none transition-all duration-300 italic border-l-2 border-gray-800 pl-3">
                    {item.rationale}
                  </p>
                </div>
                
                <div className="flex flex-col items-center justify-center p-4 bg-gray-900/40 rounded-xl min-w-[120px] border border-gray-800/50">
                  <p className="text-[10px] text-gray-500 uppercase font-bold mb-1">Restock</p>
                  <p className="text-3xl font-black text-accent-blue">{item.recommended_restock_quantity}</p>
                  <p className="text-[10px] text-gray-500 mt-1 uppercase tracking-tighter">Units Needed</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="clean-card p-6">
        <h4 className="text-lg font-semibold mb-6">Warehouse Snapshot</h4>
        <div className="max-h-[600px] overflow-y-auto pr-2 custom-scrollbar space-y-4">
          {warehouses.map((warehouse) => (
            <div key={warehouse.warehouse_location} className="p-5 bg-gray-950/40 border border-gray-800 rounded-xl hover:bg-gray-900/30 transition-colors">
              <div className="flex justify-between items-start mb-4">
                <h6 className="font-bold text-gray-200 text-sm">{warehouse.warehouse_location}</h6>
                <div className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                  warehouse.avg_stockout_probability > 0.5 ? 'bg-red-500/10 text-red-400' : 'bg-green-500/10 text-green-400'
                }`}>
                  {Math.round(warehouse.avg_stockout_probability * 100)}% Risk
                </div>
              </div>
              
              <div className="space-y-3">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-gray-500">Days to Stockout</span>
                  <span className={`font-bold ${warehouse.avg_days_to_stockout < 3 ? 'text-accent-yellow' : 'text-gray-300'}`}>
                    {warehouse.avg_days_to_stockout.toFixed(1)}d
                  </span>
                </div>
                <div className="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden">
                  <div 
                    className={`h-full rounded-full transition-all duration-1000 ${
                      warehouse.avg_days_to_stockout < 3 ? 'bg-accent-yellow' : 'bg-accent-blue'
                    }`} 
                    style={{ width: `${Math.min((warehouse.avg_days_to_stockout / 10) * 100, 100)}%` }}
                  />
                </div>
                
                <div className="grid grid-cols-2 gap-4 pt-2">
                  <div className="p-2 bg-gray-900/50 rounded-lg border border-gray-800/50">
                    <p className="text-[9px] text-gray-600 uppercase font-bold">Avg Stock</p>
                    <p className="text-xs font-bold text-gray-300">{Math.round(warehouse.avg_inventory_level).toLocaleString()}</p>
                  </div>
                  <div className="p-2 bg-gray-900/50 rounded-lg border border-gray-800/50">
                    <p className="text-[9px] text-gray-600 uppercase font-bold">Delay Link</p>
                    <p className="text-xs font-bold text-gray-300">{Math.round(warehouse.avg_linked_delay_risk * 100)}%</p>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </MotionDiv>
  );
};

export default RestockPanel;
