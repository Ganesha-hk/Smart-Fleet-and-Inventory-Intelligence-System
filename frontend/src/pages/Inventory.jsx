import React, { useEffect, useState } from 'react';
import DemandChart from '../components/inventory/DemandChart';
import InventoryCard from '../components/inventory/InventoryCard';
import RestockPanel from '../components/inventory/RestockPanel';
import StockAlert from '../components/inventory/StockAlert';
import { api } from '../services/api';

const Inventory = () => {
  const [summary, setSummary] = useState({
    metrics: [],
    demand_trend: [],
    stock_alerts: [],
    restock_recommendations: [],
    warehouse_snapshots: [],
  });
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadSummary = async () => {
      try {
        const response = await api.fetchInventorySummary();
        setSummary(response);
      } catch (loadError) {
        setError(loadError.message);
      }
    };

    loadSummary();
    const timer = setInterval(loadSummary, 3000);

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="space-y-8 pb-10 max-w-[1600px] mx-auto px-4">
      <div className="flex justify-between items-end mb-2">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Inventory Intelligence</h2>
          <p className="text-gray-400 mt-2 max-w-2xl">
            Inventory forecasts synchronized with fleet delay signals and route risk analysis.
          </p>
        </div>
      </div>

      {error && (
        <div className="clean-card p-6 border-red-500/20 bg-red-500/5 text-red-400">
          Failed to load inventory intelligence: {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {summary.metrics.map((card, index) => (
          <InventoryCard key={card.title} card={card} index={index} />
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <DemandChart data={summary.demand_trend} />
        </div>
        <StockAlert alerts={summary.stock_alerts} />
      </div>

      <RestockPanel
        recommendations={summary.restock_recommendations}
        warehouses={summary.warehouse_snapshots}
      />
    </div>
  );
};

export default Inventory;
