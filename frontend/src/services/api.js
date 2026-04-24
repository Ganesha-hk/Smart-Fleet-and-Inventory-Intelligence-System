const BASE_URL = 'http://127.0.0.1:8000/api/v1';

export const api = {
  fetchVehicles: async () => {
    const response = await fetch(`${BASE_URL}/predict/sample-batch`);
    if (!response.ok) throw new Error('Failed to fetch vehicles');
    return response.json();
  },

  fetchDashboardMetrics: async () => {
    const response = await fetch(`${BASE_URL}/predict/dashboard-metrics`);
    if (!response.ok) throw new Error('Failed to fetch dashboard metrics');
    return response.json();
  },

  fetchPrediction: async (data) => {
    const response = await fetch(`${BASE_URL}/predict/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!response.ok) throw new Error('Failed to fetch prediction');
    return response.json();
  },
  
  fetchSamplePredict: async () => {
    const response = await fetch(`${BASE_URL}/predict/sample-predict`);
    if (!response.ok) throw new Error('Failed to fetch sample prediction');
    return response.json();
  },

  fetchInventorySummary: async () => {
    const response = await fetch(`${BASE_URL}/inventory/summary`);
    if (!response.ok) throw new Error('Failed to fetch inventory summary');
    return response.json();
  },

  predictInventoryDemand: async (data) => {
    const response = await fetch(`${BASE_URL}/inventory/predict-demand`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!response.ok) throw new Error('Failed to fetch inventory demand prediction');
    return response.json();
  },

  fetchStockoutRisk: async (data) => {
    const response = await fetch(`${BASE_URL}/inventory/stockout-risk`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!response.ok) throw new Error('Failed to fetch stockout risk');
    return response.json();
  },

  fetchRestockRecommendation: async (data) => {
    const response = await fetch(`${BASE_URL}/inventory/restock-recommendation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!response.ok) throw new Error('Failed to fetch restock recommendation');
    return response.json();
  }
};
