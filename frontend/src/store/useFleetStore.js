import { create } from 'zustand';
import { api } from '../services/api';

const MAP_BOUNDS = {
  lat: { min: 11.5, max: 25.40 },
  lng: { min: 55.0, max: 78.5 },
};
const MAX_ALERTS = 25;

const inRange = (value, min, max) => value >= min && value <= max;

const sanitizeVehicleCoordinates = (vehicle) => {
  let lat = Number(vehicle.lat);
  let lng = Number(vehicle.lng);

  if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
    return null;
  }

  if (lat > 50 && lng < 30) {
    [lat, lng] = [lng, lat];
  }

  if (
    !inRange(lat, MAP_BOUNDS.lat.min, MAP_BOUNDS.lat.max) ||
    !inRange(lng, MAP_BOUNDS.lng.min, MAP_BOUNDS.lng.max)
  ) {
    return null;
  }

  return {
    ...vehicle,
    id: vehicle.vehicle_id ?? vehicle.id,
    vehicle_id: vehicle.vehicle_id ?? vehicle.id,
    lat,
    lng,
    final_risk: Number(vehicle.final_risk ?? vehicle.delay_risk ?? 0),
    delayRisk: Number(vehicle.delayRisk ?? Number(vehicle.final_risk ?? vehicle.delay_risk ?? 0) * 100),
    delay_risk: Number(vehicle.final_risk ?? vehicle.delay_risk ?? Number(vehicle.delayRisk ?? 0) / 100),
    driverScore: Number(vehicle.driverScore ?? vehicle.driver_score ?? 0),
    driver_score: Number(vehicle.driver_score ?? vehicle.driverScore ?? 0),
    risk_level: vehicle.risk_level ?? 'SAFE',
    alert_active: Boolean(vehicle.alert_active),
    reasons: Array.isArray(vehicle.reasons) ? vehicle.reasons : [],
  };
};

const sanitizeVehicles = (vehicles = []) =>
  vehicles.map(sanitizeVehicleCoordinates).filter(Boolean);

const getTotalCount = (response, fallback = 0) =>
  response?.totalCount ?? response?.total_count ?? response?.total ?? fallback;

const getVehicleId = (vehicle) => vehicle?.vehicle_id ?? vehicle?.id;
const getEventId = (event) => event?.id ?? `${event?.vehicle_id ?? event?.vehicleId}:${event?.timestamp}`;

const mergeVehiclesById = (currentVehicles = [], incomingVehicles = []) => {
  const merged = new Map(currentVehicles.map((vehicle) => [getVehicleId(vehicle), vehicle]));
  incomingVehicles.forEach((vehicle) => {
    const vehicleId = getVehicleId(vehicle);
    if (!vehicleId) {
      return;
    }
    merged.set(vehicleId, {
      ...merged.get(vehicleId),
      ...vehicle,
    });
  });
  return Array.from(merged.values()).sort((left, right) => getVehicleId(left).localeCompare(getVehicleId(right)));
};

const syncSelectedVehicle = (selectedVehicle, vehicles) => {
  if (!selectedVehicle) {
    return null;
  }
  const selectedVehicleId = getVehicleId(selectedVehicle);
  return vehicles.find((vehicle) => getVehicleId(vehicle) === selectedVehicleId) ?? null;
};

const normalizeEvent = (event) => {
  const vehicleId = event?.vehicle_id ?? event?.vehicleId;
  const timestamp = event?.timestamp ?? new Date().toISOString();
  if (!vehicleId) {
    return null;
  }
  const riskLevel = String(event?.risk_level ?? event?.riskLevel ?? 'MID').toUpperCase();
  let severity = 'info';
  if (riskLevel === 'CRITICAL') {
    severity = 'critical';
  } else if (riskLevel === 'HIGH' || event?.type === 'risk_change' || event?.type === 'fatigue_spike') {
    severity = 'warning';
  }
  return {
    ...event,
    id: getEventId({ ...event, vehicle_id: vehicleId, timestamp }),
    vehicle_id: vehicleId,
    vehicleId,
    timestamp,
    type: severity,
    event_type: event?.type ?? severity,
    title: event?.title ?? `Vehicle ${vehicleId} update`,
    message: event?.message ?? `Vehicle ${vehicleId} changed state`,
    riskLevel: riskLevel,
    risk_level: riskLevel,
    delayRisk: Number((event?.final_risk ?? 0) * 100),
    delay_risk: Number(event?.final_risk ?? 0),
    reasons: Array.isArray(event?.reasons) ? event.reasons : [],
    time: new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
  };
};

const mergeEvents = (currentEvents = [], incomingEvents = []) => {
  const merged = new Map(currentEvents.map((event) => [getEventId(event), event]));
  incomingEvents.map(normalizeEvent).filter(Boolean).forEach((event) => {
    merged.set(getEventId(event), event);
  });
  return Array.from(merged.values())
    .sort((left, right) => new Date(right.timestamp).getTime() - new Date(left.timestamp).getTime())
    .slice(0, MAX_ALERTS);
};

const useFleetStore = create((set, get) => ({
  vehicles: [],
  selectedVehicle: null,
  isSidebarCollapsed: false,
  isNotificationsOpen: false,
  isLoading: false,
  error: null,
  pollingId: null,
  interpolationId: null,
  interpolationPhase: 0,
  previousAlertRiskByVehicle: {},
  dismissedAlertRiskByVehicle: {},
  metrics: {
    delayRisk: 0,
    activeAnomalies: 0,
    activeVehicles: 0,
    avgTimeToDelay: '0h',
    fleetPerformance: 0,
    mitigationConfidence: 0,
    failuresAverted: 0,
    featureImportance: [],
    delayTrend: [],
    riskDistribution: [],
    survivalCurve: []
  },
  alerts: [],

  setSelectedVehicle: (vehicle) => set({ selectedVehicle: vehicle }),
  toggleSidebar: () => set((state) => ({ isSidebarCollapsed: !state.isSidebarCollapsed })),
  toggleNotifications: () => set((state) => ({ isNotificationsOpen: !state.isNotificationsOpen })),
  clearAlerts: () => set((state) => ({
    alerts: [],
    dismissedAlertRiskByVehicle: {
      ...state.dismissedAlertRiskByVehicle,
      ...Object.fromEntries(state.alerts.map((alert) => [alert.vehicle_id, alert.delay_risk])),
    },
  })),
  removeAlert: (alertId) => set((state) => {
    const removedAlert = state.alerts.find((alert) => alert.id === alertId);
    return {
      alerts: state.alerts.filter((alert) => alert.id !== alertId),
      dismissedAlertRiskByVehicle: removedAlert ? {
        ...state.dismissedAlertRiskByVehicle,
        [removedAlert.vehicle_id]: removedAlert.delay_risk,
      } : state.dismissedAlertRiskByVehicle,
    };
  }),
  quickActionAlert: (alertId) => set((state) => {
    const resolvedAlert = state.alerts.find((alert) => alert.id === alertId);
    return {
      alerts: state.alerts.filter((alert) => alert.id !== alertId),
      dismissedAlertRiskByVehicle: resolvedAlert ? {
        ...state.dismissedAlertRiskByVehicle,
        [resolvedAlert.vehicle_id]: resolvedAlert.delay_risk,
      } : state.dismissedAlertRiskByVehicle,
    };
  }),

  applyFleetResponse: (vehicleResponse, metrics, isLoading = false) => set((state) => {
    const incomingVehicles = sanitizeVehicles(vehicleResponse?.vehicles ?? []);
    const vehicles = mergeVehiclesById(state.vehicles, incomingVehicles);
    const alerts = mergeEvents(state.alerts, vehicleResponse?.events ?? []);

    return {
      vehicles,
      selectedVehicle: syncSelectedVehicle(state.selectedVehicle, vehicles),
      alerts,
      metrics: {
        ...state.metrics,
        ...metrics,
        activeVehicles: getTotalCount(vehicleResponse, state.metrics.activeVehicles),
      },
      interpolationPhase: 0,
      isLoading,
      error: null,
    };
  }),

  fetchInitialData: async () => {
    set({ isLoading: true, error: null });
    try {
      const vehicleResponse = await api.fetchVehicles();
      const metrics = await api.fetchDashboardMetrics();
      get().applyFleetResponse(vehicleResponse, metrics, false);
    } catch (error) {
      set({ error: error.message, isLoading: false });
    }
  },

  refreshData: async () => {
    try {
      const vehicleResponse = await api.fetchVehicles();
      const metrics = await api.fetchDashboardMetrics();
      get().applyFleetResponse(vehicleResponse, metrics, false);
    } catch (error) {
      console.error('Failed to refresh data:', error);
    }
  },

  startRealtimeUpdates: () => {
    if (get().pollingId || get().interpolationId) {
      return;
    }
    const pollingId = setInterval(() => {
      get().refreshData();
    }, 2500);
    const interpolationId = setInterval(() => {
      set((state) => ({
        interpolationPhase: Math.min(state.interpolationPhase + 0.18, 0.92),
      }));
    }, 500);
    set({ pollingId, interpolationId });
  },

  stopRealtimeUpdates: () => {
    const { pollingId, interpolationId } = get();
    if (pollingId) {
      clearInterval(pollingId);
    }
    if (interpolationId) {
      clearInterval(interpolationId);
    }
    set({ pollingId: null, interpolationId: null });
  },
}));

export default useFleetStore;
