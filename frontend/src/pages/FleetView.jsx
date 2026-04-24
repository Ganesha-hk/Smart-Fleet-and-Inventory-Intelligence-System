import React, { useEffect, useMemo, useRef, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { motion, AnimatePresence } from 'framer-motion';
import { ShieldCheck, Zap, Activity, Clock, Navigation } from 'lucide-react';
import useFleetStore from '../store/useFleetStore';

const GRID_DIVISIONS = 8;
const TARGET_COUNTS = {
  LOW: 140,
  MID: 120,
  HIGH: 60,
  CRITICAL: 80,
};
const MAX_RENDER = Object.values(TARGET_COUNTS).reduce((sum, count) => sum + count, 0);
const ROTATION_INTERVAL_MS = 4000;
const ROTATING_SHARE = {
  LOW: 0.18,
  MID: 0.14,
  HIGH: 0.0,
  CRITICAL: 0.0,
};
const ZOOM_RISK_VISIBILITY = {
  LOW: 4,
  MID: 0,
  HIGH: 0,
  CRITICAL: 0,
};

const MAP_BOUNDS = {
  lat: { min: 11.5, max: 25.40 },
  lng: { min: 55.0, max: 78.5 },
};

const inRange = (value, min, max) => value >= min && value <= max;
const MotionDiv = motion.div;

const normalizeRiskLevel = (vehicle) => {
  const riskLevel = String(vehicle.risk_level ?? 'LOW').toUpperCase();
  if (riskLevel === 'SAFE') return 'LOW';
  if (riskLevel === 'MEDIUM') return 'MID';
  return ['LOW', 'MID', 'HIGH', 'CRITICAL'].includes(riskLevel) ? riskLevel : 'LOW';
};

const getVehicleId = (vehicle) => String(vehicle.vehicle_id ?? vehicle.id ?? 'fleet');
const seededUnit = (seed) => (hashString(seed) % 1000003) / 1000003;

const getMarkerStyle = (riskBand, zoom) => {
  const zoomedOut = zoom < 5;
  const radiusScale = zoomedOut ? 0.92 : 1;
  if (riskBand === 'CRITICAL') {
    return { radius: 5 * radiusScale, fillColor: '#ef4444', fillOpacity: 0.75, strokeColor: '#fca5a5' };
  }
  if (riskBand === 'HIGH') {
    return { radius: 5 * radiusScale, fillColor: '#f6a91a', fillOpacity: zoomedOut ? 0.62 : 0.7, strokeColor: '#fdba74' };
  }
  if (riskBand === 'MID') {
    return { radius: 4.5 * radiusScale, fillColor: '#4d8ff7', fillOpacity: zoomedOut ? 0.58 : 0.65, strokeColor: '#93c5fd' };
  }
  return { radius: 4 * radiusScale, fillColor: '#34b76a', fillOpacity: zoomedOut ? 0.54 : 0.6, strokeColor: '#86efac' };
};

const hashString = (value) => {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = ((hash << 5) - hash) + value.charCodeAt(index);
    hash |= 0;
  }
  return Math.abs(hash);
};

const applyStableJitter = (vehicle, lat, lng) => {
  const vehicleId = getVehicleId(vehicle);
  const latSpread = (seededUnit(`${vehicleId}-spread-lat`) - 0.5) * 0.02;
  const lngSpread = (seededUnit(`${vehicleId}-spread-lng`) - 0.5) * 0.02;
  const latJitter = (seededUnit(`${vehicleId}-jitter-lat`) - 0.5) * 0.00035;
  const lngJitter = (seededUnit(`${vehicleId}-jitter-lng`) - 0.5) * 0.00035;

  return {
    lat: Math.min(MAP_BOUNDS.lat.max, Math.max(MAP_BOUNDS.lat.min, lat + latSpread + latJitter)),
    lng: Math.min(MAP_BOUNDS.lng.max, Math.max(MAP_BOUNDS.lng.min, lng + lngSpread + lngJitter)),
  };
};

const getGridKey = (vehicle) => {
  const lat = Number(vehicle.lat);
  const lng = Number(vehicle.lng);
  const latSpan = MAP_BOUNDS.lat.max - MAP_BOUNDS.lat.min;
  const lngSpan = MAP_BOUNDS.lng.max - MAP_BOUNDS.lng.min;
  const latIndex = Math.min(
    GRID_DIVISIONS - 1,
    Math.max(0, Math.floor(((lat - MAP_BOUNDS.lat.min) / latSpan) * GRID_DIVISIONS)),
  );
  const lngIndex = Math.min(
    GRID_DIVISIONS - 1,
    Math.max(0, Math.floor(((lng - MAP_BOUNDS.lng.min) / lngSpan) * GRID_DIVISIONS)),
  );
  return `${latIndex}-${lngIndex}`;
};

const getSeededOrder = (vehicles, seed, prioritiseHigherRisk = false) => [...vehicles].sort((left, right) => {
  const leftRisk = Number(left.final_risk ?? 0);
  const rightRisk = Number(right.final_risk ?? 0);
  if (prioritiseHigherRisk && Math.abs(rightRisk - leftRisk) > 0.015) {
    return rightRisk - leftRisk;
  }
  if (!prioritiseHigherRisk && Math.abs(leftRisk - rightRisk) > 0.015) {
    return leftRisk - rightRisk;
  }
  return seededUnit(`${seed}-${getVehicleId(left)}`) - seededUnit(`${seed}-${getVehicleId(right)}`);
});

const spatiallyBalancedSample = (vehicles, limit, seed, prioritiseHigherRisk = false, excludedIds = new Set()) => {
  const eligibleVehicles = vehicles.filter((vehicle) => !excludedIds.has(getVehicleId(vehicle)));
  if (eligibleVehicles.length <= limit) {
    return getSeededOrder(eligibleVehicles, seed, prioritiseHigherRisk);
  }

  const buckets = new Map();
  getSeededOrder(eligibleVehicles, seed, prioritiseHigherRisk).forEach((vehicle) => {
    const key = getGridKey(vehicle);
    if (!buckets.has(key)) {
      buckets.set(key, []);
    }
    buckets.get(key).push(vehicle);
  });

  const bucketKeys = Array.from(buckets.keys()).sort();
  const sampled = [];
  let bucketIndex = 0;

  while (sampled.length < limit && bucketKeys.some((key) => buckets.get(key)?.length > 0)) {
    const key = bucketKeys[bucketIndex % bucketKeys.length];
    const bucket = buckets.get(key);
    if (bucket?.length) {
      sampled.push(bucket.shift());
    }
    bucketIndex += 1;
  }

  return sampled;
};

const sampleBand = (vehicles, limit, riskBand, rotationTick) => {
  if (vehicles.length <= limit) {
    return spatiallyBalancedSample(vehicles, limit, `${riskBand}-stable`, riskBand !== 'LOW');
  }

  const rotatingCount = Math.min(Math.round(limit * (ROTATING_SHARE[riskBand] ?? 0)), Math.max(limit - 1, 0));
  const stableCount = Math.max(limit - rotatingCount, 0);
  const prioritiseHigherRisk = riskBand === 'HIGH' || riskBand === 'CRITICAL';
  const stableSelection = spatiallyBalancedSample(
    vehicles,
    stableCount,
    `${riskBand}-stable`,
    prioritiseHigherRisk,
  );
  if (rotatingCount === 0) {
    return stableSelection;
  }
  const stableIds = new Set(stableSelection.map((vehicle) => getVehicleId(vehicle)));
  const rotatingSelection = spatiallyBalancedSample(
    vehicles,
    rotatingCount,
    `${riskBand}-rotation-${rotationTick}`,
    prioritiseHigherRisk,
    stableIds,
  );
  return [...stableSelection, ...rotatingSelection];
};

const stratifiedSample = (vehicles, zoom, rotationTick) => {
  const grouped = {
    LOW: vehicles.filter((vehicle) => normalizeRiskLevel(vehicle) === 'LOW'),
    MID: vehicles.filter((vehicle) => normalizeRiskLevel(vehicle) === 'MID'),
    HIGH: vehicles.filter((vehicle) => normalizeRiskLevel(vehicle) === 'HIGH'),
    CRITICAL: vehicles.filter((vehicle) => normalizeRiskLevel(vehicle) === 'CRITICAL'),
  };

  return ['LOW', 'MID', 'HIGH', 'CRITICAL']
    .filter((riskBand) => zoom >= ZOOM_RISK_VISIBILITY[riskBand])
    .flatMap((riskBand) => sampleBand(grouped[riskBand], TARGET_COUNTS[riskBand], riskBand, rotationTick))
    .slice(0, MAX_RENDER);
};

const selectVisibleVehicles = (vehicles, zoom, rotationTick) => {
  const selected = stratifiedSample(vehicles, zoom, rotationTick);
  return {
    criticalRendered: selected.filter((vehicle) => normalizeRiskLevel(vehicle) === 'CRITICAL').length,
    visibleVehicles: selected,
  };
};

const smoothStep = (value) => value * value * (3 - (2 * value));

const normalizeVehiclePosition = (vehicle, interpolationPhase = 0) => {
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

  const latStep = Number(vehicle.lat_step ?? 0);
  const lngStep = Number(vehicle.lng_step ?? 0);
  const interpolationFactor = smoothStep(Math.max(0, Math.min(interpolationPhase, 0.95)));
  lat = Math.min(MAP_BOUNDS.lat.max, Math.max(MAP_BOUNDS.lat.min, lat + (latStep * interpolationFactor)));
  lng = Math.min(MAP_BOUNDS.lng.max, Math.max(MAP_BOUNDS.lng.min, lng + (lngStep * interpolationFactor)));
  const jittered = applyStableJitter(vehicle, lat, lng);

  return {
    ...vehicle,
    lat: jittered.lat,
    lng: jittered.lng,
  };
};

const MapInitializer = ({ vehicles }) => {
  const map = useMap();
  const initializedRef = useRef(false);

  useEffect(() => {
    map.setMaxBounds([
      [MAP_BOUNDS.lat.min, MAP_BOUNDS.lng.min],
      [MAP_BOUNDS.lat.max, MAP_BOUNDS.lng.max],
    ]);
  }, [map]);

  useEffect(() => {
    if (vehicles.length && !initializedRef.current) {
      const bounds = L.latLngBounds(vehicles.map(v => [v.lat, v.lng]));
      map.fitBounds(bounds, { padding: [50, 50] });
      initializedRef.current = true;
    }
  }, [map, vehicles]);

  return null;
};

const MapZoomTracker = ({ onZoomChange }) => {
  const map = useMapEvents({
    zoomend: () => {
      onZoomChange(map.getZoom());
    },
  });

  useEffect(() => {
    onZoomChange(map.getZoom());
  }, [map, onZoomChange]);

  return null;
};

const MapFocusController = ({ vehicle }) => {
  const map = useMap();

  useEffect(() => {
    if (!vehicle) {
      return;
    }
    map.flyTo([vehicle.lat, vehicle.lng], Math.max(map.getZoom(), 7), {
      duration: 0.8,
    });
  }, [map, vehicle]);

  return null;
};

const LEGEND_ITEMS = [
  { label: 'LOW', color: '#34b76a', opacity: 0.6 },
  { label: 'MID', color: '#4d8ff7', opacity: 0.65 },
  { label: 'HIGH', color: '#f6a91a', opacity: 0.7 },
  { label: 'CRITICAL', color: '#ef4444', opacity: 0.75 },
];

const formatSpeed = (vehicle) => `${Math.round(vehicle.speed)} ${vehicle.type === 'ship' ? 'knots' : 'km/h'}`;
const formatDistance = (vehicle) => {
  const distance = Number(vehicle.distance_to_destination ?? 0);
  return `${distance.toFixed(1)} ${vehicle.type === 'ship' ? 'nautical miles' : 'km'}`;
};

const FleetView = () => {
  const { vehicles, selectedVehicle, setSelectedVehicle, interpolationPhase } = useFleetStore();
  const [zoomLevel, setZoomLevel] = useState(4);
  const [rotationTick, setRotationTick] = useState(0);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      setRotationTick((current) => current + 1);
    }, ROTATION_INTERVAL_MS);
    return () => window.clearInterval(intervalId);
  }, []);

  const mapBounds = [
    [MAP_BOUNDS.lat.min, MAP_BOUNDS.lng.min],
    [MAP_BOUNDS.lat.max, MAP_BOUNDS.lng.max],
  ];
  const selection = useMemo(
    () => selectVisibleVehicles(vehicles, zoomLevel, rotationTick),
    [vehicles, zoomLevel, rotationTick],
  );
  const visibleVehicles = selection.visibleVehicles;
  const validVehicles = useMemo(
    () => visibleVehicles
      .map((vehicle) => normalizeVehiclePosition(vehicle, interpolationPhase))
      .filter(Boolean),
    [interpolationPhase, visibleVehicles],
  );
  const markerModels = useMemo(
    () => validVehicles.map((vehicle) => ({
      ...vehicle,
      visualizationBand: normalizeRiskLevel(vehicle),
      markerStyle: getMarkerStyle(normalizeRiskLevel(vehicle), zoomLevel),
      eventHandlers: {
        click: () => setSelectedVehicle(vehicle),
      },
    })),
    [setSelectedVehicle, validVehicles, zoomLevel],
  );
  const selectedVehiclePosition = selectedVehicle
    ? normalizeVehiclePosition(selectedVehicle, interpolationPhase)
    : null;

  return (
    <div className="h-[calc(100vh-128px)] flex gap-6 relative">
      {/* Map Content */}
      <div className="flex-1 clean-card overflow-hidden relative z-10">
        <div className="absolute top-4 left-4 z-[1000] bg-gray-950/80 border border-gray-800 rounded-lg px-3 py-2 text-xs text-gray-200 backdrop-blur-sm">
          <div className="font-semibold mb-2 tracking-wide text-gray-100">Risk</div>
          <div className="space-y-1.5">
            {LEGEND_ITEMS.map((item) => (
              <div key={item.label} className="flex items-center gap-2">
                <span
                  className="inline-block h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: item.color, opacity: item.opacity }}
                />
                <span>{item.label}</span>
              </div>
            ))}
          </div>
        </div>
        <MapContainer
          center={[20, 65]}
          zoom={4}
          preferCanvas={true}
          maxBounds={mapBounds}
          maxBoundsViscosity={1.0}
          className="h-full w-full"
        >
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          />
          <MapZoomTracker onZoomChange={setZoomLevel} />
          {markerModels.map((v) => (
            <React.Fragment key={v.vehicle_id}>
              {selectedVehiclePosition?.vehicle_id === v.vehicle_id && (
                <CircleMarker
                  center={[v.lat, v.lng]}
                  radius={v.markerStyle.radius + 4}
                  pathOptions={{
                    color: '#f8fafc',
                    weight: 1.2,
                    opacity: 0.8,
                    fillColor: '#f8fafc',
                    fillOpacity: 0.08,
                  }}
                />
              )}
              {v.visualizationBand === 'CRITICAL' && (
                <CircleMarker
                  center={[v.lat, v.lng]}
                  radius={(zoomLevel < 5 ? 5.8 : 6.6) + (interpolationPhase * 0.15)}
                  pathOptions={{
                    color: '#ef4444',
                    weight: 0.8,
                    opacity: Math.max(0.08, (zoomLevel < 5 ? 0.14 : 0.18) - (interpolationPhase * 0.04)),
                    fillColor: '#ef4444',
                    fillOpacity: Math.max(0.015, (zoomLevel < 5 ? 0.03 : 0.04) - (interpolationPhase * 0.008)),
                  }}
                />
              )}
              <CircleMarker
                center={[v.lat, v.lng]}
                radius={v.markerStyle.radius}
                pathOptions={{
                  color: v.markerStyle.strokeColor,
                  weight: v.visualizationBand === 'CRITICAL' ? 1.1 : 0.9,
                  fillColor: v.markerStyle.fillColor,
                  fillOpacity: v.markerStyle.fillOpacity,
                  opacity: v.markerStyle.fillOpacity,
                }}
                eventHandlers={v.eventHandlers}
              >
                <Popup className="dark-popup">
                  <div className="p-2">
                    <p className="font-bold text-gray-900">{v.vehicle_id}</p>
                    <p className="text-sm text-gray-600">Speed: {formatSpeed(v)}</p>
                    <p className="text-sm text-gray-600">{`Risk: ${Math.round((v.final_risk ?? 0) * 100)}% • ${v.risk_level}`}</p>
                  </div>
                </Popup>
              </CircleMarker>
            </React.Fragment>
          ))}
          <MapFocusController vehicle={selectedVehiclePosition} />
          <MapInitializer vehicles={validVehicles} />
        </MapContainer>
      </div>

      {/* Side Panel */}
      <AnimatePresence>
        {selectedVehiclePosition && (
          <MotionDiv
            initial={{ x: 400, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 400, opacity: 0 }}
            className="w-96 clean-card p-6 flex flex-col gap-6 z-20"
          >
            <div className="flex justify-between items-start">
              <div>
                <h3 className="text-xl font-bold">{selectedVehiclePosition.vehicle_id}</h3>
                <div className="flex items-center gap-2 mt-1">
                  <div className={`w-2 h-2 rounded-full ${
                    selectedVehiclePosition.status === 'normal' ? 'bg-green-500' : 
                    selectedVehiclePosition.status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
                  }`} />
                  <span className="text-sm text-gray-400 capitalize">{selectedVehiclePosition.status} Operations</span>
                </div>
              </div>
              <button onClick={() => setSelectedVehicle(null)} className="text-gray-500 hover:text-white">&times;</button>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-900/50 p-4 rounded-xl border border-gray-800">
                <p className="text-xs text-gray-500 uppercase font-bold tracking-wider">Delay Risk</p>
                <div className="flex items-center gap-2 mt-1 text-accent-blue">
                  <Activity className="w-4 h-4" />
                  <span className="text-xl font-bold">{selectedVehiclePosition.delayRisk}%</span>
                </div>
              </div>
              <div className="bg-gray-900/50 p-4 rounded-xl border border-gray-800">
                <p className="text-xs text-gray-500 uppercase font-bold tracking-wider">Driver Score</p>
                <div className="flex items-center gap-2 mt-1 text-green-400">
                  <ShieldCheck className="w-4 h-4" />
                  <span className="text-xl font-bold">{selectedVehiclePosition.driverScore}</span>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex py-3 border-b border-gray-800 justify-between items-center">
                <div className="flex items-center gap-2 text-gray-400">
                  <Navigation className="w-4 h-4" />
                  <span className="text-sm">Current Speed</span>
                </div>
                <span className="font-mono">{formatSpeed(selectedVehiclePosition)}</span>
              </div>
              <div className="flex py-3 border-b border-gray-800 justify-between items-center">
                <div className="flex items-center gap-2 text-gray-400">
                  <Zap className="w-4 h-4" />
                  <span className="text-sm">Battery / Fuel</span>
                </div>
                <span className="font-mono">{selectedVehiclePosition.battery}%</span>
              </div>
              <div className="flex py-3 border-b border-gray-800 justify-between items-center">
                <div className="flex items-center gap-2 text-gray-400">
                  <Clock className="w-4 h-4" />
                  <span className="text-sm">Time to Delay</span>
                </div>
                <span className="font-mono text-accent-blue">
                  {selectedVehiclePosition.estimated_time_to_delay}h
                </span>
              </div>
              <div className="flex py-3 border-b border-gray-800 justify-between items-center">
                <div className="flex items-center gap-2 text-gray-400">
                  <Clock className="w-4 h-4" />
                  <span className="text-sm">Distance</span>
                </div>
                <span className="font-mono">{formatDistance(selectedVehiclePosition)}</span>
              </div>
              {selectedVehiclePosition.type === 'ship' && (
                <>
                  <div className="flex py-3 border-b border-gray-800 justify-between items-center">
                    <span className="text-sm text-gray-400">Route Name</span>
                    <span className="font-mono">{selectedVehiclePosition.route_name}</span>
                  </div>
                  <div className="flex py-3 border-b border-gray-800 justify-between items-center">
                    <span className="text-sm text-gray-400">Convoy Size</span>
                    <span className="font-mono">{selectedVehiclePosition.convoy_size}</span>
                  </div>
                  <div className="flex py-3 border-b border-gray-800 justify-between items-center">
                    <span className="text-sm text-gray-400">Sea Condition</span>
                    <span className="font-mono capitalize">{selectedVehiclePosition.sea_condition}</span>
                  </div>
                </>
              )}
            </div>

            <div className="mt-auto space-y-3">
              <button disabled title="Coming soon" className="w-full py-3 bg-accent-blue text-white rounded-xl font-bold transition-all opacity-50 cursor-not-allowed">
                Contact Driver
              </button>
              <button disabled title="Coming soon" className="w-full py-3 bg-gray-800 text-gray-300 rounded-xl font-bold transition-all opacity-50 cursor-not-allowed">
                Reroute Request
              </button>
            </div>
          </MotionDiv>
        )}
      </AnimatePresence>
    </div>
  );
};

export default FleetView;
