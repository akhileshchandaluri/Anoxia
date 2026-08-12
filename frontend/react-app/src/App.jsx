import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FlowPathwaysPanel } from "./components/features/FlowPathwaysPanel";
import { MainMap } from "./components/features/MainMap";
import { SidebarPanel } from "./components/features/SidebarPanel";
import { TopBar } from "./components/layout/TopBar";
import { FALLBACK_DATA } from "./data/mockData";
import { buildFlowPathwaysFigure, buildMainMapFigure } from "./lib/mapBuilders";
import {
  fetchDeadZoneMarkers,
  fetchDzProbabilityField,
  fetchFertilizerRunoff,
  fetchFlowPathways,
  fetchHealth,
  fetchInterventionMeasures,
  fetchOceanCurrents,
  fetchPrecursorConditions,
  fetchWindVectors,
} from "./services/api";

const DATA_REFRESH_MS = 300000;

function initialTheme() {
  const stored = localStorage.getItem("anoxia-theme");
  if (stored === "light" || stored === "dark") {
    return stored;
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function findZoneByName(data, zoneName) {
  return (data?.zones || []).find((item) => item.name === zoneName);
}

function formatCoordinate(value, positiveSuffix, negativeSuffix) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "--";
  }

  const suffix = numeric >= 0 ? positiveSuffix : negativeSuffix;
  return `${Math.abs(numeric).toFixed(2)}${suffix}`;
}

function formatCoordinateLabel(lat, lon) {
  return `${formatCoordinate(lat, "N", "S")}, ${formatCoordinate(lon, "E", "W")}`;
}

function isOceanWaterPoint(lat, lon) {
  const inArabianSea = lat >= 6 && lat <= 24 && lon >= 57 && lon <= 74.8;
  const inBayOfBengal = lat >= 6 && lat <= 24 && lon >= 80.3 && lon <= 96.5;

  if (!inArabianSea && !inBayOfBengal) {
    return false;
  }

  const onIndianPeninsula = lat >= 8 && lat <= 22 && lon >= 73.2 && lon <= 80.8;
  const onSriLanka = lat >= 5.5 && lat <= 10.5 && lon >= 79 && lon <= 82.2;
  const onBangladeshDelta = lat >= 20.5 && lat <= 23.8 && lon >= 88 && lon <= 92;

  return !(onIndianPeninsula || onSriLanka || onBangladeshDelta);
}

function expandDeadZoneToHeat(zone) {
  const centerLat = Number(zone.lat);
  const centerLon = Number(zone.lon);
  if (!Number.isFinite(centerLat) || !Number.isFinite(centerLon)) {
    return [];
  }

  const baseProb = Math.max(0.32, Math.min(1, Number(zone.hypoxia_probability || 0.3)));
  const radiusKm = Number(zone.radius_km || 50);
  const radiusDeg = Math.max(0.8, radiusKm / 66);
  const step = Math.max(0.09, radiusDeg / 12);

  const points = [];

  for (let lat = centerLat - radiusDeg; lat <= centerLat + radiusDeg; lat += step) {
    for (let lon = centerLon - radiusDeg; lon <= centerLon + radiusDeg; lon += step) {
      if (!isOceanWaterPoint(lat, lon)) {
        continue;
      }

      const dLat = lat - centerLat;
      const dLon = lon - centerLon;
      const distanceRatio = Math.sqrt(dLat * dLat + dLon * dLon) / radiusDeg;

      if (distanceRatio > 1) {
        continue;
      }

      const edgeDecay = 1 - distanceRatio * 0.12;
      const prob = Math.max(0.28, Math.min(1, baseProb * edgeDecay));

      points.push({
        lat: Number(lat.toFixed(4)),
        lon: Number(lon.toFixed(4)),
        prob,
      });
    }
  }

  if (!points.length && isOceanWaterPoint(centerLat, centerLon)) {
    points.push({ lat: centerLat, lon: centerLon, prob: baseProb });
  }

  return points;
}

function mapApiPrecursorsToBars(apiPayload) {
  const precursors = apiPayload?.precursors || {};
  const nitrate = Number(precursors.nitrate_anomaly || 0);
  const chlorophyll = Number(precursors.chlorophyll_modis || 0);

  return [
    ["Nitrate anomaly", `${nitrate >= 0 ? "+" : ""}${nitrate.toFixed(0)}%`, Math.min(100, Math.abs(nitrate))],
    ["Chlorophyll-a (MODIS)", `${chlorophyll >= 0 ? "+" : ""}${chlorophyll.toFixed(0)}%`, Math.min(100, Math.abs(chlorophyll))],
    [
      "Thermal stratification",
      String(precursors.thermal_stratification || "UNKNOWN"),
      Math.max(0, Math.min(100, Number(precursors.thermal_stratification_pct || 0))),
    ],
    [
      "Wind stress mixing",
      String(precursors.wind_stress || "UNKNOWN"),
      Math.max(0, Math.min(100, Number(precursors.wind_stress_pct || 0))),
    ],
    [
      "DO drawdown rate",
      String(precursors.do_drawdown || "UNKNOWN"),
      Math.max(0, Math.min(100, Number(precursors.do_drawdown_pct || 0))),
    ],
  ];
}

function normalizePlumePath(path = []) {
  const lats = [];
  const lons = [];

  path.forEach((point) => {
    if (Array.isArray(point) && point.length >= 2) {
      const lat = Number(point[0]);
      const lon = Number(point[1]);
      if (Number.isFinite(lat) && Number.isFinite(lon)) {
        lats.push(lat);
        lons.push(lon);
      }
      return;
    }

    if (point && typeof point === "object") {
      const lat = Number(point.lat);
      const lon = Number(point.lon);
      if (Number.isFinite(lat) && Number.isFinite(lon)) {
        lats.push(lat);
        lons.push(lon);
      }
    }
  });

  return { lats, lons };
}

function transformBackendMapData(fertilizerData, deadZoneData, dzFieldData, fallback) {
  const zones = deadZoneData?.zones || [];
  const correlations = fertilizerData?.correlations || [];

  const zoneHeatPoints = zones.flatMap(expandDeadZoneToHeat);

  const fieldLats = dzFieldData?.lats || [];
  const fieldLons = dzFieldData?.lons || [];
  const fieldProbs = dzFieldData?.probs || [];

  const dz_lats = zoneHeatPoints.map((point) => point.lat);
  const dz_lons = zoneHeatPoints.map((point) => point.lon);
  const dz_probs = zoneHeatPoints.map((point) => point.prob);

  const pointCount = Math.min(fieldLats.length, fieldLons.length, fieldProbs.length);
  for (let i = 0; i < pointCount; i += 1) {
    const lat = Number(fieldLats[i]);
    const lon = Number(fieldLons[i]);
    const prob = Number(fieldProbs[i]);

    if (!Number.isFinite(lat) || !Number.isFinite(lon) || !Number.isFinite(prob)) {
      continue;
    }

    if (!isOceanWaterPoint(lat, lon)) {
      continue;
    }

    dz_lats.push(lat);
    dz_lons.push(lon);
    dz_probs.push(Math.max(0, Math.min(1, prob)));
  }

  const drift_paths = correlations
    .map((link, idx) => {
      const coords = normalizePlumePath(link.plume_path || []);
      return {
        id: idx,
        lats: coords.lats,
        lons: coords.lons,
      };
    })
    .filter((path) => path.lats.length >= 2 && path.lons.length >= 2);

  const traps = zones
    .map((zone) => ({
      lat: Number(zone.lat),
      lon: Number(zone.lon),
      p_hypoxia: Math.max(0, Math.min(1, Number(zone.hypoxia_probability || 0))),
      severity: Number(zone.hypoxia_probability || 0) * 3,
      window_days: Math.max(3, Math.round(16 - Number(zone.hypoxia_probability || 0) * 10)),
    }))
    .filter((trap) => Number.isFinite(trap.lat) && Number.isFinite(trap.lon));

  const uiZones = zones.map((zone) => {
    const prob = Math.max(0, Math.min(1, Number(zone.hypoxia_probability || 0)));
    const status = prob >= 0.75 ? "CRITICAL" : prob >= 0.55 ? "WARNING" : "WATCH";
    const approxDo = Math.max(0.8, 4.2 - prob * 3.2);
    const days = Math.max(3, Math.round(16 - prob * 10));

    return {
      name: zone.name || "Unnamed Zone",
      lat: Number(zone.lat),
      lon: Number(zone.lon),
      do: Number(approxDo.toFixed(1)),
      p_hypoxia: prob,
      gear_paths: drift_paths.length,
      days,
      status,
    };
  });

  return {
    dz_lats: dz_lats.length ? dz_lats : fallback.dz_lats,
    dz_lons: dz_lons.length ? dz_lons : fallback.dz_lons,
    dz_probs: dz_probs.length ? dz_probs : fallback.dz_probs,
    drift_paths: drift_paths.length ? drift_paths : fallback.drift_paths,
    traps: traps.length ? traps : fallback.traps,
    zones: uiZones.length ? uiZones : fallback.zones,
  };
}

export default function App() {
  const [theme, setTheme] = useState(initialTheme);
  const [layer, setLayer] = useState("all");

  const [dashboardData, setDashboardData] = useState(FALLBACK_DATA);
  const [selectedZone, setSelectedZone] = useState(FALLBACK_DATA.zones[0]?.name || "DZ-A (Arabian Sea)");
  const [clickLocation, setClickLocation] = useState(null);

  const [flowPathwaysData, setFlowPathwaysData] = useState({ links: [] });
  const [flowLoading, setFlowLoading] = useState(false);
  const [flowError, setFlowError] = useState("");

  const [windEnabled, setWindEnabled] = useState(false);
  const [currentsEnabled, setCurrentsEnabled] = useState(false);
  const [windData, setWindData] = useState(null);
  const [currentsData, setCurrentsData] = useState(null);
  const [windLoading, setWindLoading] = useState(false);
  const [currentsLoading, setCurrentsLoading] = useState(false);

  const [healthStatus, setHealthStatus] = useState("unknown");
  const [overlayStatus, setOverlayStatus] = useState("");
  const conditionRequestRef = useRef(0);
  const [mapProbabilityOverlay, setMapProbabilityOverlay] = useState(null);
  const [precursorState, setPrecursorState] = useState({
    bars: [],
    interventions: [],
    locationLabel: "Loading backend baseline...",
    hypoxiaProbability: 0,
    loading: false,
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("anoxia-theme", theme);
  }, [theme]);

  useEffect(() => {
    let mounted = true;

    async function checkHealth() {
      try {
        await fetchHealth();
        if (mounted) {
          setHealthStatus("online");
        }
      } catch {
        if (mounted) {
          setHealthStatus("offline");
        }
      }
    }

    checkHealth();
    const timer = setInterval(checkHealth, DATA_REFRESH_MS);

    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    let mounted = true;

    async function loadMapDataFromBackend() {
      try {
        const [fertilizerData, deadZoneData, dzFieldData] = await Promise.all([
          fetchFertilizerRunoff(),
          fetchDeadZoneMarkers(),
          fetchDzProbabilityField(),
        ]);

        if (!mounted) {
          return;
        }

        const transformed = transformBackendMapData(fertilizerData, deadZoneData, dzFieldData, FALLBACK_DATA);
        setDashboardData(transformed);

        // Keep selected zone valid after backend data refresh.
        if (!transformed.zones.some((zone) => zone.name === selectedZone) && transformed.zones.length) {
          setSelectedZone(transformed.zones[0].name);
        }
      } catch {
        if (mounted) {
          setDashboardData(FALLBACK_DATA);
        }
      }
    }

    loadMapDataFromBackend();
    const timer = setInterval(loadMapDataFromBackend, DATA_REFRESH_MS);

    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, []);

  const loadFlowPathways = useCallback(async () => {
    setFlowLoading(true);
    setFlowError("");

    try {
      const response = await fetchFlowPathways();
      const links = response?.links || [];
      setFlowPathwaysData({ ...response, links });
    } catch {
      setFlowError("Unable to fetch runoff pathways from backend");
      setFlowPathwaysData({ links: [] });
    } finally {
      setFlowLoading(false);
    }
  }, []);

  useEffect(() => {
    loadFlowPathways();
    const timer = setInterval(loadFlowPathways, DATA_REFRESH_MS);
    return () => clearInterval(timer);
  }, [loadFlowPathways]);

  const loadBackendConditions = useCallback(async (lat, lon, locationLabel, options = {}) => {
    const { showOnMap = false } = options;
    const requestId = conditionRequestRef.current + 1;
    conditionRequestRef.current = requestId;

    if (showOnMap) {
      setMapProbabilityOverlay(null);
    }

    setPrecursorState((prev) => ({
      ...prev,
      loading: true,
      locationLabel,
    }));

    try {
      const [precursorData, interventionsData] = await Promise.all([
        fetchPrecursorConditions(lat, lon),
        fetchInterventionMeasures(lat, lon),
      ]);

      if (conditionRequestRef.current !== requestId) {
        return;
      }

      const probability = Math.max(
        0,
        Math.min(1, Number(precursorData?.probabilities?.hypoxia_30day ?? 0)),
      );

      setPrecursorState({
        bars: mapApiPrecursorsToBars(precursorData),
        interventions: interventionsData?.interventions || [],
        locationLabel,
        hypoxiaProbability: probability,
        loading: false,
      });

      if (showOnMap) {
        setMapProbabilityOverlay({ lat, lon, probability });
      }
    } catch {
      if (conditionRequestRef.current !== requestId) {
        return;
      }

      if (showOnMap) {
        setMapProbabilityOverlay(null);
      }

      setPrecursorState((prev) => ({
        ...prev,
        loading: false,
        locationLabel: `${locationLabel} (backend unavailable)`,
      }));
    }
  }, []);

  const updateFromZone = useCallback(
    async (zoneName, sourceData = dashboardData) => {
      const zone = findZoneByName(sourceData, zoneName);
      if (!zone) {
        return;
      }

      const lat = Number(zone.lat);
      const lon = Number(zone.lon);

      if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
        return;
      }

      setClickLocation(null);
      setMapProbabilityOverlay(null);
      await loadBackendConditions(lat, lon, `${zone.name} center`);
    },
    [dashboardData, loadBackendConditions],
  );

  const onSelectZone = useCallback(
    (zoneName) => {
      setClickLocation(null);
      setMapProbabilityOverlay(null);
      setSelectedZone(zoneName);
    },
    [],
  );

  useEffect(() => {
    if (!selectedZone || !dashboardData?.zones?.length) {
      return;
    }

    const selectedZoneData = findZoneByName(dashboardData, selectedZone);
    if (!selectedZoneData) {
      return;
    }

    const lat = Number(selectedZoneData.lat);
    const lon = Number(selectedZoneData.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      return;
    }

    void updateFromZone(selectedZone, dashboardData);
  }, [selectedZone, dashboardData, updateFromZone]);

  const handleCoordinateClick = useCallback(
    (rawLat, rawLon) => {
      const lat = Number(rawLat);
      const lon = Number(rawLon);

      if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
        return;
      }

      setClickLocation({ lat, lon });

      const label = formatCoordinateLabel(lat, lon);
      void loadBackendConditions(lat, lon, label, { showOnMap: true });
    },
    [loadBackendConditions],
  );

  const onToggleWind = useCallback(async () => {
    if (windEnabled) {
      setWindEnabled(false);
      setOverlayStatus("Wind overlay disabled");
      return;
    }

    setWindLoading(true);
    setOverlayStatus("Loading wind patterns from backend...");

    try {
      const data = await fetchWindVectors();
      setWindData(data);
      setWindEnabled(true);
      setOverlayStatus(`Wind patterns loaded (${data?.vectors?.length || 0} vectors)`);
    } catch {
      setOverlayStatus("Failed to load wind patterns from backend");
    } finally {
      setWindLoading(false);
    }
  }, [windEnabled]);

  const onToggleCurrents = useCallback(async () => {
    if (currentsEnabled) {
      setCurrentsEnabled(false);
      setOverlayStatus("Ocean currents overlay disabled");
      return;
    }

    setCurrentsLoading(true);
    setOverlayStatus("Loading ocean currents from backend...");

    try {
      const data = await fetchOceanCurrents();
      setCurrentsData(data);
      setCurrentsEnabled(true);
      setOverlayStatus(`Ocean currents loaded (${data?.vectors?.length || 0} vectors)`);
    } catch {
      setOverlayStatus("Failed to load ocean currents from backend");
    } finally {
      setCurrentsLoading(false);
    }
  }, [currentsEnabled]);

  const mapFigure = useMemo(
    () =>
      buildMainMapFigure({
        layer,
        data: dashboardData,
        theme,
        windData: windEnabled ? windData : null,
        currentsData: currentsEnabled ? currentsData : null,
        selectedProbabilityPoint: mapProbabilityOverlay,
      }),
    [
      layer,
      dashboardData,
      theme,
      windEnabled,
      windData,
      currentsEnabled,
      currentsData,
      mapProbabilityOverlay,
    ],
  );

  const flowFigure = useMemo(
    () => buildFlowPathwaysFigure({ pathwaysData: flowPathwaysData, theme }),
    [flowPathwaysData, theme],
  );

  return (
    <div className="min-h-screen bg-surface-bg text-text-primary antialiased">
      <div className="page-atmosphere" />

      <div className="relative z-10 mx-auto flex min-h-screen max-w-[1700px] flex-col">
        <TopBar
          theme={theme}
          layer={layer}
          windEnabled={windEnabled || windLoading}
          currentsEnabled={currentsEnabled || currentsLoading}
          onChangeLayer={setLayer}
          onToggleWind={onToggleWind}
          onToggleCurrents={onToggleCurrents}
          onToggleTheme={() => setTheme((prev) => (prev === "dark" ? "light" : "dark"))}
        />

        <div className="px-3 pb-3 pt-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-text-muted md:px-6">
          Backend status: {healthStatus}
          {clickLocation ? ` • Live point: ${clickLocation.lat.toFixed(2)}N ${clickLocation.lon.toFixed(2)}E` : ""}
          {overlayStatus ? ` • ${overlayStatus}` : ""}
        </div>

        <main className="flex flex-1 flex-col gap-3 px-3 pb-3 md:px-6 md:pb-6">
          <section className="flex min-h-[520px] flex-col gap-3 lg:h-[64vh] lg:min-h-0 lg:flex-row">
            <MainMap
              figure={mapFigure}
              onMapCoordinateClick={handleCoordinateClick}
            />

            <SidebarPanel
              data={dashboardData}
              selectedZone={selectedZone}
              onChangeZone={onSelectZone}
              precursorState={precursorState}
            />
          </section>

          <FlowPathwaysPanel figure={flowFigure} loading={flowLoading} error={flowError} />
        </main>
      </div>
    </div>
  );
}
