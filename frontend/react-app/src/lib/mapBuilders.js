const THEME_COLORS = {
  dark: {
    bgPrimary: "#0f172a",
    bgPanel: "#111827",
    bgCard: "#1e293b",
    bgSoft: "#0b1220",
    textPrimary: "#f9fafb",
    textSecondary: "#cbd5e1",
    textMuted: "#94a3b8",
    border: "#334155",
    accent: "#22d3ee",
    red: "#ef4444",
    orange: "#f97316",
    yellow: "#fbbf24",
    purple: "#a855f7",
    green: "#22c55e",
    blue: "#3b82f6",
    mapStyle: "open-street-map",
  },
  light: {
    bgPrimary: "#f8fafc",
    bgPanel: "#ffffff",
    bgCard: "#eef2ff",
    bgSoft: "#e2e8f0",
    textPrimary: "#111827",
    textSecondary: "#334155",
    textMuted: "#475569",
    border: "#cbd5e1",
    accent: "#0284c7",
    red: "#dc2626",
    orange: "#ea580c",
    yellow: "#d97706",
    purple: "#7e22ce",
    green: "#16a34a",
    blue: "#2563eb",
    mapStyle: "open-street-map",
  },
};

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const hexToRgba = (hex, alpha = 1) => {
  const normalized = hex.replace("#", "");
  const num = Number.parseInt(normalized, 16);
  const r = (num >> 16) & 255;
  const g = (num >> 8) & 255;
  const b = num & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

export function getThemeColors(theme) {
  return THEME_COLORS[theme] || THEME_COLORS.dark;
}

function buildHeatmapTraces(data) {
  const dzProbs = data?.dz_probs || [];
  const dzLats = data?.dz_lats || [];
  const dzLons = data?.dz_lons || [];

  const filteredLats = [];
  const filteredLons = [];
  const filteredProbs = [];

  for (let i = 0; i < dzProbs.length; i += 1) {
    const lat = dzLats[i];
    const lon = dzLons[i];
    const prob = dzProbs[i];

    if (Number.isFinite(lat) && Number.isFinite(lon) && Number.isFinite(prob)) {
      filteredLats.push(lat);
      filteredLons.push(lon);
      filteredProbs.push(clamp(prob, 0, 1));
    }
  }

  if (!filteredProbs.length) {
    return [];
  }

  return [
    {
      type: "densitymapbox",
      lat: filteredLats,
      lon: filteredLons,
      z: filteredProbs,
      radius: 58,
      opacity: 0.98,
      zmin: 0.08,
      zmax: 1.0,
      colorscale: [
        [0.0, "rgba(254, 243, 199, 0.0)"],
        [0.14, "rgba(251, 191, 36, 0.75)"],
        [0.34, "rgba(249, 115, 22, 0.9)"],
        [0.52, "rgba(239, 68, 68, 0.96)"],
        [1.0, "rgba(185, 28, 28, 1)"],
      ],
      name: "Hypoxia Heatmap",
      showscale: true,
      colorbar: {
        thickness: 15,
        len: 0.72,
        x: 1.02,
        tickvals: [0.08, 0.3, 0.52, 1.0],
        ticktext: ["Watch", "Warning", "Dead Zone", "Extreme"],
        title: { text: "Hypoxia Severity", side: "right" },
      },
      hovertemplate: "<b>Hypoxia Region</b><br>P(hypoxia): %{z:.1%}<extra></extra>",
    },
  ];
}

function buildSelectedProbabilityTrace(selectedProbabilityPoint, colors) {
  const lat = Number(selectedProbabilityPoint?.lat);
  const lon = Number(selectedProbabilityPoint?.lon);
  const prob = Number(selectedProbabilityPoint?.probability);

  if (!Number.isFinite(lat) || !Number.isFinite(lon) || !Number.isFinite(prob)) {
    return [];
  }

  const clippedProb = clamp(prob, 0, 1);
  const label = `P(hypoxia): ${(clippedProb * 100).toFixed(1)}%`;

  return [
    {
      type: "scattermapbox",
      lat: [lat],
      lon: [lon],
      mode: "markers+text",
      marker: {
        size: 11,
        color: colors.red,
        opacity: 0.95,
      },
      text: [label],
      textposition: "top right",
      textfont: {
        size: 12,
        color: colors.textPrimary,
      },
      showlegend: false,
      name: "Clicked Probability",
      hovertemplate: `<b>Clicked Location</b><br>${label}<extra></extra>`,
    },
  ];
}

function buildGhostGearTraces(data, colors) {
  const paths = data?.drift_paths || [];
  const traces = [];

  paths.forEach((path) => {
    const variants = [
      { latOffset: 0, lonOffset: 0, width: 2.6, opacity: 0.92 },
      { latOffset: 0.12, lonOffset: -0.09, width: 1.8, opacity: 0.72 },
      { latOffset: -0.11, lonOffset: 0.08, width: 1.8, opacity: 0.68 },
    ];

    variants.forEach((variant, variantIndex) => {
      const latSeries = path.lats.map(
        (lat, idx) => lat + variant.latOffset + Math.sin(idx * 0.5 + variantIndex) * 0.03,
      );
      const lonSeries = path.lons.map(
        (lon, idx) => lon + variant.lonOffset + Math.cos(idx * 0.45 + variantIndex) * 0.03,
      );

      traces.push({
        type: "scattermapbox",
        lat: latSeries,
        lon: lonSeries,
        mode: "lines+markers",
        line: { color: colors.purple, width: variant.width },
        marker: { size: variantIndex === 0 ? 6 : 4, color: colors.purple, opacity: variant.opacity },
        opacity: variant.opacity,
        name: variantIndex === 0 ? `Ghost Gear Path ${path.id}` : `Ghost Gear Path ${path.id} (drift)` ,
        hovertemplate: `<b>Ghost Gear Path ${path.id}</b><br>Drift segment<extra></extra>`,
        showlegend: variantIndex === 0,
      });
    });
  });

  return traces;
}

function buildTrapTraces(data, colors) {
  const traps = data?.traps || [];
  return traps.map((trap) => ({
    type: "scattermapbox",
    lat: [trap.lat],
    lon: [trap.lon],
    mode: "markers",
    marker: { size: 18, color: colors.orange, opacity: 0.95 },
    name: "Biodiversity Trap",
    hovertemplate: `<b>Trap</b><br>P(hypoxia): ${(trap.p_hypoxia * 100).toFixed(0)}%<extra></extra>`,
  }));
}

function buildWindTraces(windData, colors) {
  if (!windData?.vectors?.length) {
    return [];
  }

  const traces = [];
  windData.vectors.forEach((vector) => {
    const lat = vector.lat;
    const lon = vector.lon;
    const u = vector.u * 2;
    const v = vector.v * 2;
    const mag = vector.magnitude;

    const arrowColor = mag > 0.25 ? colors.red : mag > 0.15 ? colors.orange : colors.blue;

    traces.push({
      type: "scattermapbox",
      lat: [lat, lat + v],
      lon: [lon, lon + u],
      mode: "lines",
      line: { color: arrowColor, width: 2.4 },
      showlegend: false,
      hovertemplate: `<b>Wind</b><br>Speed: ${mag.toFixed(2)} m/s<extra></extra>`,
      name: "Wind Vector",
    });

    traces.push({
      type: "scattermapbox",
      lat: [lat + v],
      lon: [lon + u],
      mode: "markers",
      marker: { size: 8, color: arrowColor, symbol: "circle" },
      showlegend: false,
      hovertemplate: `<b>Wind Direction</b><br>Speed: ${mag.toFixed(2)} m/s<extra></extra>`,
      name: "Wind Direction",
    });
  });

  return traces;
}

function buildCurrentTraces(currentsData, colors) {
  if (!currentsData?.vectors?.length) {
    return [];
  }

  const traces = [];
  currentsData.vectors.forEach((vector) => {
    const lat = vector.lat;
    const lon = vector.lon;
    const u = vector.u * 1.5;
    const v = vector.v * 1.5;

    traces.push({
      type: "scattermapbox",
      lat: [lat, lat + v],
      lon: [lon, lon + u],
      mode: "lines",
      line: { color: colors.purple, width: 2.2 },
      showlegend: false,
      hovertemplate: `<b>Current</b><br>Speed: ${vector.magnitude.toFixed(2)} cm/s<extra></extra>`,
      name: "Ocean Current",
    });

    traces.push({
      type: "scattermapbox",
      lat: [lat + v],
      lon: [lon + u],
      mode: "markers",
      marker: { size: 7, color: colors.purple, symbol: "circle" },
      showlegend: false,
      hovertemplate: `<b>Current Direction</b><br>Speed: ${vector.magnitude.toFixed(2)} cm/s<extra></extra>`,
      name: "Current Direction",
    });
  });

  return traces;
}

export function buildMainMapFigure({ layer, data, theme, windData, currentsData, selectedProbabilityPoint }) {
  const colors = getThemeColors(theme);
  const traces = [];

  if (layer === "all" || layer === "dz") {
    traces.push(...buildHeatmapTraces(data));
  }

  if (layer === "all" || layer === "gg") {
    traces.push(...buildGhostGearTraces(data, colors));
  }

  if (layer === "all") {
    traces.push(...buildTrapTraces(data, colors));
  }

  if (windData) {
    traces.push(...buildWindTraces(windData, colors));
  }

  if (currentsData) {
    traces.push(...buildCurrentTraces(currentsData, colors));
  }

  traces.push(...buildSelectedProbabilityTrace(selectedProbabilityPoint, colors));

  // Ensure map canvas initializes even when no visible traces pass filters.
  traces.unshift({
    type: "scattermapbox",
    lat: [13.5],
    lon: [75],
    mode: "markers",
    marker: { size: 1, color: "rgba(0,0,0,0)" },
    showlegend: false,
    hoverinfo: "skip",
    name: "Map Anchor",
  });

  const layout = {
    mapbox: {
      style: colors.mapStyle,
      center: { lat: 13.5, lon: 75 },
      zoom: 4.5,
      bearing: 0,
      pitch: 0,
    },
    uirevision: "constant",
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    showlegend: true,
    dragmode: "pan",
    hovermode: "closest",
    clickmode: "event+select",
    legend: {
      bgcolor: hexToRgba(colors.bgPanel, 0.86),
      bordercolor: colors.border,
      borderwidth: 1,
      font: { color: colors.textPrimary, size: 11 },
      x: 0.01,
      y: 0.99,
      yanchor: "top",
    },
    autosize: true,
  };

  return {
    data: traces,
    layout,
    config: {
      displayModeBar: false,
      responsive: true,
      scrollZoom: true,
      doubleClick: "reset",
    },
  };
}

function parsePathCoordinates(path) {
  const lats = [];
  const lons = [];

  (path || []).forEach((point) => {
    if (Array.isArray(point) && point.length >= 2) {
      lats.push(Number(point[0]));
      lons.push(Number(point[1]));
    } else if (point && typeof point === "object") {
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

export function buildFlowPathwaysFigure({ pathwaysData, theme }) {
  const colors = getThemeColors(theme);
  const traces = [];

  const links = pathwaysData?.links || [];

  links.forEach((pathway, idx) => {
    const { lats, lons } = parsePathCoordinates(pathway.plume_path);
    if (lats.length < 2 || lons.length < 2) {
      return;
    }

    const correlation = clamp(Number(pathway.correlation_strength ?? 0.5), 0, 1);
    const riverName = pathway.river_name || `River ${idx + 1}`;
    const targetZone = pathway.downstream_zone || "Dead Zone";

    let color = "#fbbf24";
    let markerColor = "#f59e0b";
    if (correlation > 0.7) {
      color = "#dc2626";
      markerColor = "#991b1b";
    } else if (correlation > 0.4) {
      color = "#f97316";
      markerColor = "#ea580c";
    }

    const opacity = 0.4 + correlation * 0.6;
    const lineWidth = 1 + correlation * 3;

    traces.push({
      type: "scattermapbox",
      lat: lats,
      lon: lons,
      mode: "lines",
      line: { color, width: lineWidth },
      opacity,
      showlegend: true,
      name: `Transport: ${riverName} -> ${targetZone}`,
      hovertemplate:
        `<b>Nutrient Transport Path</b><br>` +
        `River: ${riverName}<br>` +
        `Target Zone: ${targetZone}<br>` +
        `Correlation Strength: ${(correlation * 100).toFixed(0)}%<br>` +
        `<i>Nutrient runoff contributing to hypoxia</i><extra></extra>`,
    });

    if (lats.length > 3) {
      const markerStep = Math.max(1, Math.floor(lats.length / 4));
      const markerLats = [];
      const markerLons = [];
      for (let i = markerStep; i < lats.length - 1; i += markerStep) {
        markerLats.push(lats[i]);
        markerLons.push(lons[i]);
      }

      if (markerLats.length) {
        traces.push({
          type: "scattermapbox",
          lat: markerLats,
          lon: markerLons,
          mode: "markers",
          marker: { size: 6, color: markerColor, opacity: 0.74, symbol: "circle" },
          showlegend: false,
          name: "Flow Direction",
          hovertemplate: `<b>Flow checkpoint</b><br>${riverName} transport<extra></extra>`,
        });
      }
    }

    traces.push({
      type: "scattermapbox",
      lat: [lats[0]],
      lon: [lons[0]],
      mode: "markers",
      marker: { size: 14, color: "#15803d", opacity: 0.9, symbol: "circle" },
      showlegend: true,
      name: `River Source: ${riverName}`,
      hovertemplate: `<b>Nutrient Source</b><br>River: ${riverName}<extra></extra>`,
    });

    traces.push({
      type: "scattermapbox",
      lat: [lats[lats.length - 1]],
      lon: [lons[lons.length - 1]],
      mode: "markers+text",
      marker: { size: 20, color: "#dc2626", opacity: 0.95, symbol: "circle" },
      text: ["!"],
      textposition: "middle center",
      textfont: { size: 12 },
      showlegend: true,
      name: `Dead Zone: ${targetZone}`,
      hovertemplate:
        `<b>Hypoxic Dead Zone</b><br>` +
        `Zone: ${targetZone}<br>` +
        `Nutrient Impact: ${(correlation * 100).toFixed(0)}%<extra></extra>`,
    });
  });

  const noData = traces.length === 0;

  return {
    data: traces,
    layout: {
      mapbox: {
        style: colors.mapStyle,
        center: { lat: 15, lon: 80 },
        zoom: 4,
      },
      margin: { l: 0, r: 0, t: 0, b: 0 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      hovermode: "closest",
      dragmode: "pan",
      uirevision: "constant",
      showlegend: true,
      legend: {
        bgcolor: hexToRgba(colors.bgPanel, 0.86),
        font: { color: colors.textPrimary, size: 9 },
        bordercolor: colors.border,
        borderwidth: 1,
        x: 0.01,
        y: 0.95,
        yanchor: "top",
      },
      annotations: noData
        ? [
            {
              text: "No runoff-hypoxia pathways available",
              showarrow: false,
              xref: "paper",
              yref: "paper",
              x: 0.5,
              y: 0.5,
              font: { color: colors.textMuted, size: 14 },
            },
          ]
        : [],
      autosize: true,
    },
    config: {
      displayModeBar: true,
      responsive: true,
      scrollZoom: true,
    },
  };
}
