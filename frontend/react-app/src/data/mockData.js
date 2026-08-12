export const FALLBACK_DATA = {
  dz_lats: [
    14.5, 15.0, 15.5, 16.0, 14.0, 13.5, 13.0, 12.5, 12.0, 11.5, 11.0, 10.5, 10.0,
    9.5, 9.0, 8.5,
  ],
  dz_lons: [
    65.0, 65.5, 66.0, 66.5, 64.5, 64.0, 63.5, 63.0, 82.0, 82.5, 83.0, 83.5, 84.0,
    84.5, 85.0, 85.5,
  ],
  dz_probs: [
    0.83, 0.81, 0.79, 0.76, 0.80, 0.77, 0.74, 0.71, 0.72, 0.75, 0.78, 0.80, 0.82,
    0.79, 0.76, 0.73,
  ],
  drift_paths: [
    { id: 0, lats: [16.0, 15.8, 15.5, 15.1, 14.7, 14.4, 14.1, 13.8], lons: [72.0, 71.5, 71.0, 70.3, 69.5, 68.7, 67.8, 66.9] },
    { id: 1, lats: [17.2, 17.0, 16.7, 16.3, 15.9, 15.5, 15.1, 14.7], lons: [73.5, 73.0, 72.4, 71.7, 71.0, 70.2, 69.3, 68.4] },
    { id: 2, lats: [14.0, 13.8, 13.5, 13.2, 12.9, 12.6, 12.3, 12.0], lons: [80.5, 81.0, 81.5, 82.0, 82.5, 83.0, 83.4, 83.8] },
    { id: 3, lats: [15.5, 15.2, 14.9, 14.5, 14.1, 13.7, 13.3, 12.9], lons: [79.0, 79.5, 80.0, 80.6, 81.2, 81.8, 82.3, 82.7] },
  ],
  traps: [
    { lat: 14.1, lon: 66.9, p_hypoxia: 0.83, severity: 2.82, window_days: 8 },
    { lat: 12.6, lon: 83.0, p_hypoxia: 0.78, severity: 2.65, window_days: 8 },
  ],
  zones: [
    { name: "DZ-A (Arabian Sea)", do: 1.8, p_hypoxia: 0.83, gear_paths: 2, days: 6, status: "CRITICAL" },
    { name: "DZ-B (Bay of Bengal)", do: 2.1, p_hypoxia: 0.71, gear_paths: 2, days: 8, status: "CRITICAL" },
    { name: "Godavari Delta", do: 3.2, p_hypoxia: 0.48, gear_paths: 0, days: 14, status: "WARNING" },
  ],
};

export const PRECURSOR_DATA = {
  "DZ-A (Arabian Sea)": [
    ["Nitrate anomaly", "+280%", 85],
    ["Chlorophyl-a (MODIS)", "+310%", 95],
    ["Thermal stratification", "HIGH", 80],
    ["Wind stress mixing", "LOW", 25],
    ["DO drawdown rate", "FAST", 90],
  ],
  "DZ-B (Bay of Bengal)": [
    ["Nitrate anomaly", "+340%", 100],
    ["Chlorophyl-a (MODIS)", "+290%", 88],
    ["Thermal stratification", "MEDIUM", 55],
    ["Wind stress mixing", "LOW", 30],
    ["DO drawdown rate", "FAST", 85],
  ],
  "Godavari Delta": [
    ["Nitrate anomaly", "+190%", 55],
    ["Chlorophyl-a (MODIS)", "+140%", 42],
    ["Thermal stratification", "LOW", 35],
    ["Wind stress mixing", "MEDIUM", 50],
    ["DO drawdown rate", "SLOW", 25],
  ],
};
