const API_BASE = import.meta.env.VITE_API_BASE || "";

function asFloatPathPart(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    throw new Error("Invalid coordinate value");
  }
  // Keep decimal representation so Flask <float:...> routes match consistently.
  return numeric.toFixed(4);
}

async function requestJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

export async function fetchFlowPathways() {
  return requestJson("/api/runoff-to-hypoxia-pathways");
}

export async function fetchWindVectors() {
  return requestJson("/api/wind-vectors");
}

export async function fetchOceanCurrents() {
  return requestJson("/api/ocean-currents");
}

export async function fetchPrecursorConditions(lat, lon) {
  return requestJson(`/api/precursor-conditions/${asFloatPathPart(lat)}/${asFloatPathPart(lon)}`);
}

export async function fetchInterventionMeasures(lat, lon) {
  return requestJson(`/api/intervention-measures/${asFloatPathPart(lat)}/${asFloatPathPart(lon)}`);
}

export async function fetchHealth() {
  return requestJson("/api/health");
}

export async function fetchFertilizerRunoff() {
  return requestJson("/api/fertilizer-runoff");
}

export async function fetchDeadZoneMarkers() {
  return requestJson("/api/dead-zone-markers");
}

export async function fetchDzProbabilityField() {
  return requestJson("/api/dz-probability-field?ocean_only=true&min_prob=0.02");
}
