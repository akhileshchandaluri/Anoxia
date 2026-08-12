#!/usr/bin/env python3
"""
ANOXIA Spatial Fields Preparation

PURPOSE: Create spatially gridded fields for fertilizer runoff visualization

OUTPUTS:
  - ./outputs/nitrate_anomaly.nc (NetCDF grid with spatial patterns)
  - ./outputs/runoff_sources.json (river mouth metadata)
  - ./outputs/runoff_to_hypoxia_links.json (plume-to-dead-zone correlations)

PART A: Create synthetic nitrate anomaly field
PART B: Identify major runoff sources (rivers)
PART C: Correlate runoff plumes with downstream dead zones
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path('./outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

# Spatial grid parameters
LAT_MIN, LAT_MAX = 0, 30        # 0-30°N
LON_MIN, LON_MAX = 55, 100      # 55-100°E
RESOLUTION = 0.5                # 0.5° grid

# River mouth locations with flow rates
RIVER_SOURCES = [
    {"name": "Mahanadi", "lat": 20.5, "lon": 86.5, "flow_rate": "HIGH", "basin_size": 141589},
    {"name": "Godavari", "lat": 16.7, "lon": 82.3, "flow_rate": "HIGH", "basin_size": 312812},
    {"name": "Krishna", "lat": 14.3, "lon": 79.8, "flow_rate": "MEDIUM", "basin_size": 258948},
    {"name": "Narmada", "lat": 21.8, "lon": 72.6, "flow_rate": "MEDIUM", "basin_size": 98796},
    {"name": "Indus", "lat": 24.8, "lon": 67.2, "flow_rate": "HIGH", "basin_size": 1165500},
]

# Monsoon current direction (NE to SW plume spreading)
MONSOON_DIRECTION = (-0.707, -0.707)  # South-West direction (normalized)

# ─────────────────────────────────────────────────────────────────
# PART A: CREATE SYNTHETIC NITRATE ANOMALY FIELD
# ─────────────────────────────────────────────────────────────────

def create_nitrate_grid():
    """
    Create synthetic nitrate anomaly field (60×90 grid at 0.5° resolution).
    
    Pattern:
    - Peak anomalies (+300% to +500%) near river mouths
    - Decreases with distance from coast
    - Monsoon seasonality (Aug-Oct peak)
    - Background noise
    
    Returns:
        tuple: (lats, lons, nitrate_anomaly_grid)
    """
    logger.info("Creating nitrate anomaly field...")
    
    # Create coordinate arrays
    lats = np.arange(LAT_MIN, LAT_MAX + RESOLUTION, RESOLUTION)
    lons = np.arange(LON_MIN, LON_MAX + RESOLUTION, RESOLUTION)
    
    logger.info(f"  Grid size: {len(lats)} × {len(lons)} = {len(lats) * len(lons)} cells")
    logger.info(f"  Lat range: {LAT_MIN}-{LAT_MAX}°N, Lon range: {LON_MIN}-{LON_MAX}°E")
    
    # Initialize grid with background nitrate anomaly
    nitrate_grid = np.ones((len(lats), len(lons))) * -20  # -20% baseline
    
    # Add river plume contributions
    for river in RIVER_SOURCES:
        logger.info(f"  Processing: {river['name']} ({river['lat']}°N, {river['lon']}°E)")
        
        river_plume = create_river_plume(
            lats, lons,
            river['lat'], river['lon'],
            river['flow_rate'],
            river['basin_size']
        )
        nitrate_grid += river_plume
    
    # Add spatial noise and coastal effects
    nitrate_grid += add_coastal_effects(lats, lons)
    nitrate_grid += np.random.normal(0, 5, nitrate_grid.shape)  # Random noise
    
    # Monsoon seasonality: Aug-Oct peak (scale by 1.3x)
    current_month = datetime.now().month
    monsoon_factor = 1.3 if 8 <= current_month <= 10 else 1.0
    nitrate_grid *= monsoon_factor
    
    # Clamp values to realistic range (-50% to +500%)
    nitrate_grid = np.clip(nitrate_grid, -50, 500)
    
    logger.info(f"  Nitrate range: {nitrate_grid.min():.1f}% to {nitrate_grid.max():.1f}%")
    
    return lats, lons, nitrate_grid


def create_river_plume(lats, lons, river_lat, river_lon, flow_rate, basin_size):
    """
    Create spatially spreading plume from river mouth.
    
    Args:
        lats, lons: Grid coordinates
        river_lat, river_lon: River mouth location
        flow_rate: "HIGH" or "MEDIUM"
        basin_size: Basin area for flow scaling
    
    Returns:
        2D array with plume contribution
    """
    plume = np.zeros((len(lats), len(lons)))
    
    # Peak anomaly based on flow rate and basin size
    if flow_rate == "HIGH":
        peak_anomaly = 300 + (basin_size / 1e6) * 50  # 300-400%
        decay_scale = 2.5
    else:
        peak_anomaly = 150 + (basin_size / 1e6) * 30  # 150-250%
        decay_scale = 2.0
    
    # Create 2D grids for distance calculation
    LAT_GRID, LON_GRID = np.meshgrid(lons, lats)
    
    # Distance from river mouth (in degrees, approximately km/111)
    distance = np.sqrt((LAT_GRID - river_lon)**2 + (lats[:, np.newaxis] - river_lat)**2)
    
    # Gaussian plume: spread from river outlet
    plume_spread = np.exp(-(distance**2) / (2 * decay_scale**2))
    
    # Monsoon plume direction: extend further downstream
    lat_offset = (lats[:, np.newaxis] - river_lat) * MONSOON_DIRECTION[0]
    lon_offset = (LON_GRID - river_lon) * MONSOON_DIRECTION[1]
    directional_factor = np.exp(-(lat_offset**2 + lon_offset**2) / (decay_scale**2))
    
    # Combine spreading and directional components
    plume = peak_anomaly * plume_spread * (0.5 + 0.5 * directional_factor)
    
    return plume


def add_coastal_effects(lats, lons):
    """
    Add coastal upwelling and nutrient concentration effects.
    
    Returns:
        2D array with coastal enhancement
    """
    coastal_effect = np.zeros((len(lats), len(lons)))
    
    # Indian coast at approx lon 67-90°E
    COAST_LON_MIN, COAST_LON_MAX = 67, 90
    COASTAL_WIDTH = 3.0  # degrees
    
    for i, lon in enumerate(lons):
        # Distance from coast
        dist_to_coast = min(
            abs(lon - COAST_LON_MIN),
            abs(lon - COAST_LON_MAX)
        )
        
        # Coastal enhancement: higher near shore
        if dist_to_coast < COASTAL_WIDTH:
            enhancement = 50 * (1 - dist_to_coast / COASTAL_WIDTH)
            coastal_effect[:, i] = enhancement
    
    return coastal_effect


def save_nitrate_netcdf(lats, lons, nitrate_grid):
    """Save nitrate field to NetCDF format."""
    try:
        import xarray as xr
        
        logger.info("Saving nitrate anomaly to NetCDF...")
        
        # Create xarray Dataset
        ds = xr.Dataset(
            {
                'nitrate_anomaly': (['lat', 'lon'], nitrate_grid),
            },
            coords={
                'lat': lats,
                'lon': lons,
            },
            attrs={
                'title': 'Nitrate Anomaly Field - ANOXIA Project',
                'description': 'Synthetic nitrate anomaly (%) from fertilizer runoff',
                'source': 'Calculated from river discharge and monsoon currents',
                'created': datetime.now().isoformat(),
                'units': 'percent anomaly (%)',
                'range': f'{nitrate_grid.min():.1f}% to {nitrate_grid.max():.1f}%',
            }
        )
        
        # Save to file
        output_file = OUTPUT_DIR / 'nitrate_anomaly.nc'
        ds.to_netcdf(output_file)
        logger.info(f"  Saved: {output_file}")
        
        return output_file
    
    except ImportError:
        logger.warning("xarray not available, skipping NetCDF output")
        logger.info("  Install xarray: pip install xarray netcdf4")
        return None


# ─────────────────────────────────────────────────────────────────
# PART B: IDENTIFY MAJOR RUNOFF SOURCES
# ─────────────────────────────────────────────────────────────────

def create_runoff_sources():
    """
    Create structured runoff source metadata with plume directions.
    
    Returns:
        dict: River metadata with calculated plume directions
    """
    logger.info("Creating runoff sources metadata...")
    
    runoff_data = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'source': 'ANOXIA Project - River Discharge Analysis',
            'monsoon_season': 'June-October',
            'region': 'Indian Ocean - Arabian Sea and Bay of Bengal',
        },
        'rivers': []
    }
    
    for river in RIVER_SOURCES:
        # Calculate plume direction (normalized)
        plume_dir_lat = MONSOON_DIRECTION[0]
        plume_dir_lon = MONSOON_DIRECTION[1]
        
        # Estimate plume end point (spreading distance ~200 km)
        plume_end_lat = river['lat'] + plume_dir_lat * 2.0
        plume_end_lon = river['lon'] + plume_dir_lon * 2.0
        
        river_data = {
            'name': river['name'],
            'lat': river['lat'],
            'lon': river['lon'],
            'flow_rate': river['flow_rate'],
            'basin_size_km2': river['basin_size'],
            'discharge_class': estimate_discharge(river['basin_size']),
            'plume_direction': {
                'north_component': plume_dir_lat,
                'east_component': plume_dir_lon,
                'description': 'Southwest (monsoon-driven)',
            },
            'plume_endpoint': {
                'lat': round(plume_end_lat, 2),
                'lon': round(plume_end_lon, 2),
                'description': 'Estimated plume center after 200 km spread',
            },
            'seasonal_pattern': {
                'monsoon_peak': 'August-October',
                'intensity_factor': 1.3,
                'dry_season_factor': 0.6,
            }
        }
        runoff_data['rivers'].append(river_data)
        logger.info(f"  {river['name']}: {river['flow_rate']} discharge, {river['basin_size']} km²")
    
    return runoff_data


def estimate_discharge(basin_size):
    """Estimate discharge class from basin size."""
    if basin_size > 300000:
        return "VERY HIGH"
    elif basin_size > 150000:
        return "HIGH"
    elif basin_size > 75000:
        return "MEDIUM"
    else:
        return "LOW"


def save_runoff_sources(runoff_data):
    """Save runoff sources to JSON."""
    logger.info("Saving runoff sources to JSON...")
    
    output_file = OUTPUT_DIR / 'runoff_sources.json'
    with open(output_file, 'w') as f:
        json.dump(runoff_data, f, indent=2)
    
    logger.info(f"  Saved: {output_file}")
    return output_file


# ─────────────────────────────────────────────────────────────────
# PART C: CORRELATE RUNOFF WITH DEAD ZONES
# ─────────────────────────────────────────────────────────────────

def correlate_runoff_to_hypoxia(lats, lons, nitrate_grid):
    """
    Correlate runoff plumes with downstream dead zones.
    
    Looks for:
    1. High nitrate regions (runoff plumes)
    2. Nearby hypoxic zones (from dead zone data)
    3. Directional alignment (plume points downstream to dead zone)
    
    Returns:
        dict: Links between rivers and downstream dead zones
    """
    logger.info("Correlating runoff plumes with hypoxic zones...")
    
    links_data = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'method': 'Spatial correlation + directional analysis',
            'monsoon_direction': 'Southwest (NE-SW)',
            'correlation_threshold': 0.5,
        },
        'links': []
    }
    
    # Load dead zone data if available
    dz_data = load_dead_zone_data()
    
    if dz_data is None:
        logger.warning("  Dead zone data not found, creating synthetic correlations...")
        dz_data = create_synthetic_dead_zones()
    
    # For each river, find downstream dead zones
    for river in RIVER_SOURCES:
        logger.info(f"  Analyzing plume from {river['name']}...")
        
        # Find highest nitrate cells in plume spread
        plume_cells = identify_plume_cells(lats, lons, nitrate_grid, river)
        
        # Find dead zones aligned with plume direction
        aligned_zones = find_aligned_dead_zones(
            dz_data, river,
            plume_cells,
            MONSOON_DIRECTION
        )
        
        # Create links for each aligned zone
        for zone, correlation, path in aligned_zones:
            link = {
                'river_name': river['name'],
                'river_lat': river['lat'],
                'river_lon': river['lon'],
                'downstream_zone': zone.get('name', 'Unknown Zone'),
                'zone_lat': zone['lat'],
                'zone_lon': zone['lon'],
                'correlation_strength': round(correlation, 3),
                'plume_path': [
                    {'lat': round(p[0], 2), 'lon': round(p[1], 2)}
                    for p in path[:10]  # First 10 waypoints
                ],
                'explanation': f"Monsoon plume from {river['name']} basin "
                              f"({river['basin_size']} km²) correlates with "
                              f"downstream hypoxic zone at ({zone['lat']:.1f}°N, {zone['lon']:.1f}°E)"
            }
            links_data['links'].append(link)
    
    logger.info(f"  Found {len(links_data['links'])} river-to-zone correlations")
    
    return links_data


def identify_plume_cells(lats, lons, nitrate_grid, river):
    """Identify high-nitrate cells in river plume."""
    plume_cells = []
    
    # Look around river mouth
    search_radius = 3.0  # degrees
    
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            if abs(lat - river['lat']) < search_radius and \
               abs(lon - river['lon']) < search_radius:
                
                nitrate = nitrate_grid[i, j]
                if nitrate > 100:  # High anomaly threshold
                    distance = np.sqrt(
                        (lat - river['lat'])**2 +
                        (lon - river['lon'])**2
                    )
                    plume_cells.append({
                        'lat': lat,
                        'lon': lon,
                        'nitrate': nitrate,
                        'distance': distance
                    })
    
    return sorted(plume_cells, key=lambda x: x['distance'])


def find_aligned_dead_zones(dz_data, river, plume_cells, direction):
    """Find dead zones aligned with plume direction."""
    aligned_zones = []
    
    for dz in dz_data.get('zones', []):
        # Vector from river to dead zone
        dz_lat_vec = dz['lat'] - river['lat']
        dz_lon_vec = dz['lon'] - river['lon']
        
        # Normalize
        dz_distance = np.sqrt(dz_lat_vec**2 + dz_lon_vec**2)
        if dz_distance < 0.1:
            continue
        
        dz_direction_norm = (
            dz_lat_vec / dz_distance,
            dz_lon_vec / dz_distance
        )
        
        # Calculate alignment (dot product with monsoon direction)
        alignment = abs(
            dz_direction_norm[0] * direction[0] +
            dz_direction_norm[1] * direction[1]
        )
        
        # Distance threshold: plume spreads 2-6 degrees
        if 2 < dz_distance < 6:
            # Correlation: alignment + proximity + high nitrate in plume
            correlation = alignment * (1 - dz_distance / 6)
            
            if correlation > 0.3:
                # Create path from river through plumes to zone
                path = create_plume_path(
                    river, plume_cells, dz,
                    num_waypoints=20
                )
                
                aligned_zones.append((dz, correlation, path))
    
    return sorted(aligned_zones, key=lambda x: x[1], reverse=True)


def create_plume_path(river, plume_cells, dz, num_waypoints=20):
    """Create interpolated path from river to dead zone."""
    path = []
    
    # Start at river
    path.append((river['lat'], river['lon']))
    
    # Interpolate through plume cells to dead zone
    for i in range(1, num_waypoints - 1):
        t = i / (num_waypoints - 1)
        
        # Find nearest plume cell at this progress
        if i < len(plume_cells):
            cell = plume_cells[min(i, len(plume_cells) - 1)]
            path.append((cell['lat'], cell['lon']))
        else:
            # Interpolate toward dead zone
            lat = river['lat'] + (dz['lat'] - river['lat']) * t
            lon = river['lon'] + (dz['lon'] - river['lon']) * t
            path.append((lat, lon))
    
    # End at dead zone
    path.append((dz['lat'], dz['lon']))
    
    return path


def load_dead_zone_data():
    """Load dead zone data from outputs if available."""
    try:
        dz_file = OUTPUT_DIR / 'dead_zones_summary.json'
        if dz_file.exists():
            with open(dz_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load dead zone data: {e}")
    
    return None


def create_synthetic_dead_zones():
    """Create synthetic dead zone data for testing."""
    return {
        'zones': [
            {'name': 'Arabian Sea Central', 'lat': 16.5, 'lon': 75.0, 'severity': 0.8},
            {'name': 'Bay of Bengal North', 'lat': 18.0, 'lon': 88.0, 'severity': 0.75},
            {'name': 'Bay of Bengal Central', 'lat': 15.0, 'lon': 87.0, 'severity': 0.85},
            {'name': 'Arabian Sea Coastal', 'lat': 19.0, 'lon': 70.0, 'severity': 0.6},
            {'name': 'Bay of Bengal South', 'lat': 12.5, 'lon': 85.0, 'severity': 0.7},
        ]
    }


def save_runoff_hypoxia_links(links_data):
    """Save runoff-to-hypoxia correlations to JSON."""
    logger.info("Saving runoff-to-hypoxia links to JSON...")
    
    output_file = OUTPUT_DIR / 'runoff_to_hypoxia_links.json'
    with open(output_file, 'w') as f:
        json.dump(links_data, f, indent=2)
    
    logger.info(f"  Saved: {output_file}")
    return output_file


# ─────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────

def main():
    """Main execution: create all spatial fields."""
    logger.info("="*70)
    logger.info("ANOXIA Spatial Fields Preparation")
    logger.info("="*70)
    
    # PART A: Create nitrate field
    logger.info("\n--- PART A: Creating Nitrate Anomaly Field ---")
    lats, lons, nitrate_grid = create_nitrate_grid()
    save_nitrate_netcdf(lats, lons, nitrate_grid)
    
    # PART B: Create runoff sources
    logger.info("\n--- PART B: Creating Runoff Sources Metadata ---")
    runoff_data = create_runoff_sources()
    save_runoff_sources(runoff_data)
    
    # PART C: Correlate with dead zones
    logger.info("\n--- PART C: Correlating Runoff with Dead Zones ---")
    links_data = correlate_runoff_to_hypoxia(lats, lons, nitrate_grid)
    save_runoff_hypoxia_links(links_data)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("COMPLETE: Spatial fields prepared successfully")
    logger.info("="*70)
    logger.info("\nOutput files created:")
    logger.info(f"  ✓ {OUTPUT_DIR}/nitrate_anomaly.nc (NetCDF grid)")
    logger.info(f"  ✓ {OUTPUT_DIR}/runoff_sources.json (River metadata)")
    logger.info(f"  ✓ {OUTPUT_DIR}/runoff_to_hypoxia_links.json (Correlations)")
    logger.info("\nThese files are ready for backend API serving.")
    
    return True


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)
