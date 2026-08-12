"""
ANOXIA Backend REST API
Serves oceanographic data and precursor conditions for dead zone prediction
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import warnings
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={"/*": {"origins": "*"}})

# Global data storage
DATA = {
    'dashboard_data': None,
    'dz_probability': None,
    'nitrate_anomaly': None,
    'runoff_sources': None,
    'runoff_to_hypoxia': None,
    'wind_data': None,
    'current_data': None,
}

# Interpolators
INTERPOLATORS = {
    'dz_probability': None,
    'nitrate_anomaly': None,
}

# Base paths
BASE_PATH = Path(__file__).parent
OUTPUT_PATH = BASE_PATH / 'outputs'
DATA_PATH = BASE_PATH / 'data'

# Grid definitions
LAT_MIN, LAT_MAX = 0, 30
LON_MIN, LON_MAX = 55, 100
RESOLUTION = 0.5  # degrees


def create_synthetic_grid(lat_min=0, lat_max=30, lon_min=55, lon_max=100, resolution=0.5):
    """Create synthetic 2D grids for latitude and longitude."""
    lats = np.arange(lat_min, lat_max + resolution, resolution)
    lons = np.arange(lon_min, lon_max + resolution, resolution)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    return lats, lons, lat_grid, lon_grid


def generate_realistic_plume_path(river_lat, river_lon, target_lat, target_lon, num_points=18):
    """
    Generate a realistic curved plume path from river to dead zone.
    Uses monsoon-like flow direction (southwest) with curvature.
    """
    path_lats = []
    path_lons = []
    
    # Create curved path using cubic curve interpolation
    for i in np.linspace(0, 1, num_points):
        # Quadratic bezier-like curve for realistic flow
        lat = river_lat + (target_lat - river_lat) * i
        lon = river_lon + (target_lon - river_lon) * i
        
        # Add curvature effect (monsoon/prevailing current effect)
        # Curve parameter increases then decreases
        curve_factor = 4 * i * (1 - i)  # Peaks at middle of curve
        
        # Add perpendicular displacement for realistic flow
        perp_lat_offset = curve_factor * 0.15 * np.sin(i * np.pi)
        perp_lon_offset = curve_factor * 0.1 * np.cos(i * np.pi)
        
        lat += perp_lat_offset
        lon += perp_lon_offset
        
        path_lats.append(lat)
        path_lons.append(lon)
    
    return path_lats, path_lons


def load_dashboard_data():
    """Load or create dashboard data (dead zones, traps, etc)."""
    try:
        path = OUTPUT_PATH / 'dashboard_data.json'
        if path.exists():
            with open(path, 'r') as f:
                DATA['dashboard_data'] = json.load(f)
            logger.info("✓ Loaded dashboard_data.json")
        else:
            # Create synthetic dashboard data
            DATA['dashboard_data'] = create_synthetic_dashboard_data()
            logger.warning("⚠ dashboard_data.json not found, using synthetic data")
    except Exception as e:
        logger.error(f"Error loading dashboard_data: {e}")
        DATA['dashboard_data'] = create_synthetic_dashboard_data()


def create_synthetic_dashboard_data():
    """Create synthetic dead zone data."""
    return {
        'dead_zones': [
            {
                'id': 'arabian_sea_1',
                'name': 'Arabian Sea Coastal',
                'lat': 19.0,
                'lon': 70.0,
                'radius_km': 50,
                'severity': 'high',
                'description': 'Seasonal hypoxia zone south of Narmada'
            },
            {
                'id': 'bay_bengal_1',
                'name': 'Bay of Bengal Coastal',
                'lat': 15.0,
                'lon': 85.0,
                'radius_km': 75,
                'severity': 'severe',
                'description': 'Persistent hypoxia zone near Godavari delta'
            },
            {
                'id': 'bay_bengal_2',
                'name': 'Northern Bay Shelf',
                'lat': 18.5,
                'lon': 88.0,
                'radius_km': 60,
                'severity': 'moderate',
                'description': 'Seasonal hypoxia on continental shelf'
            }
        ],
        'traps': [],
        'meta': {
            'created': datetime.now().isoformat(),
            'data_source': 'synthetic'
        }
    }


def load_nitrate_anomaly():
    """Load nitrate anomaly field from NetCDF or create synthetic."""
    try:
        # Try to load from NetCDF first
        nc_path = OUTPUT_PATH / 'nitrate_anomaly.nc'
        if nc_path.exists():
            ds = xr.open_dataset(nc_path)
            lats = ds.coords['lat'].values
            lons = ds.coords['lon'].values
            data = ds['nitrate_anomaly'].values
            DATA['nitrate_anomaly'] = {
                'lats': lats,
                'lons': lons,
                'data': data,
                'source': 'netcdf'
            }
            logger.info("✓ Loaded nitrate_anomaly.nc")
            return
    except Exception as e:
        logger.warning(f"Could not load NetCDF: {e}")
    
    # Fallback to synthetic data
    lats, lons, lat_grid, lon_grid = create_synthetic_grid()
    
    # Create realistic nitrate pattern with river plumes
    nitrate_data = np.ones_like(lat_grid) * (-20)  # Baseline
    
    # Add plumes for major rivers
    rivers = [
        {'lat': 20.5, 'lon': 86.5, 'strength': 250},  # Mahanadi
        {'lat': 16.7, 'lon': 82.3, 'strength': 350},  # Godavari
        {'lat': 14.3, 'lon': 79.8, 'strength': 200},  # Krishna
        {'lat': 21.8, 'lon': 72.6, 'strength': 180},  # Narmada
        {'lat': 24.8, 'lon': 67.2, 'strength': 220},  # Indus
    ]
    
    for river in rivers:
        distance = np.sqrt((lat_grid - river['lat'])**2 + (lon_grid - river['lon'])**2)
        plume = river['strength'] * np.exp(-distance**2 / (2 * 2**2))
        nitrate_data += plume
    
    # Add random variation
    nitrate_data += np.random.normal(0, 5, nitrate_data.shape)
    
    DATA['nitrate_anomaly'] = {
        'lats': lats,
        'lons': lons,
        'data': np.clip(nitrate_data, -50, 500),
        'source': 'synthetic'
    }
    logger.warning("⚠ Nitrate anomaly not found, using synthetic data")


def load_dz_probability():
    """Load dead zone probability field from NetCDF or create synthetic."""
    try:
        nc_path = OUTPUT_PATH / 'dz_prediction_30day.nc'
        if nc_path.exists():
            ds = xr.open_dataset(nc_path)
            lats = ds.coords['lat'].values
            lons = ds.coords['lon'].values
            data = ds["prob_hypoxia"].values
            DATA['dz_probability'] = {
                'lats': lats,
                'lons': lons,
                'data': data,
                'source': 'netcdf'
            }
            logger.info("✓ Loaded dz_prediction_30day.nc")
            return
    except Exception as e:
        logger.warning(f"Could not load DZ probability: {e}")
    
    # Synthetic probability field
    lats, lons, lat_grid, lon_grid = create_synthetic_grid()
    
    # Higher probability near dead zones
    dead_zones = [
        {'lat': 19.0, 'lon': 70.0, 'strength': 0.7},
        {'lat': 15.0, 'lon': 85.0, 'strength': 0.8},
        {'lat': 18.5, 'lon': 88.0, 'strength': 0.6},
    ]
    
    prob_data = np.ones_like(lat_grid) * 0.1  # Baseline
    
    for zone in dead_zones:
        distance = np.sqrt((lat_grid - zone['lat'])**2 + (lon_grid - zone['lon'])**2)
        peak = zone['strength'] * np.exp(-distance**2 / (2 * 3**2))
        prob_data = np.maximum(prob_data, peak)
    
    DATA['dz_probability'] = {
        'lats': lats,
        'lons': lons,
        'data': np.clip(prob_data, 0, 1),
        'source': 'synthetic'
    }
    logger.warning("⚠ DZ probability not found, using synthetic data")


def load_runoff_sources():
    """Load river runoff source metadata."""
    try:
        path = OUTPUT_PATH / 'runoff_sources.json'
        if path.exists():
            with open(path, 'r') as f:
                DATA['runoff_sources'] = json.load(f)
            logger.info("✓ Loaded runoff_sources.json")
            return
    except Exception as e:
        logger.error(f"Error loading runoff_sources: {e}")
    
    # Synthetic river data
    DATA['runoff_sources'] = {
        'rivers': [
            {
                'name': 'Godavari',
                'lat': 16.7,
                'lon': 82.3,
                'flow_rate': 'HIGH',
                'basin_size_km2': 312812,
                'discharge_class': 'HIGH',
                'plume_direction': {'lat_component': -0.707, 'lon_component': -0.707}
            },
            {
                'name': 'Indus',
                'lat': 24.8,
                'lon': 67.2,
                'flow_rate': 'HIGH',
                'basin_size_km2': 1165500,
                'discharge_class': 'HIGH',
                'plume_direction': {'lat_component': -0.707, 'lon_component': -0.707}
            },
            {
                'name': 'Krishna',
                'lat': 14.3,
                'lon': 79.8,
                'flow_rate': 'HIGH',
                'basin_size_km2': 258948,
                'discharge_class': 'HIGH',
                'plume_direction': {'lat_component': -0.707, 'lon_component': -0.707}
            },
            {
                'name': 'Mahanadi',
                'lat': 20.5,
                'lon': 86.5,
                'flow_rate': 'MEDIUM',
                'basin_size_km2': 141589,
                'discharge_class': 'MEDIUM',
                'plume_direction': {'lat_component': -0.707, 'lon_component': -0.707}
            },
            {
                'name': 'Narmada',
                'lat': 21.8,
                'lon': 72.6,
                'flow_rate': 'MEDIUM',
                'basin_size_km2': 98796,
                'discharge_class': 'MEDIUM',
                'plume_direction': {'lat_component': -0.707, 'lon_component': -0.707}
            }
        ],
        'meta': {'data_source': 'synthetic'}
    }
    logger.warning("⚠ runoff_sources.json not found, using synthetic data")


def load_runoff_to_hypoxia():
    """Load plume pathways to dead zones."""
    try:
        path = OUTPUT_PATH / 'runoff_to_hypoxia_links.json'
        if path.exists():
            with open(path, 'r') as f:
                loaded_data = json.load(f)
                # Only use if it has all 5 pathways
                if loaded_data.get('links') and len(loaded_data.get('links', [])) >= 5:
                    DATA['runoff_to_hypoxia'] = loaded_data
                    logger.info(f"✓ Loaded runoff_to_hypoxia_links.json with {len(loaded_data.get('links', []))} pathways")
                    return
                else:
                    logger.warning(f"⚠ JSON file incomplete ({len(loaded_data.get('links', []))} pathways), regenerating all 5...")
    except Exception as e:
        logger.error(f"Error loading runoff_to_hypoxia: {e}")
    
    # Always generate all 5 river pathways
    # River coordinates and their target dead zones
    river_links = [
        {
            'river_name': 'Narmada',
            'river_lat': 21.8,
            'river_lon': 72.6,
            'downstream_zone': 'Arabian Sea Coastal',
            'zone_lat': 19.0,
            'zone_lon': 70.0,
            'correlation_strength': 0.36
        },
        {
            'river_name': 'Godavari',
            'river_lat': 16.7,
            'river_lon': 82.3,
            'downstream_zone': 'Bay of Bengal Coastal',
            'zone_lat': 15.0,
            'zone_lon': 85.0,
            'correlation_strength': 0.75
        },
        {
            'river_name': 'Krishna',
            'river_lat': 14.3,
            'river_lon': 79.8,
            'downstream_zone': 'Bay of Bengal Coastal',
            'zone_lat': 15.0,
            'zone_lon': 85.0,
            'correlation_strength': 0.58
        },
        {
            'river_name': 'Indus',
            'river_lat': 24.8,
            'river_lon': 67.2,
            'downstream_zone': 'Arabian Sea Coastal',
            'zone_lat': 19.0,
            'zone_lon': 70.0,
            'correlation_strength': 0.42
        },
        {
            'river_name': 'Mahanadi',
            'river_lat': 20.5,
            'river_lon': 86.5,
            'downstream_zone': 'Northern Bay Shelf',
            'zone_lat': 18.5,
            'zone_lon': 88.0,
            'correlation_strength': 0.52
        }
    ]
    
    # Generate realistic plume paths for each link
    links = []
    for link in river_links:
        plume_lats, plume_lons = generate_realistic_plume_path(
            link['river_lat'], link['river_lon'],
            link['zone_lat'], link['zone_lon'],
            num_points=18
        )
        
        links.append({
            'river_name': link['river_name'],
            'downstream_zone': link['downstream_zone'],
            'correlation_strength': link['correlation_strength'],
            'plume_path': [[lat, lon] for lat, lon in zip(plume_lats, plume_lons)]
        })
    
    DATA['runoff_to_hypoxia'] = {'links': links}
    logger.warning("⚠ runoff_to_hypoxia_links.json not found, generated realistic paths")


def setup_interpolators():
    """Create RegularGridInterpolator objects for spatial queries."""
    try:
        # DZ Probability interpolator
        if DATA['dz_probability']:
            lats = DATA['dz_probability']['lats']
            lons = DATA['dz_probability']['lons']
            values = DATA['dz_probability']['data']
            
            # Ensure lats and lons are sorted
            if lats[0] > lats[-1]:
                lats = lats[::-1]
                values = values[::-1, :]
            if lons[0] > lons[-1]:
                lons = lons[::-1]
                values = values[:, ::-1]
            
            INTERPOLATORS['dz_probability'] = RegularGridInterpolator(
                (lats, lons), values,
                bounds_error=False,
                fill_value=np.mean(values)
            )
            logger.info("✓ DZ probability interpolator ready")
        
        # Nitrate anomaly interpolator
        if DATA['nitrate_anomaly']:
            lats = DATA['nitrate_anomaly']['lats']
            lons = DATA['nitrate_anomaly']['lons']
            values = DATA['nitrate_anomaly']['data']
            
            if lats[0] > lats[-1]:
                lats = lats[::-1]
                values = values[::-1, :]
            if lons[0] > lons[-1]:
                lons = lons[::-1]
                values = values[:, ::-1]
            
            INTERPOLATORS['nitrate_anomaly'] = RegularGridInterpolator(
                (lats, lons), values,
                bounds_error=False,
                fill_value=np.mean(values)
            )
            logger.info("✓ Nitrate anomaly interpolator ready")
    
    except Exception as e:
        logger.error(f"Error setting up interpolators: {e}")


def load_all_data():
    """Load all global data at startup."""
    logger.info("=" * 50)
    logger.info("Loading ANOXIA backend data...")
    logger.info("=" * 50)
    
    load_dashboard_data()
    load_nitrate_anomaly()
    load_dz_probability()
    load_runoff_sources()
    load_runoff_to_hypoxia()
    setup_interpolators()
    
    logger.info("=" * 50)
    logger.info("Data loading complete")
    logger.info("=" * 50)


def find_nearby_zones(lat, lon, radius_km=100):
    """Find dead zones within radius of point."""
    if not DATA['dashboard_data'] or 'dead_zones' not in DATA['dashboard_data']:
        return []
    
    nearby = []
    # Rough km per degree at equator
    km_per_degree = 111.0
    degree_radius = radius_km / km_per_degree
    
    for zone in DATA['dashboard_data']['dead_zones']:
        distance = np.sqrt((zone['lat'] - lat)**2 + (zone['lon'] - lon)**2)
        if distance <= degree_radius:
            nearby.append({
                'name': zone['name'],
                'distance_km': float(distance * km_per_degree),
                'severity': zone['severity'],
                'probability': float(query_interpolator('dz_probability', lat, lon))
            })
    
    return sorted(nearby, key=lambda x: x['distance_km'])


def identify_zone(lat, lon):
    """Identify which oceanographic zone the location is in."""
    # Arabian Sea: 8-20°N, 50-75°E
    if 8 <= lat <= 20 and 50 <= lon <= 75:
        return 'Arabian Sea Coastal'
    # Bay of Bengal: 8-20°N, 85-100°E
    elif 8 <= lat <= 20 and 85 <= lon <= 100:
        return 'Bay of Bengal Coastal'
    # Northern Bay Shelf: 20-30°N, 85-100°E
    elif 20 <= lat <= 30 and 85 <= lon <= 100:
        return 'Northern Bay Shelf'
    # General Indian Ocean
    else:
        return 'Indian Ocean Regional'


def is_ocean_water_point(lat, lon):
    """Coarse ocean mask for Arabian Sea and Bay of Bengal, excluding major land masses."""
    in_arabian_sea = 6 <= lat <= 24 and 57 <= lon <= 74.8
    in_bay_of_bengal = 6 <= lat <= 24 and 80.3 <= lon <= 96.5

    if not (in_arabian_sea or in_bay_of_bengal):
        return False

    on_indian_peninsula = 8 <= lat <= 22 and 73.2 <= lon <= 80.8
    on_sri_lanka = 5.5 <= lat <= 10.5 and 79 <= lon <= 82.2
    on_bangladesh_delta = 20.5 <= lat <= 23.8 and 88 <= lon <= 92

    return not (on_indian_peninsula or on_sri_lanka or on_bangladesh_delta)


def calculate_severity_score(nitrate_pct, thermal_strat, wind_stress, do_drawdown, hypoxia_prob):
    """
    Calculate overall severity score (0-3+).
    > 2.0: CRITICAL
    1.0-2.0: WARNING
    < 1.0: LOW
    """
    score = 0.0
    
    # Nitrate contribution (0-1.0)
    if nitrate_pct > 200:
        score += 1.0
    elif nitrate_pct > 150:
        score += 0.8
    elif nitrate_pct > 100:
        score += 0.6
    elif nitrate_pct > 50:
        score += 0.4
    
    # DO Drawdown contribution (0-1.0)
    if do_drawdown > 0.7:
        score += 1.0
    elif do_drawdown > 0.5:
        score += 0.7
    elif do_drawdown > 0.3:
        score += 0.4
    
    # Stratification + Low wind = stagnation (0-0.8)
    if thermal_strat > 0.6 and wind_stress < 0.15:
        score += 0.5
    
    # Hypoxia probability (0-0.7)
    score += hypoxia_prob * 0.7
    
    return round(score, 2)


def determine_severity_level(score):
    """Determine severity level from score."""
    if score > 2.0:
        return "CRITICAL"
    elif score >= 1.0:
        return "WARNING"
    else:
        return "SAFE"


def generate_interventions(nitrate_pct, thermal_strat, wind_stress, do_drawdown, zone, hypoxia_prob=0.15):
    """
    Generate dynamic intervention measures based on precursor conditions.
    
    Returns list of prioritized interventions with urgency levels, severity score, and count.
    """
    interventions = []
    
    # Calculate overall severity
    severity_score = calculate_severity_score(nitrate_pct, thermal_strat, wind_stress, do_drawdown, hypoxia_prob)
    severity_level = determine_severity_level(severity_score)
    
    # --- HIGH NITRATE (Primary driver) ---
    if nitrate_pct > 150:
        interventions.append({
            'title': 'Reduce agricultural fertilizer runoff',
            'priority': 'CRITICAL' if severity_score > 2.0 else 'URGENT',
            'category': 'Nutrient Reduction',
            'reason': f'High nitrate anomaly detected ({nitrate_pct:.0f}% above baseline)',
            'impact': 'Prevents algal bloom proliferation and oxygen depletion',
            'timeline': 'Immediate - 2 months',
            'measures': [
                'Implement precision agriculture with drip irrigation',
                'Establish riparian buffer zones (500m minimum)',
                'Reduce synthetic fertilizer use by 50-70%',
                'Enforce wastewater treatment standards',
                'Monitor discharge from agricultural runoff'
            ]
        })
    elif nitrate_pct > 75:
        interventions.append({
            'title': 'Optimize fertilizer application rates',
            'priority': 'URGENT',
            'category': 'Nutrient Reduction',
            'reason': f'Elevated nitrate levels detected ({nitrate_pct:.0f}% above baseline)',
            'impact': 'Reduces algal bloom risk and improves water quality',
            'timeline': '2-4 weeks',
            'measures': [
                'Reduce fertilizer application in sensitive zones',
                'Improve irrigation efficiency',
                'Implement best management practices (BMPs)',
                'Monitor nutrient levels bi-weekly'
            ]
        })
    elif nitrate_pct > 25:
        interventions.append({
            'title': 'Monitor nutrient levels regularly',
            'priority': 'ROUTINE',
            'category': 'Nutrient Monitoring',
            'reason': f'Moderate nitrate levels present ({nitrate_pct:.0f}% above baseline)',
            'impact': 'Early detection of potential nutrient stress',
            'timeline': 'Ongoing - weekly checks',
            'measures': [
                'Routine monitoring of nutrient levels',
                'Maintain current agricultural practices',
                'Track seasonal variations'
            ]
        })
    
    # --- STRONG THERMAL STRATIFICATION (Prevents oxygen mixing) ---
    if thermal_strat > 0.7 and do_drawdown > 0.5:
        interventions.append({
            'title': 'Deploy artificial aeration systems',
            'priority': 'CRITICAL',
            'category': 'Water Mixing Enhancement',
            'reason': f'Strong stratification ({int(thermal_strat*100)}%) preventing oxygen mixing with fast drawdown',
            'impact': 'Restores oxygen circulation and prevents anoxia formation',
            'timeline': 'Immediate - 1 week',
            'measures': [
                'Deploy artificial aeration systems',
                'Mechanical destratification (dredging/pumping)',
                'Alter dam discharge patterns for mixing',
                'Monitor thermocline depth continuously'
            ]
        })
    elif thermal_strat > 0.5:
        interventions.append({
            'title': 'Increase water column mixing',
            'priority': 'URGENT',
            'category': 'Water Mixing Enhancement',
            'reason': f'Moderate-to-strong stratification ({int(thermal_strat*100)}%) detected',
            'impact': 'Facilitates oxygen distribution and prevents stagnation',
            'timeline': '1-2 weeks',
            'measures': [
                'Increase water circulation',
                'Manage reservoir discharge timing',
                'Monitor stratification patterns'
            ]
        })
    
    # --- VERY FAST DO DRAWDOWN (Oxygen depletion) ---
    if do_drawdown > 0.6:
        interventions.append({
            'title': 'Emergency oxygen restoration deployment',
            'priority': 'CRITICAL',
            'category': 'Oxygen Restoration',
            'reason': f'Very fast dissolved oxygen depletion rate detected ({int(do_drawdown*100)}%)',
            'impact': 'Prevents complete anoxia and protects aquatic ecosystems',
            'timeline': 'IMMEDIATE - within 48 hours',
            'measures': [
                'Emergency oxygenation deployment',
                'Increase freshwater flushing',
                'Reduce hypoxic zone through nutrient control',
                'Deploy oxygen diffusers in affected areas',
                '🚨 Alert fisheries and maritime authorities'
            ]
        })
    elif do_drawdown > 0.4:
        interventions.append({
            'title': 'Prepare oxygen support infrastructure',
            'priority': 'URGENT',
            'category': 'Oxygen Support',
            'reason': f'Fast dissolved oxygen drawdown rate ({int(do_drawdown*100)}%) detected',
            'impact': 'Enables rapid response to hypoxia events',
            'timeline': '1 week',
            'measures': [
                'Prepare emergency response protocols',
                'Increase monitoring to daily',
                'Coordinate oxygen injection capability'
            ]
        })
    
    # --- LOW WIND STRESS (Insufficient natural mixing) ---
    if wind_stress < 0.1 and thermal_strat > 0.5:
        interventions.append({
            'title': 'Monitor stagnation risk zones',
            'priority': 'WARNING',
            'category': 'Wind-Driven Mixing',
            'reason': f'Low wind mixing ({int(wind_stress*100)}%) combined with stratification may cause stagnation',
            'impact': 'Early warning of potential prolonged hypoxia periods',
            'timeline': 'Seasonal monitoring',
            'measures': [
                'Intensify monitoring during calm periods',
                'Prepare for off-monsoon stratification',
                'Deploy mechanical mixing if needed',
                'Track seasonal wind pattern changes'
            ]
        })
    
    # --- ZONE-SPECIFIC INTERVENTIONS ---
    if zone == 'Bay of Bengal Coastal':
        interventions.append({
            'title': '⚠️ Regional coordination: Bay of Bengal',
            'priority': 'HIGH',
            'category': 'Regional Management',
            'reason': f'Highest-risk zone for hypoxia development during monsoon',
            'impact': 'Coordinates multi-stakeholder response in Bay of Bengal',
            'timeline': 'Seasonal - pre/post monsoon',
            'measures': [
                'Coordinate with Godavari/Mahanadi river dam operators',
                'Manage monsoon surge planning',
                'Monitor fishery impact zones',
                'Implement regional nutrient policy'
            ]
        })
    elif zone == 'Arabian Sea Coastal':
        interventions.append({
            'title': '⚠️ Regional coordination: Arabian Sea',
            'priority': 'HIGH',
            'category': 'Regional Management',
            'reason': f'Arabian Sea susceptible to upwelling-driven hypoxia',
            'impact': 'Protects fishing zones and coastal ecosystems',
            'timeline': 'Seasonal management',
            'measures': [
                'Coordinate with Narmada river management',
                'Control shrimp farming nutrient impacts',
                'Implement coastal water quality standards',
                'Monitor upwelling-induced oxygen depletion',
                'Track seasonal current patterns'
            ]
        })
    
    # Add general/ongoing measures if none critical
    if not interventions:
        interventions.append({
            'title': '✅ Maintain preventive monitoring',
            'priority': 'ROUTINE',
            'category': 'Preventive Monitoring',
            'reason': 'Current conditions are within safe operating parameters',
            'impact': 'Continuous assessment prevents degradation',
            'timeline': 'Ongoing - weekly checks',
            'measures': [
                'Continue regular monitoring',
                'Maintain current best practices',
                'Track seasonal patterns',
                'Update forecasting models'
            ]
        })
    
    # Sort interventions by priority
    priority_order = {'CRITICAL': 0, 'URGENT': 1, 'WARNING': 2, 'HIGH': 3, 'ROUTINE': 4}
    interventions.sort(key=lambda x: priority_order.get(x['priority'], 5))
    
    return interventions, severity_score, severity_level


def query_interpolator(name, lat, lon):
    """Query interpolator at (lat, lon)."""
    try:
        if INTERPOLATORS[name] is None:
            return np.nan
        
        point = np.array([[lat, lon]])
        result = INTERPOLATORS[name](point)[0]
        return float(result)
    except Exception as e:
        logger.warning(f"Interpolation error for {name}: {e}")
        return np.nan


def calculate_synthetic_thermal_stratification(lat, lon, nitrate):
    """Estimate thermal stratification from location and nitrate."""
    # Higher nitrate → more eutrophication → weaker stratification
    # But coastal areas tend to be stratified
    base = 0.6 if 15 < lat < 20 else 0.4
    reduction = (nitrate / 500.0) * 0.3
    value = max(0.2, base - reduction)
    return value


def calculate_synthetic_wind_stress(lat, lon):
    """Estimate wind stress from location."""
    # Monsoon-driven wind stress
    # SW monsoon = higher stress
    if 15 < lat < 25 and 65 < lon < 85:
        return 0.3  # HIGH
    elif 10 < lat < 30 and 55 < lon < 100:
        return 0.2  # MEDIUM
    else:
        return 0.1  # LOW


def calculate_synthetic_do_drawdown(lat, lon, nitrate):
    """Estimate dissolved oxygen drawdown from nutrients."""
    # High nitrate + coastal = strong drawdown
    if 14 < lat < 20 and 78 < lon < 90:
        return 0.7 + (nitrate / 500.0) * 0.2
    else:
        return 0.3 + (nitrate / 500.0) * 0.3


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/precursor-conditions/<float:lat>/<float:lon>', methods=['GET'])
def get_precursor_conditions(lat, lon):
    """
    Return precursor conditions at clicked location.
    
    URL: /api/precursor-conditions/15.5/71.3
    """
    try:
        # Input validation
        if not (0 <= lat <= 30 and 55 <= lon <= 100):
            return jsonify({
                'error': 'Location outside coverage area (0-30N, 55-100E)',
                'lat': lat,
                'lon': lon
            }), 400
        
        # Query gridded fields
        nitrate_pct = query_interpolator('nitrate_anomaly', lat, lon)
        dz_prob = query_interpolator('dz_probability', lat, lon)
        
        # Synthetic fields (since we don't have real data)
        thermal_strat = calculate_synthetic_thermal_stratification(lat, lon, nitrate_pct)
        wind_stress = calculate_synthetic_wind_stress(lat, lon)
        do_drawdown = calculate_synthetic_do_drawdown(lat, lon, nitrate_pct)
        
        # Map to categorical strings with percentages
        if thermal_strat > 0.7:
            thermal_str = "STRONG"
            thermal_pct = int(thermal_strat * 100)
        elif thermal_strat > 0.4:
            thermal_str = "MODERATE"
            thermal_pct = int((thermal_strat - 0.4) * 100 + 40)
        else:
            thermal_str = "WEAK"
            thermal_pct = int(thermal_strat * 50)
        
        if wind_stress > 0.25:
            wind_str = "HIGH"
            wind_pct = int(wind_stress * 100)
        elif wind_stress > 0.15:
            wind_str = "MEDIUM"
            wind_pct = int((wind_stress - 0.15) * 100 + 50)
        else:
            wind_str = "LOW"
            wind_pct = int(wind_stress * 100)
        
        if do_drawdown > 0.6:
            do_str = "VERY FAST"
            do_pct = int(do_drawdown * 100)
        elif do_drawdown > 0.4:
            do_str = "FAST"
            do_pct = int((do_drawdown - 0.4) * 100 + 40)
        else:
            do_str = "MODERATE"
            do_pct = int(do_drawdown * 100)
        
        # Find nearby dead zones
        nearby_zones = find_nearby_zones(lat, lon)
        
        response = {
            'location': {
                'lat': lat,
                'lon': lon
            },
            'precursors': {
                'nitrate_anomaly': max(-50, min(500, float(nitrate_pct))),
                'chlorophyll_modis': float(nitrate_pct * 0.5),  # Correlated
                'thermal_stratification': thermal_str,
                'thermal_stratification_pct': int(thermal_pct),
                'wind_stress': wind_str,
                'wind_stress_pct': int(wind_pct),
                'do_drawdown': do_str,
                'do_drawdown_pct': int(do_pct)
            },
            'probabilities': {
                'hypoxia_30day': float(dz_prob) if not np.isnan(dz_prob) else 0.15
            },
            'nearby_zones': nearby_zones,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'data_source': 'synthetic'
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error in precursor-conditions: {e}")
        return jsonify({'error': str(e), 'lat': lat, 'lon': lon}), 500


@app.route('/api/intervention-measures/<float:lat>/<float:lon>', methods=['GET'])
def get_intervention_measures(lat, lon):
    """
    Generate intervention measures based on precursor conditions and location.
    
    URL: /api/intervention-measures/15.5/71.3
    
    Response includes:
    - Dynamic interventions based on precursor conditions
    - Severity level (CRITICAL/WARNING/SAFE)
    - Count of recommended actions
    """
    try:
        # Input validation
        if not (0 <= lat <= 30 and 55 <= lon <= 100):
            return jsonify({
                'error': 'Location outside coverage area (0-30N, 55-100E)',
                'lat': lat,
                'lon': lon
            }), 400
        
        # Get precursor data
        nitrate_pct = query_interpolator('nitrate_anomaly', lat, lon)
        dz_prob = query_interpolator('dz_probability', lat, lon)
        thermal_strat = calculate_synthetic_thermal_stratification(lat, lon, nitrate_pct)
        wind_stress = calculate_synthetic_wind_stress(lat, lon)
        do_drawdown = calculate_synthetic_do_drawdown(lat, lon, nitrate_pct)
        
        # Handle NaN values
        if np.isnan(nitrate_pct):
            nitrate_pct = 60.0  # Default moderate value
        if np.isnan(dz_prob):
            dz_prob = 0.15
        
        # Determine zone
        zone = identify_zone(lat, lon)
        
        # Generate interventions based on precursor severity
        interventions, severity_score, severity_level = generate_interventions(
            nitrate_pct, thermal_strat, wind_stress, do_drawdown, zone, dz_prob
        )
        
        response = {
            'location': {
                'lat': float(lat),
                'lon': float(lon)
            },
            'zone': zone,
            'precursor_summary': {
                'nitrate_anomaly': float(max(-50, min(500, nitrate_pct))),
                'hypoxia_probability': float(dz_prob),
                'thermal_stratification': f"{int(thermal_strat*100)}%",
                'wind_stress': f"{int(wind_stress*100)}%",
                'do_drawdown_rate': f"{int(do_drawdown*100)}%"
            },
            'severity_score': severity_score,
            'severity_level': severity_level,
            'interventions': interventions,
            'recommended_actions_count': len(interventions),
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'data_source': 'dynamic_analysis',
                'ai_generated': True,
                'note': 'AI-generated recommendations based on environmental conditions'
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error in intervention-measures: {e}")
        return jsonify({
            'error': str(e),
            'lat': lat,
            'lon': lon,
            'severity_level': 'UNKNOWN',
            'interventions': []
        }), 500


@app.route('/api/wind-vectors', methods=['GET'])
def get_wind_vectors():
    """Return wind vectors at coarse grid resolution."""
    try:
        # Create coarse grid (every 2 degrees)
        lats = np.arange(0, 31, 2.0)
        lons = np.arange(55, 101, 2.0)
        
        vectors = []
        for lat in lats:
            for lon in lons:
                # Monsoon wind pattern (SW direction)
                magnitude = calculate_synthetic_wind_stress(lat, lon)
                u = -magnitude * 0.707  # SW component (westward)
                v = -magnitude * 0.707  # SW component (southward)
                
                vectors.append({
                    'lat': float(lat),
                    'lon': float(lon),
                    'u': float(u),
                    'v': float(v),
                    'magnitude': float(magnitude)
                })
        
        return jsonify({
            'vectors': vectors,
            'grid': {
                'lat_range': [0, 30],
                'lon_range': [55, 100],
                'resolution': 2.0
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'description': 'Wind vectors for visualization'
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Error in wind-vectors: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ocean-currents', methods=['GET'])
def get_ocean_currents():
    """Return ocean current vectors."""
    try:
        # Create coarse grid (every 2 degrees)
        lats = np.arange(0, 31, 2.0)
        lons = np.arange(55, 101, 2.0)
        
        vectors = []
        for lat in lats:
            for lon in lons:
                # Monsoon-driven currents (SW direction)
                if 15 < lat < 25:
                    magnitude = 0.4  # Strong current during monsoon
                else:
                    magnitude = 0.15  # Weaker current
                
                # Add coastal current component
                if lon < 70:  # Western coast
                    u = -magnitude * 0.8
                    v = -magnitude * 0.2
                elif lon < 85:  # Central bay
                    u = -magnitude * 0.707
                    v = -magnitude * 0.707
                else:  # Eastern coast
                    u = -magnitude * 0.5
                    v = -magnitude * 0.866
                
                vectors.append({
                    'lat': float(lat),
                    'lon': float(lon),
                    'u': float(u),
                    'v': float(v),
                    'magnitude': float(magnitude)
                })
        
        return jsonify({
            'vectors': vectors,
            'grid': {
                'lat_range': [0, 30],
                'lon_range': [55, 100],
                'resolution': 2.0
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'description': 'Ocean surface currents'
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Error in ocean-currents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/fertilizer-runoff', methods=['GET'])
def get_fertilizer_runoff():
    """Return nitrate grid and river sources for heatmap."""
    try:
        if not DATA['nitrate_anomaly']:
            return jsonify({
                'error': 'Nitrate data not available',
                'status': 'fallback'
            }), 503
        
        lats = DATA['nitrate_anomaly']['lats']
        lons = DATA['nitrate_anomaly']['lons']
        values = DATA['nitrate_anomaly']['data']
        
        # Create flat gridded representation
        grid_points = []
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                grid_points.append({
                    'lat': float(lat),
                    'lon': float(lon),
                    'value': float(values[i, j])
                })
        
        response = {
            'nitrate_grid': {
                'points': grid_points,
                'bounds': {
                    'lat': [float(lats.min()), float(lats.max())],
                    'lon': [float(lons.min()), float(lons.max())]
                },
                'colorscale': 'YlOrRd',
                'value_range': [-50, 500]
            },
            'river_sources': DATA['runoff_sources']['rivers'] if DATA['runoff_sources'] else [],
            'correlations': DATA['runoff_to_hypoxia']['links'] if DATA['runoff_to_hypoxia'] else [],
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'grid_size': f"{len(lats)} × {len(lons)}",
                'points_count': len(grid_points)
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error in fertilizer-runoff: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/dead-zone-markers', methods=['GET'])
def get_dead_zone_markers():
    """Return dead zone locations with probabilities."""
    try:
        if not DATA['dashboard_data'] or 'dead_zones' not in DATA['dashboard_data']:
            return jsonify({'zones': []}), 200
        
        zones = []
        for zone in DATA['dashboard_data']['dead_zones']:
            prob = query_interpolator('dz_probability', zone['lat'], zone['lon'])
            zones.append({
                'id': zone.get('id', ''),
                'name': zone['name'],
                'lat': zone['lat'],
                'lon': zone['lon'],
                'radius_km': zone.get('radius_km', 50),
                'severity': zone.get('severity', 'unknown'),
                'hypoxia_probability': float(prob) if not np.isnan(prob) else 0.3,
                'description': zone.get('description', '')
            })
        
        return jsonify({
            'zones': zones,
            'count': len(zones),
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'coverage': 'Indian Ocean (0-30N, 55-100E)'
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Error in dead-zone-markers: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/dz-probability-field', methods=['GET'])
def get_dz_probability_field():
    """Return gridded hypoxia probability field from backend data for heatmap rendering."""
    try:
        if not DATA['dz_probability']:
            return jsonify({'lats': [], 'lons': [], 'probs': [], 'count': 0}), 200

        lats = DATA['dz_probability'].get('lats', [])
        lons = DATA['dz_probability'].get('lons', [])
        values = DATA['dz_probability'].get('data', None)

        if values is None:
            return jsonify({'lats': [], 'lons': [], 'probs': [], 'count': 0}), 200

        ocean_only = request.args.get('ocean_only', 'true').lower() != 'false'
        min_prob = request.args.get('min_prob', '0.02')
        try:
            min_prob = max(0.0, min(1.0, float(min_prob)))
        except Exception:
            min_prob = 0.02

        out_lats = []
        out_lons = []
        out_probs = []

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                prob = values[i, j]
                if np.isnan(prob):
                    continue

                lat_f = float(lat)
                lon_f = float(lon)
                prob_f = float(max(0.0, min(1.0, prob)))

                if prob_f < min_prob:
                    continue

                if ocean_only and not is_ocean_water_point(lat_f, lon_f):
                    continue

                out_lats.append(lat_f)
                out_lons.append(lon_f)
                out_probs.append(prob_f)

        return jsonify({
            'lats': out_lats,
            'lons': out_lons,
            'probs': out_probs,
            'count': len(out_probs),
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'ocean_only': ocean_only,
                'min_prob': min_prob,
                'source': DATA['dz_probability'].get('source', 'unknown')
            }
        }), 200

    except Exception as e:
        logger.error(f"Error in dz-probability-field: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/runoff-to-hypoxia-pathways', methods=['GET'])
def get_runoff_to_hypoxia_pathways():
    """Return plume pathways linking rivers to dead zones."""
    try:
        if not DATA['runoff_to_hypoxia'] or 'links' not in DATA['runoff_to_hypoxia']:
            return jsonify({'links': []}), 200
        
        links = []
        for link in DATA['runoff_to_hypoxia']['links']:
            links.append({
                'river_name': link.get('river_name', ''),
                'downstream_zone': link.get('downstream_zone', ''),
                'correlation_strength': link.get('correlation_strength', 0.0),
                'plume_path': link.get('plume_path', []),
                'color': 'rgba(255, 165, 0, 0.5)'  # Orange with transparency
            })
        
        return jsonify({
            'links': links,
            'count': len(links),
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'description': 'River plume pathways to dead zones'
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Error in runoff-to-hypoxia-pathways: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'data_loaded': {
            'dashboard_data': DATA['dashboard_data'] is not None,
            'nitrate_anomaly': DATA['nitrate_anomaly'] is not None,
            'dz_probability': DATA['dz_probability'] is not None,
            'runoff_sources': DATA['runoff_sources'] is not None,
            'runoff_to_hypoxia': DATA['runoff_to_hypoxia'] is not None
        }
    }), 200


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API documentation."""
    return jsonify({
        'api': 'ANOXIA Backend REST API',
        'version': '1.0',
        'endpoints': {
            '/api/health': 'System health check',
            '/api/precursor-conditions/<lat>/<lon>': 'Get conditions at location',
            '/api/wind-vectors': 'Get wind vectors for visualization',
            '/api/ocean-currents': 'Get ocean currents for visualization',
            '/api/fertilizer-runoff': 'Get nitrate grid and river sources',
            '/api/dead-zone-markers': 'Get dead zone locations and probabilities',
            '/api/dz-probability-field': 'Get gridded hypoxia probability field (backend truth)',
            '/api/runoff-to-hypoxia-pathways': 'Get plume pathways'
        },
        'coverage': {
            'latitude': '0-30°N',
            'longitude': '55-100°E',
            'resolution': '0.5°'
        }
    }), 200


# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================




if __name__ == '__main__':
    logger.info("Starting ANOXIA Backend API...")
    logger.info("CORS enabled for all origins")
    logger.info("Visit http://localhost:5000 for API documentation")
    
    # Load data in background thread so server starts immediately
    data_thread = threading.Thread(target=load_all_data, daemon=True)
    data_thread.start()
    logger.info("📊 Data loading initiated in background...")

    # Use platform-provided port in production (e.g., Render), fallback to 5000 locally.
    port = int(os.getenv('PORT', '5000'))
    
    # Run Flask server immediately (don't wait for data)
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False
    )
