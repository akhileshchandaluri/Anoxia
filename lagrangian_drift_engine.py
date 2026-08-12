"""
Lagrangian drift engine for ghost gear tracking.

Simulates particle trajectories using real HYCOM ocean currents.

Pipeline:
1. Load ghost gear seed points
2. Load all available HYCOM current files (time series)
3. For each seed point:
   - Initialize particle at (lat, lon)
   - For each time step (6-hour intervals):
     * Interpolate HYCOM u, v at particle location
     * Update position using velocity fields
     * Record trajectory
4. Save all trajectories to JSON

Notes:
- Uses scipy.interpolate.RegularGridInterpolator for spatial interpolation
- Converts m/s velocities to lat/lon degrees
- Clamps to domain bounds [lat 0-35, lon 50-105]
"""

import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


def load_seed_points(json_path: str = "./data/gfw/gfw_seed_points.json") -> List[Tuple[float, float]]:
    """Load ghost gear seed points."""
    print(f"Loading seed points from {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract seed points - handle both formats
    if 'seed_points' in data:
        seed_points = data['seed_points']
    else:
        seed_points = data.get('seeds', [])
    
    # Convert to tuples
    seed_points = [tuple(pt) if isinstance(pt, list) else pt for pt in seed_points]
    
    print(f"  Loaded {len(seed_points)} seed points")
    return seed_points


def find_hycom_files(hycom_dir: str = "./data/hycom/") -> List[str]:
    """
    Find and sort HYCOM NetCDF files by date.
    
    Assumes filename format: hycom_YYYYMMDD.nc
    """
    print(f"\nFinding HYCOM files in {hycom_dir}...")
    
    hycom_files = sorted(glob.glob(f"{hycom_dir}hycom_*.nc"))
    
    if not hycom_files:
        raise FileNotFoundError(f"No HYCOM files found in {hycom_dir}")
    
    print(f"  Found {len(hycom_files)} files:")
    for f in hycom_files[:3]:
        print(f"    - {Path(f).name}")
    if len(hycom_files) > 3:
        print(f"    ... and {len(hycom_files) - 3} more")
    
    return hycom_files


def load_hycom_currents(nc_file: str) -> Dict:
    """
    Load HYCOM u, v velocity fields from NetCDF file.
    
    Returns dict with:
    - u: eastward velocity [m/s]
    - v: northward velocity [m/s]
    - lats: latitude coordinates
    - lons: longitude coordinates
    """
    ds = xr.open_dataset(nc_file)
    
    # Extract coordinates
    if 'lat' in ds.coords:
        lats = ds['lat'].values
    else:
        raise ValueError(f"No 'lat' coordinate in {nc_file}")
    
    if 'lon' in ds.coords:
        lons = ds['lon'].values
    else:
        raise ValueError(f"No 'lon' coordinate in {nc_file}")
    
    # Extract velocity fields
    if 'water_u' not in ds.data_vars:
        raise ValueError(f"No 'water_u' in {nc_file}")
    if 'water_v' not in ds.data_vars:
        raise ValueError(f"No 'water_v' in {nc_file}")
    
    # Get surface velocity - handle different possible shapes
    # Could be [time, lat, lon] or [time, depth, lat, lon]
    u_data = ds['water_u'].values
    v_data = ds['water_v'].values
    
    if u_data.ndim == 4:
        # [time, depth, lat, lon] → take first time and first depth
        u = u_data[0, 0, :, :]
        v = v_data[0, 0, :, :]
    elif u_data.ndim == 3:
        # [time, lat, lon] → take first time
        u = u_data[0, :, :]
        v = v_data[0, :, :]
    else:
        raise ValueError(f"Unexpected u_data shape: {u_data.shape}")
    
    ds.close()
    
    return {
        'u': u,
        'v': v,
        'lats': lats,
        'lons': lons
    }


def lagrangian_drift(seed_points: List[Tuple[float, float]],
                     hycom_files: List[str],
                     n_days: int = 8,
                     dt_hours: float = 6) -> List[Dict]:
    """
    Simulate Lagrangian drift for all seed points using HYCOM currents.
    
    Args:
        seed_points: List of (lat, lon) starting points
        hycom_files: List of HYCOM NetCDF file paths (time-ordered)
        n_days: Total simulation duration in days
        dt_hours: Time step in hours
    
    Returns:
        List of trajectory dicts with 'id', 'lats', 'lons'
    """
    print(f"\nSimulating Lagrangian drift...")
    print(f"  Duration: {n_days} days")
    print(f"  Time step: {dt_hours} hours")
    print(f"  Expected steps: {int(n_days * 24 / dt_hours)}")
    print(f"  Available HYCOM files: {len(hycom_files)}")
    
    dt_seconds = dt_hours * 3600  # Convert to seconds
    
    trajectories = []
    
    # For each seed point
    for seed_idx, (start_lat, start_lon) in enumerate(seed_points):
        print(f"\n  Processing seed point {seed_idx + 1}/{len(seed_points)}: ({start_lat:.2f}, {start_lon:.2f})")
        
        # Initialize trajectory
        trajectory_lats = [start_lat]
        trajectory_lons = [start_lon]
        
        current_lat = start_lat
        current_lon = start_lon
        current_file_idx = 0
        
        # Load first HYCOM file
        try:
            hycom_data = load_hycom_currents(hycom_files[current_file_idx])
        except Exception as e:
            print(f"    Error loading HYCOM: {e}")
            continue
        
        # Create interpolators for u and v
        interp_u = RegularGridInterpolator(
            (hycom_data['lats'], hycom_data['lons']),
            hycom_data['u'],
            bounds_error=False,
            fill_value=0.0
        )
        interp_v = RegularGridInterpolator(
            (hycom_data['lats'], hycom_data['lons']),
            hycom_data['v'],
            bounds_error=False,
            fill_value=0.0
        )
        
        n_steps = int(n_days * 24 / dt_hours)
        
        # Time stepping
        for step in range(1, n_steps + 1):
            # Get velocity at current position
            point = np.array([current_lat, current_lon])
            
            u = float(interp_u(point))  # m/s, eastward
            v = float(interp_v(point))  # m/s, northward
            
            # Update position
            # dlat = v * dt_seconds / 111000 (111 km per degree latitude)
            dlat = v * dt_seconds / 111000
            
            # dlon = u * dt_seconds / (111000 * cos(lat))
            lat_radians = np.radians(current_lat)
            dlon = u * dt_seconds / (111000 * np.cos(lat_radians))
            
            current_lat += dlat
            current_lon += dlon
            
            # Clamp to bounds
            current_lat = np.clip(current_lat, 0, 35)
            current_lon = np.clip(current_lon, 50, 105)
            
            # Record position
            trajectory_lats.append(current_lat)
            trajectory_lons.append(current_lon)
            
            # Periodically reload HYCOM file (e.g., every 4 steps ≈ 1 day)
            if step % 4 == 0 and step < n_steps:
                next_file_idx = min(current_file_idx + 1, len(hycom_files) - 1)
                if next_file_idx != current_file_idx:
                    try:
                        hycom_data = load_hycom_currents(hycom_files[next_file_idx])
                        interp_u = RegularGridInterpolator(
                            (hycom_data['lats'], hycom_data['lons']),
                            hycom_data['u'],
                            bounds_error=False,
                            fill_value=0.0
                        )
                        interp_v = RegularGridInterpolator(
                            (hycom_data['lats'], hycom_data['lons']),
                            hycom_data['v'],
                            bounds_error=False,
                            fill_value=0.0
                        )
                        current_file_idx = next_file_idx
                    except Exception as e:
                        print(f"    Warning: Could not load HYCOM file {next_file_idx}: {e}")
        
        # Store trajectory
        trajectory_dict = {
            "id": seed_idx,
            "lat_start": start_lat,
            "lon_start": start_lon,
            "lat_end": current_lat,
            "lon_end": current_lon,
            "lats": trajectory_lats,
            "lons": trajectory_lons,
            "n_steps": len(trajectory_lats)
        }
        
        trajectories.append(trajectory_dict)
        print(f"    ✓ Trajectory complete: {start_lat:.2f},{start_lon:.2f} → {current_lat:.2f},{current_lon:.2f}")
    
    return trajectories


def save_trajectories(trajectories: List[Dict],
                     output_path: str = "./outputs/drift_trajectories.json") -> None:
    """Save Lagrangian trajectories to JSON."""
    print(f"\nSaving trajectories to {output_path}...")
    
    data = {
        "n_trajectories": len(trajectories),
        "trajectories": trajectories,
        "metadata": {
            "description": "Lagrangian ghost gear drift trajectories using HYCOM currents",
            "simulation_days": 8,
            "time_step_hours": 6
        }
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    print(f"  ✓ Saved {len(trajectories)} trajectories")


def print_summary(trajectories: List[Dict]) -> None:
    """Print trajectory summary."""
    print(f"\n{'='*70}")
    print("LAGRANGIAN DRIFT SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"Total trajectories: {len(trajectories)}")
    print(f"\nTrajectory Details:")
    
    for traj in trajectories:
        displacement = np.sqrt(
            (traj['lat_end'] - traj['lat_start'])**2 +
            (traj['lon_end'] - traj['lon_start'])**2
        )
        print(f"  Trajectory {traj['id']}:")
        print(f"    Start: ({traj['lat_start']:.3f}, {traj['lon_start']:.3f})")
        print(f"    End:   ({traj['lat_end']:.3f}, {traj['lon_end']:.3f})")
        print(f"    Displacement: {displacement:.3f}° (~{displacement*111:.0f} km)")
        print(f"    Steps: {traj['n_steps']}")


def main() -> int:
    """Main pipeline."""
    print("="*70)
    print("LAGRANGIAN DRIFT ENGINE")
    print("="*70)
    
    try:
        # 1. Load seed points
        seed_points = load_seed_points()
        
        # 2. Find HYCOM files
        hycom_files = find_hycom_files()
        
        # 3. Simulate drift
        trajectories = lagrangian_drift(seed_points, hycom_files, n_days=8, dt_hours=6)
        
        # 4. Save trajectories
        save_trajectories(trajectories)
        
        # 5. Print summary
        print_summary(trajectories)
        
        print(f"\n{'='*70}")
        print(f"Simulated {len(trajectories)} ghost gear drift paths over 8 days")
        print(f"{'='*70}\n")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
