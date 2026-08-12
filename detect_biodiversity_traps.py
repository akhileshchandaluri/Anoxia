"""
Biodiversity Traps Detection: Identify where ghost gear trajectories end in high-hypoxia zones.

A "biodiversity trap" occurs when a drifting ghost gear trajectory terminates in a region
where dissolved oxygen depletion (hypoxia) is severe (P > 0.60). These regions are ecological
death zones that trap mobile fauna, creating local biodiversity hotspots of mortality.
"""

import json
import numpy as np
import xarray as xr
from scipy.spatial import ConvexHull
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Point, Polygon


def detect_biodiversity_traps(dz_nc_path, trajectories, threshold=0.05):
    """
    Detect biodiversity traps: locations where ghost gear trajectories end in high-hypoxia zones.
    
    A trap occurs when a trajectory's final position falls within a region where P(hypoxia) > threshold.
    
    Parameters:
    -----------
    dz_nc_path : str
        Path to dead zone prediction NetCDF file
    trajectories : list
        List of trajectory dicts from drift_trajectories.json
        Expected keys: 'id', 'lat_end', 'lon_end'
    threshold : float
        Hypoxia probability threshold for high-risk zones (default: 0.05 = 5%)
    
    Returns:
    --------
    list : List of biodiversity trap dicts with coordinates, hypoxia probability, and severity
    """
    
    # ============================================================================
    # 1. Load NetCDF Dataset
    # ============================================================================
    ds = xr.open_dataset(dz_nc_path)
    print(f"  Loading NetCDF: {dz_nc_path}")
    print(f"  Dataset variables: {list(ds.data_vars)}")
    print(f"  Dataset coords: {list(ds.coords)}")
    
    # ============================================================================
    # 2. Extract lat, lon grids
    # ============================================================================
    # Try common naming conventions
    if 'latitude' in ds.coords:
        lat = ds['latitude'].values
    elif 'lat' in ds.coords:
        lat = ds['lat'].values
    else:
        raise KeyError("Neither 'latitude' nor 'lat' found in dataset coordinates")
    
    if 'longitude' in ds.coords:
        lon = ds['longitude'].values
    elif 'lon' in ds.coords:
        lon = ds['lon'].values
    else:
        raise KeyError("Neither 'longitude' nor 'lon' found in dataset coordinates")
    
    print(f"  Grid shape: lat={len(lat)}, lon={len(lon)}")
    
    # ============================================================================
    # 3. Extract probability data
    # ============================================================================
    if 'p_hypoxia' in ds.data_vars:
        p_hypoxia_var = ds['p_hypoxia']
    elif 'probability' in ds.data_vars:
        p_hypoxia_var = ds['probability']
    else:
        # Try first non-coordinate data variable
        data_vars = [v for v in ds.data_vars if v not in ['lat', 'lon', 'latitude', 'longitude']]
        if data_vars:
            p_hypoxia_var = ds[data_vars[0]]
            print(f"  Using variable: {data_vars[0]}")
        else:
            raise KeyError("Could not find probability data in dataset")
    
    p_hypoxia = p_hypoxia_var.values
    
    # Handle 3D arrays (might have time dimension)
    if p_hypoxia.ndim == 3:
        # Assume first dimension is time; take first timestep
        p_hypoxia = p_hypoxia[0]
        print(f"  3D probability array detected; using first timestep")
    elif p_hypoxia.ndim != 2:
        raise ValueError(f"Unexpected p_hypoxia shape: {p_hypoxia.shape}")
    
    print(f"  Probability shape: {p_hypoxia.shape}")
    
    # ============================================================================
    # 4. Identify high-risk hypoxic zones (P > threshold)
    # ============================================================================
    high_risk_mask = p_hypoxia > threshold
    high_risk_indices = np.argwhere(high_risk_mask)  # shape (N, 2): [(i, j), ...]
    
    print(f"  High-risk cells (P > {threshold:.4f}): {len(high_risk_indices)}")
    print(f"  Probability range: [{p_hypoxia.min():.4f}, {p_hypoxia.max():.4f}]")
    
    if len(high_risk_indices) == 0:
        print("  ⚠ No high-risk hypoxic zones found")
        ds.close()
        return []
    
    # ============================================================================
    # 5. Convert grid indices to lat/lon coordinates
    # ============================================================================
    # Handle 1D and 2D coordinate arrays
    if lat.ndim == 1:
        lat_1d = lat
    else:
        lat_1d = lat[:, 0]
    
    if lon.ndim == 1:
        lon_1d = lon
    else:
        lon_1d = lon[0, :]
    
    high_risk_coords = np.array([
        [lat_1d[idx[0]], lon_1d[idx[1]]] for idx in high_risk_indices
    ])
    
    print(f"  High-risk coords range: lat [{high_risk_coords[:, 0].min():.2f}, {high_risk_coords[:, 0].max():.2f}], "
          f"lon [{high_risk_coords[:, 1].min():.2f}, {high_risk_coords[:, 1].max():.2f}]")
    
    # ============================================================================
    # 6. Build ConvexHull from high-risk cells
    # ============================================================================
    trap_polygon = None
    
    if len(high_risk_coords) < 3:
        print(f"  Warning: Only {len(high_risk_coords)} high-risk cells (need >= 3 for ConvexHull)")
    else:
        try:
            hull = ConvexHull(high_risk_coords)
            hull_points = high_risk_coords[hull.vertices]
            trap_polygon = Polygon(hull_points)
            print(f"  ConvexHull created: {len(hull.vertices)} vertices")
        except Exception as e:
            print(f"  ConvexHull failed: {e}")
            trap_polygon = None
    
    # ============================================================================
    # 7. Create bilinear interpolator for hypoxia probability
    # ============================================================================
    interp_func = RegularGridInterpolator(
        (lat_1d, lon_1d),
        p_hypoxia,
        bounds_error=False,
        fill_value=np.nan
    )
    
    # ============================================================================
    # 8. Check each trajectory for biodiversity traps
    # ============================================================================
    traps = []
    
    for traj in trajectories:
        # Get final position
        final_lat = traj['lat_end']
        final_lon = traj['lon_end']
        final_point = Point(final_lat, final_lon)
        
        # Check if final position is inside hypoxic zone
        is_trapped = False
        
        if trap_polygon is not None:
            # Use ConvexHull polygon if available
            is_trapped = trap_polygon.contains(final_point)
        else:
            # Manual check: is point within distance threshold of any high-risk cell?
            distances = np.sqrt(
                (high_risk_coords[:, 0] - final_lat)**2 + 
                (high_risk_coords[:, 1] - final_lon)**2
            )
            is_trapped = np.min(distances) < 0.5  # Within 0.5° of any high-risk cell
        
        # ============================================================================
        # 9. If trapped, compute severity and create trap record
        # ============================================================================
        if is_trapped:
            # Get P(hypoxia) at final position via bilinear interpolation
            interp_point = np.array([[final_lat, final_lon]])
            p_hypoxia_final = interp_func(interp_point)[0]
            
            # Handle NaN from interpolation (out of bounds)
            if np.isnan(p_hypoxia_final):
                # Find nearest grid point
                distances_to_all = np.sqrt(
                    (lat_1d[:, np.newaxis] - final_lat)**2 + 
                    (lon_1d[np.newaxis, :] - final_lon)**2
                )
                nearest_idx = np.unravel_index(np.argmin(distances_to_all), distances_to_all.shape)
                p_hypoxia_final = p_hypoxia[nearest_idx]
            
            # Compute severity: P(hypoxia) * mortality multiplier
            severity = p_hypoxia_final * 3.4
            
            # Create trap dict
            trap = {
                "lat": float(final_lat),
                "lon": float(final_lon),
                "p_hypoxia": float(p_hypoxia_final),
                "severity": float(severity),
                "window_days": 8,
                "trajectory_id": int(traj['id'])
            }
            
            traps.append(trap)
    
    ds.close()
    return traps


def main():
    """Main execution: detect traps, save results, and print summary."""
    
    print("\n" + "="*70)
    print("BIODIVERSITY TRAPS DETECTION")
    print("="*70)
    
    # ========================================================================
    # 1. Load drift trajectories
    # ========================================================================
    print("\n1. Loading drift trajectories...")
    with open("./outputs/drift_trajectories.json", "r") as f:
        trajectory_data = json.load(f)
        trajectories = trajectory_data['trajectories']
    print(f"   Loaded {len(trajectories)} trajectories")
    
    # ========================================================================
    # 2. Detect biodiversity traps
    # ========================================================================
    print("\n2. Detecting biodiversity traps...")
    traps = detect_biodiversity_traps(
        "./outputs/dz_prediction_30day.nc",
        trajectories,
        threshold=0.05  # 5% hypoxia probability threshold
    )
    
    # ========================================================================
    # 3. Save results
    # ========================================================================
    print("\n3. Saving results...")
    output = {
        "n_traps": len(traps),
        "traps": traps,
        "metadata": {
            "threshold_p_hypoxia": 0.05,
            "severity_multiplier": 3.4,
            "detection_method": "ConvexHull + bilinear interpolation",
            "total_trajectories": len(trajectories)
        }
    }
    
    with open("./outputs/traps.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"   Saved to ./outputs/traps.json")
    
    # ========================================================================
    # 4. Print summary
    # ========================================================================
    print("\n" + "="*70)
    if traps:
        print(f"✓ Found {len(traps)} biodiversity traps")
        print("\nTrap Summary:")
        for i, trap in enumerate(traps, 1):
            print(f"\n  Trap {i}:")
            print(f"    Location: ({trap['lat']:.2f}°N, {trap['lon']:.2f}°E)")
            print(f"    P(hypoxia): {trap['p_hypoxia']:.4f}")
            print(f"    Severity: {trap['severity']:.4f}")
            print(f"    Trajectory ID: {trap['trajectory_id']}")
    else:
        print("✗ No biodiversity traps detected")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
