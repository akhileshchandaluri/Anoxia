"""
Fuse ghost gear drift trajectories with hypoxia predictions.

Pipeline:
1. Load hypoxia prediction grid (dz_predictions.json)
2. Load ghost gear seed points (gfw_seed_points.json)
3. Load or simulate ocean currents (HYCOM or random drift)
4. Simulate Lagrangian drift trajectories from each seed point
5. Compute drift density at each grid cell
6. Fuse with hypoxia probabilities: risk = hyp_prob × drift_density
7. Save drift paths and fusion risk map

Output:
- Drift trajectories: ./outputs/drift_paths.json
- Risk map: ./outputs/fusion_risk_map.json

Goal:
Identify regions where ghost gear drift overlaps with hypoxic zones.
"""

import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr


def load_hypoxia_grid(json_path: str = "./outputs/dz_predictions.json") -> Dict:
    """Load hypoxia predictions."""
    print(f"Loading hypoxia grid from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"  Loaded {len(data['dz_probs'])} grid points")
    return data


def load_seed_points(json_path: str = "./data/gfw/gfw_seed_points.json") -> List[Tuple[float, float]]:
    """Load ghost gear seed points."""
    print(f"\nLoading seed points from {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle both flat list and nested structure
    if 'seed_points' in data:
        seed_points = data['seed_points']
    else:
        seed_points = data.get('seeds', [])
    
    # Convert to tuples if needed
    seed_points = [tuple(pt) if isinstance(pt, list) else pt for pt in seed_points]
    
    print(f"  Loaded {len(seed_points)} seed points")
    for i, (lat, lon) in enumerate(seed_points[:3]):
        print(f"    {i+1}. ({lat:.2f}, {lon:.2f})")
    if len(seed_points) > 3:
        print(f"    ... and {len(seed_points) - 3} more")
    
    return seed_points


def load_hycom_currents() -> Dict:
    """
    Attempt to load HYCOM ocean current data.
    
    Returns dict with u, v velocity fields or None if not found.
    """
    print(f"\nSearching for HYCOM current data...")
    
    hycom_files = glob.glob("./data/hycom/hycom_*.nc")
    
    if not hycom_files:
        print(f"  No HYCOM files found - will use random drift instead")
        return None
    
    # Use most recent file
    latest_file = sorted(hycom_files)[-1]
    print(f"  Found HYCOM file: {Path(latest_file).name}")
    
    try:
        ds = xr.open_dataset(latest_file)
        
        # Extract velocity fields
        if 'water_u' in ds.data_vars and 'water_v' in ds.data_vars:
            u = ds['water_u'].values
            v = ds['water_v'].values
            lats = ds['lat'].values
            lons = ds['lon'].values
            
            print(f"  Loaded HYCOM velocities: u shape {u.shape}, v shape {v.shape}")
            
            return {
                'type': 'hycom',
                'u': u,
                'v': v,
                'lats': lats,
                'lons': lons
            }
        else:
            print(f"  HYCOM file missing water_u or water_v - using random drift")
            return None
            
    except Exception as e:
        print(f"  Error loading HYCOM: {e}")
        print(f"  Will use random drift instead")
        return None


def simulate_drift(seed_point: Tuple[float, float], 
                   n_steps: int = 15,
                   hycom_data: Dict = None,
                   random_seed: int = None) -> List[Tuple[float, float]]:
    """
    Simulate Lagrangian drift from a seed point.
    
    Args:
        seed_point: (lat, lon) starting point
        n_steps: Number of drift steps (10-20)
        hycom_data: HYCOM velocity data or None for random drift
        random_seed: For reproducibility
    
    Returns:
        List of (lat, lon) trajectory points
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    trajectory = [seed_point]
    lat, lon = seed_point
    
    # Drift parameters
    if hycom_data is None:
        # Random drift: small step size (~0.1° per step = ~10 km)
        dt = 0.1  # degrees per step
        
        for _ in range(n_steps):
            # Random walk with slight preferential direction
            dlat = np.random.normal(0, dt * 0.5)
            dlon = np.random.normal(0, dt * 0.5)
            
            lat += dlat
            lon += dlon
            
            # Boundaries (keep within study region)
            lat = np.clip(lat, 0, 30)
            lon = np.clip(lon, 55, 100)
            
            trajectory.append((lat, lon))
    else:
        # HYCOM-based drift (simplified - use mean velocity)
        u = hycom_data['u']
        v = hycom_data['v']
        
        # Use mean velocity (simplified - ideally would interpolate)
        u_mean = np.nanmean(u)
        v_mean = np.nanmean(v)
        
        dt = 0.01  # smaller step for HYCOM-based drift
        
        for _ in range(n_steps):
            # Add some random walk + mean current
            dlat = v_mean * dt + np.random.normal(0, dt * 0.3)
            dlon = u_mean * dt + np.random.normal(0, dt * 0.3)
            
            lat += dlat
            lon += dlon
            
            # Boundaries
            lat = np.clip(lat, 0, 30)
            lon = np.clip(lon, 55, 100)
            
            trajectory.append((lat, lon))
    
    return trajectory


def simulate_all_drifts(seed_points: List[Tuple[float, float]],
                        hycom_data: Dict = None,
                        n_steps: int = 15) -> List[List[Tuple[float, float]]]:
    """
    Simulate drift trajectories from all seed points.
    
    Returns:
        List of trajectories (each trajectory is a list of lat/lon tuples)
    """
    print(f"\nSimulating drift trajectories...")
    print(f"  Parameters: {n_steps} steps per trajectory")
    
    if hycom_data:
        print(f"  Using HYCOM currents")
    else:
        print(f"  Using random drift (Brownian motion)")
    
    trajectories = []
    
    for i, seed_point in enumerate(seed_points):
        traj = simulate_drift(seed_point, n_steps=n_steps, hycom_data=hycom_data, 
                             random_seed=42 + i)  # Reproducible but varied
        trajectories.append(traj)
        
        if (i + 1) % max(1, len(seed_points) // 4) == 0:
            print(f"  Simulated {i + 1}/{len(seed_points)} trajectories")
    
    print(f"  ✓ Total trajectories: {len(trajectories)}")
    
    return trajectories


def compute_drift_density(trajectories: List[List[Tuple[float, float]]],
                         grid_lats: np.ndarray,
                         grid_lons: np.ndarray) -> np.ndarray:
    """
    Compute drift density on the prediction grid.
    
    For each grid cell, count how many trajectory points pass near it.
    """
    print(f"\nComputing drift density...")
    
    n_grid = len(grid_lats)
    density = np.zeros(n_grid, dtype=float)
    
    # For each trajectory
    for traj_idx, trajectory in enumerate(trajectories):
        # For each point in trajectory
        for lat, lon in trajectory:
            # Find nearest grid point
            distances = np.sqrt((grid_lats - lat) ** 2 + (grid_lons - lon) ** 2)
            nearest_idx = np.argmin(distances)
            
            # Increase density at nearest grid point
            density[nearest_idx] += 1.0
            
            # Also add contribution to nearby cells (within ~0.5°)
            nearby_mask = distances < 0.5
            nearby_indices = np.where(nearby_mask)[0]
            
            # Inverse distance weighting
            for idx in nearby_indices:
                if idx != nearest_idx:
                    dist = distances[idx]
                    if dist > 0:
                        density[idx] += 1.0 / (1.0 + dist) * 0.5
    
    # Normalize
    if density.max() > 0:
        density = density / density.max()  # Normalize to [0, 1]
    
    print(f"  ✓ Computed drift density")
    print(f"    Max density: {density.max():.4f}")
    print(f"    Mean density: {density.mean():.4f}")
    
    return density


def compute_fusion_risk(hypoxia_probs: np.ndarray,
                       drift_density: np.ndarray,
                       density_weight: float = 0.3) -> np.ndarray:
    """
    Fuse hypoxia probabilities with drift density.
    
    risk_score = hypoxia_prob + (drift_density × density_weight)
    
    Args:
        hypoxia_probs: Array of hypoxia probabilities [0, 1]
        drift_density: Array of drift density [0, 1]
        density_weight: How much to weight drift in final score
    
    Returns:
        Risk score array [0, 1]
    """
    print(f"\nFusing hypoxia and drift data...")
    
    # Compute fusion risk
    risk = hypoxia_probs + (drift_density * density_weight)
    
    # Normalize to [0, 1]
    risk = risk / risk.max() if risk.max() > 0 else risk
    
    print(f"  ✓ Computed fusion risk")
    print(f"    Min risk: {risk.min():.4f}")
    print(f"    Max risk: {risk.max():.4f}")
    print(f"    Mean risk: {risk.mean():.4f}")
    
    return risk


def save_drift_paths(trajectories: List[List[Tuple[float, float]]],
                    output_path: str = "./outputs/drift_paths.json") -> None:
    """Save drift trajectories to JSON."""
    print(f"\nSaving drift trajectories to {output_path}...")
    
    data = {
        "n_trajectories": len(trajectories),
        "trajectories": [
            {
                "id": i,
                "path": [[float(lat), float(lon)] for lat, lon in traj]
            }
            for i, traj in enumerate(trajectories)
        ],
        "metadata": {
            "description": "Lagrangian drift trajectories from ghost gear seed points",
            "n_points_per_trajectory": len(trajectories[0]) if trajectories else 0
        }
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    print(f"  ✓ Saved {len(trajectories)} trajectories")


def save_fusion_risk_map(grid_lats: np.ndarray,
                        grid_lons: np.ndarray,
                        risk: np.ndarray,
                        output_path: str = "./outputs/fusion_risk_map.json") -> None:
    """Save fusion risk map to JSON."""
    print(f"\nSaving fusion risk map to {output_path}...")
    
    data = {
        "lat": grid_lats.tolist(),
        "lon": grid_lons.tolist(),
        "risk": risk.tolist(),
        "metadata": {
            "n_points": len(risk),
            "risk_min": float(risk.min()),
            "risk_max": float(risk.max()),
            "risk_mean": float(risk.mean()),
            "description": "Fusion of hypoxia predictions and ghost gear drift density"
        }
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    print(f"  ✓ Saved risk map with {len(risk)} grid points")


def print_summary(trajectories: List[List[Tuple[float, float]]],
                 risk: np.ndarray,
                 threshold: float = 0.3) -> None:
    """Print summary statistics."""
    print(f"\n{'='*70}")
    print("DRIFT-HYPOXIA FUSION SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nDrift Trajectories:")
    print(f"  Total trajectories: {len(trajectories)}")
    print(f"  Points per trajectory: {len(trajectories[0]) if trajectories else 0}")
    print(f"  Total drift points: {sum(len(traj) for traj in trajectories)}")
    
    print(f"\nFusion Risk Map:")
    print(f"  Grid points: {len(risk)}")
    print(f"  Min risk: {risk.min():.4f}")
    print(f"  Max risk: {risk.max():.4f}")
    print(f"  Mean risk: {risk.mean():.4f}")
    print(f"  Std dev: {risk.std():.4f}")
    
    high_risk = np.sum(risk >= threshold)
    print(f"\nHigh-Risk Zones (risk ≥ {threshold}):")
    print(f"  Grid points: {high_risk} ({100*high_risk/len(risk):.1f}%)")
    
    # Risk percentiles
    print(f"\nRisk Distribution (Percentiles):")
    for pct in [10, 25, 50, 75, 90, 95]:
        val = np.percentile(risk, pct)
        print(f"  {pct}th: {val:.4f}")
    
    print(f"\n{'='*70}")


def main() -> int:
    """Main pipeline."""
    print("="*70)
    print("GHOST GEAR DRIFT × HYPOXIA ZONE FUSION")
    print("="*70)
    
    try:
        # 1. Load data
        hyp_data = load_hypoxia_grid()
        seed_points = load_seed_points()
        hycom_data = load_hycom_currents()
        
        # Extract grid info
        grid_lats = np.array(hyp_data['dz_lats'])
        grid_lons = np.array(hyp_data['dz_lons'])
        hyp_probs = np.array(hyp_data['dz_probs'])
        
        # 2. Simulate drift
        trajectories = simulate_all_drifts(seed_points, hycom_data=hycom_data, n_steps=15)
        
        # 3. Compute drift density
        drift_density = compute_drift_density(trajectories, grid_lats, grid_lons)
        
        # 4. Fuse with hypoxia
        risk = compute_fusion_risk(hyp_probs, drift_density, density_weight=0.5)
        
        # 5. Save outputs
        save_drift_paths(trajectories)
        save_fusion_risk_map(grid_lats, grid_lons, risk)
        
        # 6. Print summary
        print_summary(trajectories, risk, threshold=0.3)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
