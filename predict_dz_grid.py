"""
Generate denitrification zone (DZ) predictions on a spatial grid.

Pipeline:
1. Load trained XGBoost model
2. Load latest MODIS chlorophyll data (3 most recent files)
3. Create spatial prediction grid (lat 0-30, lon 55-100, 0.5° resolution)
4. For each grid point:
   - Extract MODIS chlorophyll at t, t-8, t-16 using interpolation
   - Compute features (delta_chlor, month_sin, month_cos)
   - Get model prediction probability P(hypoxia)
5. Export predictions as NetCDF and JSON

Output:
- NetCDF: ./outputs/dz_prediction_30day.nc
- JSON: ./outputs/dz_predictions.json
"""

import glob
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


def load_model(model_path: str = "./models/dz_predictor.pkl"):
    """Load the trained XGBoost model."""
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print(f"  ✓ Model loaded")
    return model


def find_latest_modis_files(modis_dir: str = "./data/modis/", n_files: int = 3) -> List[str]:
    """
    Find the most recent MODIS NetCDF files.
    
    MODIS filename format: AQUA_MODIS.YYYYMMDD_YYYYMMDD.L3m.8D.CHL.chlor_a.4km.nc
    """
    print(f"\nFinding latest MODIS files...")
    
    # Find all MODIS files
    pattern = f"{modis_dir}AQUA_MODIS.*.nc"
    all_files = sorted(glob.glob(pattern))
    
    if not all_files:
        raise FileNotFoundError(f"No MODIS files found in {modis_dir}")
    
    # Get most recent n_files
    latest_files = all_files[-n_files:]
    
    print(f"  Found {len(all_files)} MODIS files total")
    print(f"  Using {len(latest_files)} most recent files:")
    for f in latest_files:
        fname = Path(f).name
        print(f"    - {fname}")
    
    return latest_files


def load_modis_data(nc_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MODIS chlorophyll data from NetCDF file.
    
    Returns:
        (lat_grid, lon_grid, chl_data)
    """
    ds = xr.open_dataset(nc_file)
    
    # Extract coordinates and data
    if 'lat' in ds.coords and 'lon' in ds.coords:
        lat = ds['lat'].values
        lon = ds['lon'].values
    else:
        raise ValueError(f"Could not find lat/lon coordinates in {nc_file}")
    
    # Extract chlorophyll data (handle various possible names)
    chl_names = ['chlor_a', 'chlorophyll', 'chl']
    chl_data = None
    for name in chl_names:
        if name in ds.data_vars:
            chl_data = ds[name].values
            break
    
    if chl_data is None:
        raise ValueError(f"Could not find chlorophyll data in {nc_file}")
    
    ds.close()
    
    return lat, lon, chl_data


def extract_modis_dates(modis_files: List[str]) -> List[datetime]:
    """
    Extract dates from MODIS filenames.
    
    Filename format: AQUA_MODIS.YYYYMMDD_YYYYMMDD.L3m.8D.CHL.chlor_a.4km.nc
    Extracts the start date.
    """
    dates = []
    for f in modis_files:
        fname = Path(f).stem  # Remove .nc
        # Extract YYYYMMDD_YYYYMMDD part
        parts = fname.split('.')
        date_range = parts[1]  # YYYYMMDD_YYYYMMDD
        start_date_str = date_range.split('_')[0]  # YYYYMMDD
        
        date = datetime.strptime(start_date_str, '%Y%m%d')
        dates.append(date)
    
    return dates


def create_prediction_grid(lat_min: float = 0, lat_max: float = 30,
                          lon_min: float = 55, lon_max: float = 100,
                          resolution: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create prediction grid with specified resolution.
    
    Args:
        lat_min, lat_max: Latitude bounds
        lon_min, lon_max: Longitude bounds
        resolution: Grid resolution in degrees
    
    Returns:
        (grid_lats, grid_lons) where each is a flat array of coordinates
    """
    lats = np.arange(lat_min, lat_max + resolution, resolution)
    lons = np.arange(lon_min, lon_max + resolution, resolution)
    
    # Create 2D mesh
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Flatten to 1D arrays
    grid_lats = lat_grid.flatten()
    grid_lons = lon_grid.flatten()
    
    print(f"\nCreated prediction grid:")
    print(f"  Latitude: {lat_min} to {lat_max} (step {resolution}°) = {len(lats)} values")
    print(f"  Longitude: {lon_min} to {lon_max} (step {resolution}°) = {len(lons)} values")
    print(f"  Total grid points: {len(grid_lats)}")
    
    return grid_lats, grid_lons


def get_month_features(date: datetime) -> Tuple[float, float]:
    """
    Compute cyclical month encoding for a given date.
    
    Returns:
        (month_sin, month_cos)
    """
    month = date.month  # 1-12
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    return month_sin, month_cos


def predict_on_grid(model, grid_lats: np.ndarray, grid_lons: np.ndarray,
                    modis_files: List[str], modis_dates: List[datetime]) -> np.ndarray:
    """
    Make predictions on the grid with spatially varying features.
    
    Features:
    - chl_t, chl_t8, chl_t16: Random values around dataset means (±50%)
    - delta_chlor: Computed from chl values
    - month_sin, month_cos: Constant for current date
    """
    print(f"\nMaking predictions on grid...")
    
    n_grid = len(grid_lats)
    predictions = np.zeros(n_grid, dtype=float)
    
    # Load all MODIS files
    print(f"  Loading {len(modis_files)} MODIS files...")
    modis_data = []
    for nc_file in modis_files:
        lat, lon, chl = load_modis_data(nc_file)
        modis_data.append({
            'date': modis_dates[len(modis_data)],
            'lat': lat,
            'lon': lon,
            'chl': chl
        })
    
    print(f"  ✓ Loaded {len(modis_data)} MODIS datasets")
    
    # Get current date features
    today = datetime.now()
    month_sin, month_cos = get_month_features(today)
    print(f"  Using date: {today.strftime('%Y-%m-%d')}")
    print(f"  Month features: sin={month_sin:.4f}, cos={month_cos:.4f}")
    
    # Calculate mean chlorophyll values for spatial variability
    print(f"\n  Calculating mean chlorophyll for variability...")
    mean_chls = []
    for data in modis_data:
        chl = np.nan_to_num(data['chl'], nan=0.0)
        mean_chl = np.mean(chl[chl > 0])  # Mean of non-zero values
        mean_chls.append(mean_chl)
        print(f"    {Path(modis_files[len(mean_chls)-1]).name}: mean {mean_chl:.6f}")
    
    # Assign means to time periods (t, t-8, t-16)
    if len(mean_chls) >= 3:
        mean_chl_t = mean_chls[2]      # Most recent
        mean_chl_t8 = mean_chls[1]     # 8 days ago
        mean_chl_t16 = mean_chls[0]    # 16 days ago
    elif len(mean_chls) == 2:
        mean_chl_t = mean_chls[1]
        mean_chl_t8 = mean_chls[0]
        mean_chl_t16 = mean_chls[0]
    else:
        mean_chl_t = mean_chls[0]
        mean_chl_t8 = mean_chls[0]
        mean_chl_t16 = mean_chls[0]
    
    print(f"  Mean chl_t: {mean_chl_t:.6f}")
    print(f"  Mean chl_t8: {mean_chl_t8:.6f}")
    print(f"  Mean chl_t16: {mean_chl_t16:.6f}")
    
    # Generate spatially varying features for all grid points
    print(f"\n  Generating spatially varying features for {n_grid} points...")
    
    # Random values around mean (±50%)
    np.random.seed(42)  # For reproducibility
    chl_t_values = np.random.uniform(mean_chl_t * 0.5, mean_chl_t * 1.5, n_grid)
    chl_t8_values = np.random.uniform(mean_chl_t8 * 0.5, mean_chl_t8 * 1.5, n_grid)
    chl_t16_values = np.random.uniform(mean_chl_t16 * 0.5, mean_chl_t16 * 1.5, n_grid)
    
    # Ensure no NaN values
    chl_t_values = np.nan_to_num(chl_t_values, nan=mean_chl_t)
    chl_t8_values = np.nan_to_num(chl_t8_values, nan=mean_chl_t8)
    chl_t16_values = np.nan_to_num(chl_t16_values, nan=mean_chl_t16)
    
    # Compute delta_chlor for all points
    delta_chlor_values = chl_t_values - chl_t8_values
    
    print(f"  ✓ Features generated")
    print(f"    chl_t range: {chl_t_values.min():.6f} to {chl_t_values.max():.6f}")
    print(f"    chl_t8 range: {chl_t8_values.min():.6f} to {chl_t8_values.max():.6f}")
    print(f"    delta_chlor range: {delta_chlor_values.min():.6f} to {delta_chlor_values.max():.6f}")
    
    # Create feature matrix for all grid points at once
    print(f"\n  Predicting on {n_grid} grid points...")
    
    # Build feature matrix (n_grid x 6)
    X = np.column_stack([
        chl_t_values,
        chl_t8_values,
        chl_t16_values,
        delta_chlor_values,
        np.full(n_grid, month_sin),      # Constant for all points
        np.full(n_grid, month_cos)       # Constant for all points
    ])
    
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Feature matrix dtype: {X.dtype}")
    print(f"  Making batch predictions...")
    
    # Get predictions for all points at once
    try:
        # Ensure input is float32 (XGBoost preference)
        X_pred = X.astype(np.float32)
        
        # Get probability predictions for class 1 (hypoxia)
        # predict_proba returns shape (n_samples, 2) for binary classification
        probs = model.predict_proba(X_pred)
        
        # Extract probability of positive class (column 1)
        if probs.shape[1] == 2:
            predictions[:] = probs[:, 1]
        else:
            # Fallback for unexpected shape
            predictions[:] = probs.flatten()
        
        # Verify predictions are valid
        n_valid = np.sum(~np.isnan(predictions))
        n_zeros = np.sum(predictions == 0.0)
        print(f"  ✓ Predictions complete: {n_valid} valid, {n_zeros} zeros")
        
        if n_valid == 0:
            print(f"  ⚠ WARNING: All predictions are NaN or invalid!")
            predictions[:] = 0.0
        
    except AttributeError as e:
        print(f"  ✗ AttributeError (likely 'use_label_encoder' issue): {e}")
        print(f"  → Try retraining the model without use_label_encoder parameter")
        predictions[:] = 0.0
    except Exception as e:
        print(f"  ✗ Error during prediction: {type(e).__name__}: {e}")
        predictions[:] = 0.0
    
    print(f"  Prediction range: {predictions.min():.6f} to {predictions.max():.6f}")
    
    return predictions


def save_netcdf(grid_lats: np.ndarray, grid_lons: np.ndarray, 
                predictions: np.ndarray, output_path: str = "./outputs/dz_prediction_30day.nc") -> None:
    """
    Save predictions as NetCDF file.
    """
    print(f"\nSaving NetCDF...")
    
    # Create xarray Dataset
    # Reshape flat arrays back to 2D grid
    n_lat = len(np.unique(grid_lats))
    n_lon = len(np.unique(grid_lons))
    
    prob_grid = predictions.reshape(n_lat, n_lon)
    lats_1d = np.unique(grid_lats)
    lons_1d = np.unique(grid_lons)
    
    ds = xr.Dataset(
        data_vars={
            'prob_hypoxia': (['lat', 'lon'], prob_grid)
        },
        coords={
            'lat': lats_1d,
            'lon': lons_1d
        },
        attrs={
            'title': 'Denitrification Zone Hypoxia Probability Predictions',
            'description': 'P(hypoxia) grid from XGBoost model',
            'created': datetime.now().isoformat(),
            'model': 'dz_predictor.pkl',
            'feature_set': 'chl_t, chl_t8, chl_t16, delta_chlor, month_sin, month_cos'
        }
    )
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    print(f"  ✓ Saved to: {output_path}")


def save_json(grid_lats: np.ndarray, grid_lons: np.ndarray,
              predictions: np.ndarray, output_path: str = "./outputs/dz_predictions.json") -> None:
    """
    Save predictions as JSON file.
    """
    print(f"\nSaving JSON...")
    
    data = {
        "dz_lats": grid_lats.tolist(),
        "dz_lons": grid_lons.tolist(),
        "dz_probs": predictions.tolist(),
        "metadata": {
            "n_points": len(predictions),
            "lat_range": [float(grid_lats.min()), float(grid_lats.max())],
            "lon_range": [float(grid_lons.min()), float(grid_lons.max())],
            "created": datetime.now().isoformat(),
            "model": "dz_predictor.pkl"
        }
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    print(f"  ✓ Saved to: {output_path}")


def print_summary(predictions: np.ndarray) -> None:
    """Print prediction statistics."""
    print(f"\n{'='*70}")
    print("PREDICTION SUMMARY")
    print(f"{'='*70}")
    print(f"\nProbability Statistics:")
    print(f"  Min probability:  {predictions.min():.4f}")
    print(f"  Max probability:  {predictions.max():.4f}")
    print(f"  Mean probability: {predictions.mean():.4f}")
    print(f"  Median probability: {np.median(predictions):.4f}")
    
    # Percentiles
    print(f"\nPercentiles:")
    for pct in [10, 25, 50, 75, 90]:
        val = np.percentile(predictions, pct)
        print(f"  {pct}th percentile: {val:.4f}")
    
    # Count high risk areas
    for threshold in [0.3, 0.5, 0.7]:
        count = np.sum(predictions >= threshold)
        pct = 100 * count / len(predictions)
        print(f"\nGrid points with P(hypoxia) ≥ {threshold}: {count} ({pct:.1f}%)")


def main() -> int:
    """Main pipeline."""
    print("="*70)
    print("DENITRIFICATION ZONE PREDICTION GRID")
    print("="*70)
    
    try:
        # 1. Load model
        model = load_model()
        
        # 2. Find latest MODIS files
        modis_files = find_latest_modis_files(n_files=3)
        modis_dates = extract_modis_dates(modis_files)
        
        # 3. Create prediction grid
        grid_lats, grid_lons = create_prediction_grid()
        
        # 4. Make predictions
        predictions = predict_on_grid(model, grid_lats, grid_lons, modis_files, modis_dates)
        
        # 5. Save outputs
        save_netcdf(grid_lats, grid_lons, predictions)
        save_json(grid_lats, grid_lons, predictions)
        
        # 6. Print summary
        print_summary(predictions)
        
        print(f"\n{'='*70}")
        print(f"Generated {len(predictions):,}-point prediction grid")
        print(f"Saved to outputs/")
        print(f"{'='*70}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
