"""
Build feature matrix from Argo profiles and MODIS chlorophyll data.

Process:
1. Load Argo profiles from ./data/argo/argo_profiles.json
2. For each profile, find MODIS chlorophyll at:
   - date t (same day)
   - date t-8 (8 days before)
   - date t-16 (16 days before)
3. Use RegularGridInterpolator for spatial interpolation
4. Compute time-based features (month_sin, month_cos)
5. Create label: 1 if DO < 2.0 else 0
6. Save feature matrix to ./data/argo/feature_matrix.parquet

Output columns:
  [lat, lon, chl_t, chl_t8, chl_t16, delta_chlor, month_sin, month_cos, label]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

warnings.filterwarnings("ignore")


def _find_modis_file_for_date(date: datetime, modis_dir: Path) -> Optional[Path]:
    """
    Find MODIS file that covers the given date.
    MODIS files are 8-day composites; filename format: YYYYMMDD_YYYYMMDD
    """
    target_date = date.date()
    
    for modis_file in sorted(modis_dir.glob("AQUA_MODIS.*.nc")):
        # Extract date range from filename: AQUA_MODIS.20240329_20240405.L3m.8D...
        parts = modis_file.stem.split(".")
        if len(parts) < 2:
            continue
        
        date_range = parts[1]
        if "_" not in date_range:
            continue
        
        start_str, end_str = date_range.split("_")
        try:
            file_start = datetime.strptime(start_str, "%Y%m%d").date()
            file_end = datetime.strptime(end_str, "%Y%m%d").date()
            
            # Check if target date is within range
            if file_start <= target_date <= file_end:
                return modis_file
        except ValueError:
            continue
    
    return None


def _load_modis_data(modis_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MODIS NetCDF and return (lat, lon, chlor_a).
    Returns 1D lat, 1D lon, and 2D chlorophyll data.
    """
    ds = xr.open_dataset(modis_file)
    
    lat = ds["lat"].values
    lon = ds["lon"].values
    chl = ds["chlor_a"].values
    
    ds.close()
    
    return lat, lon, chl


def _interpolate_chlor(
    lat_point: float,
    lon_point: float,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    chl_data: np.ndarray,
) -> Optional[float]:
    """
    Interpolate chlorophyll at a point using RegularGridInterpolator.
    Returns None if interpolation fails or point is outside bounds.
    """
    try:
        # Handle global wraparound for longitude
        # MODIS lat: [-90, 90], lon: [-180, 180]
        # Normalize point longitude to match grid
        
        # Create interpolator
        interp = RegularGridInterpolator(
            (lat_grid, lon_grid),
            chl_data,
            bounds_error=False,
            fill_value=np.nan,
        )
        
        # Evaluate at point
        value = interp([lat_point, lon_point])[0]
        
        # Return None if extrapolated (NaN)
        if np.isnan(value):
            return None
        
        return float(value)
    
    except Exception as e:
        return None


def _get_month_features(date: datetime) -> Tuple[float, float]:
    """
    Compute cyclical month encoding: month_sin, month_cos.
    """
    month = date.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    return float(month_sin), float(month_cos)


def build_feature_matrix(
    argo_file: Path,
    modis_dir: Path,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build feature matrix from Argo profiles and MODIS data.
    """
    
    # Load Argo profiles
    with open(argo_file) as f:
        argo_profiles = json.load(f)
    
    if verbose:
        print(f"Loaded {len(argo_profiles)} Argo profiles", file=sys.stderr)
    
    # Cache for loaded MODIS files to avoid repeated disk I/O
    modis_cache: Dict[Path, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    
    rows = []
    skipped = 0
    
    for i, profile in enumerate(argo_profiles):
        if verbose and (i + 1) % 100 == 0:
            print(f"Processing profile {i + 1}/{len(argo_profiles)}", file=sys.stderr)
        
        lat = profile["lat"]
        lon = profile["lon"]
        date_str = profile["date"]  # YYYY-MM-DD
        do_value = profile["do"]
        
        try:
            profile_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            skipped += 1
            continue
        
        # Find MODIS data for three timestamps
        date_t = profile_date
        date_t8 = profile_date - timedelta(days=8)
        date_t16 = profile_date - timedelta(days=16)
        
        chl_t = None
        chl_t8 = None
        chl_t16 = None
        
        # Load and interpolate for each date
        for target_date, date_label in [(date_t, "t"), (date_t8, "t8"), (date_t16, "t16")]:
            modis_file = _find_modis_file_for_date(target_date, modis_dir)
            
            if modis_file is None:
                if verbose and date_label == "t":
                    print(
                        f"  No MODIS data for {target_date.date()} ({date_label})",
                        file=sys.stderr,
                    )
                continue
            
            # Load from cache or disk
            if modis_file not in modis_cache:
                lat_grid, lon_grid, chl_data = _load_modis_data(modis_file)
                modis_cache[modis_file] = (lat_grid, lon_grid, chl_data)
            else:
                lat_grid, lon_grid, chl_data = modis_cache[modis_file]
            
            # Interpolate
            chl_val = _interpolate_chlor(lat, lon, lat_grid, lon_grid, chl_data)
            
            if date_label == "t":
                chl_t = chl_val
            elif date_label == "t8":
                chl_t8 = chl_val
            elif date_label == "t16":
                chl_t16 = chl_val
        
        # Skip if we don't have at least chl_t and chl_t8
        if chl_t is None or chl_t8 is None:
            skipped += 1
            continue
        
        # Compute delta chlorophyll
        delta_chlor = chl_t - chl_t8
        
        # Compute time features
        month_sin, month_cos = _get_month_features(profile_date)
        
        # Create label
        label = 1 if do_value < 2.0 else 0
        
        # Append row
        rows.append({
            "lat": lat,
            "lon": lon,
            "chl_t": chl_t,
            "chl_t8": chl_t8,
            "chl_t16": chl_t16,
            "delta_chlor": delta_chlor,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "label": label,
        })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Drop rows with NaN
    initial_rows = len(df)
    df = df.dropna()
    dropped = initial_rows - len(df)
    
    if verbose:
        print(f"Skipped {skipped} profiles (missing MODIS data)", file=sys.stderr)
        print(f"Dropped {dropped} rows with NaN values", file=sys.stderr)
        print(f"Final feature matrix: {len(df)} rows", file=sys.stderr)
    
    return df


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build feature matrix from Argo and MODIS data.")
    p.add_argument(
        "--argo",
        type=Path,
        default=Path("./data/argo/argo_profiles.json"),
        help="Path to Argo profiles JSON (default: ./data/argo/argo_profiles.json)",
    )
    p.add_argument(
        "--modis",
        type=Path,
        default=Path("./data/modis"),
        help="Directory with MODIS NetCDF files (default: ./data/modis)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("./data/argo/feature_matrix.parquet"),
        help="Output parquet file (default: ./data/argo/feature_matrix.parquet)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    args = p.parse_args(argv)
    
    # Verify inputs exist
    if not args.argo.exists():
        print(f"Error: Argo file not found: {args.argo}", file=sys.stderr)
        return 1
    
    if not args.modis.exists():
        print(f"Error: MODIS directory not found: {args.modis}", file=sys.stderr)
        return 1
    
    # Build feature matrix
    df = build_feature_matrix(args.argo, args.modis, verbose=args.verbose)
    
    if len(df) == 0:
        print("Error: No valid feature matrix rows created", file=sys.stderr)
        return 1
    
    # Create output directory
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    df.to_parquet(args.out, index=False, compression="snappy")
    
    # Compute statistics
    n_rows = len(df)
    n_hypoxic = (df["label"] == 1).sum()
    n_oxic = (df["label"] == 0).sum()
    pct_hypoxic = 100 * n_hypoxic / n_rows if n_rows > 0 else 0
    pct_oxic = 100 * n_oxic / n_rows if n_rows > 0 else 0
    
    # Print summary
    print(
        f"Built feature matrix with {n_rows} rows, "
        f"{pct_hypoxic:.1f}% hypoxic, {pct_oxic:.1f}% oxic"
    )
    
    if args.verbose:
        print(f"\nFeature matrix shape: {df.shape}", file=sys.stderr)
        print(f"Columns: {list(df.columns)}", file=sys.stderr)
        print(f"\nData types:", file=sys.stderr)
        print(df.dtypes, file=sys.stderr)
        print(f"\nBasic statistics:", file=sys.stderr)
        print(df.describe(), file=sys.stderr)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
