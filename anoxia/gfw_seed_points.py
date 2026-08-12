"""
Extract GFW (Global Fishing Watch) seed points from monthly CSV data.

Loads fishing effort data from Global Fishing Watch monthly fleet CSV files,
filters to Indian Ocean region, aggregates by location, and extracts the
top 8 grid cells with highest fishing activity.

Usage:
  python gfw_seed_points.py [--csv-path PATH] [--output OUTPUT]

Output: JSON file with seed points for drift simulation modeling
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# Indian Ocean bounding box
INDIAN_OCEAN_BBOX = {
    'lat_min': 0,
    'lat_max': 30,
    'lon_min': 55,
    'lon_max': 100,
}


def find_fleet_monthly_csvs() -> Optional[Path]:
    """
    Search for fleet-monthly-csvs ZIP files in common locations.
    Looks in Downloads folder for files matching pattern.
    """
    downloads = Path.home() / "Downloads"
    
    if not downloads.exists():
        return None
    
    # Look for fleet-monthly-csvs ZIP files
    for pattern in ["fleet-monthly-csvs-10-v3-2024.zip", 
                    "fleet-monthly-csvs-10-v3-2023.zip",
                    "fleet-monthly-csvs-*.zip"]:
        matches = list(downloads.glob(pattern))
        if matches:
            # Return the most recent one
            return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    
    return None


def extract_csv_from_zip(zip_path: Path) -> Optional[str]:
    """
    Extract a CSV file from the ZIP archive.
    Returns the content as a string.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Find CSV files in the ZIP
            csv_files = [f for f in zf.namelist() if f.lower().endswith('.csv')]
            
            if not csv_files:
                print(f"  Error: No CSV files found in {zip_path.name}")
                return None
            
            # Use the first CSV file
            csv_file = csv_files[0]
            print(f"  Found CSV: {csv_file}")
            
            # Read the CSV content
            with zf.open(csv_file) as f:
                return f.read().decode('utf-8')
    except Exception as e:
        print(f"  Error extracting from ZIP: {e}")
        return None


def load_gfw_data(csv_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load GFW CSV data from file path or ZIP archive.
    """
    # If no path provided, try to find the ZIP file
    if csv_path is None:
        print("Searching for fleet-monthly-csvs ZIP file...")
        zip_path = find_fleet_monthly_csvs()
        
        if zip_path is None:
            print(f"Error: Could not find fleet-monthly-csvs ZIP file")
            print(f"Expected location: {Path.home() / 'Downloads'}")
            return None
        
        print(f"  Found: {zip_path.name}")
        
        # Extract CSV from ZIP
        csv_content = extract_csv_from_zip(zip_path)
        if csv_content is None:
            return None
        
        # Load from string
        try:
            from io import StringIO
            df = pd.read_csv(StringIO(csv_content))
        except Exception as e:
            print(f"  Error loading CSV from ZIP: {e}")
            return None
    else:
        # Load from file path
        try:
            if csv_path.endswith('.zip'):
                zip_path = Path(csv_path)
                csv_content = extract_csv_from_zip(zip_path)
                if csv_content is None:
                    return None
                from io import StringIO
                df = pd.read_csv(StringIO(csv_content))
            else:
                df = pd.read_csv(csv_path, low_memory=False)
        except FileNotFoundError:
            print(f"Error: File not found: {csv_path}")
            return None
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None
    
    return df


def standardize_columns(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Inspect and standardize column names.
    
    Maps common GFW column names to standard names:
    - latitude/lat/cell_ll_lat → lat
    - longitude/lon/cell_ll_lon → lon
    - fishing_hours/hours → fishing_hours
    """
    # Create lowercase column mapping
    lower_cols = {col.lower(): col for col in df.columns}
    
    # Define column mappings
    lat_aliases = ['lat', 'latitude', 'cell_ll_lat', 'lat_bin']
    lon_aliases = ['lon', 'longitude', 'cell_ll_lon', 'lon_bin']
    effort_aliases = ['fishing_hours', 'hours', 'fishing_effort', 'effort']
    
    # Find the actual columns
    lat_col = None
    lon_col = None
    effort_col = None
    
    for alias in lat_aliases:
        if alias in lower_cols:
            lat_col = lower_cols[alias]
            break
    
    for alias in lon_aliases:
        if alias in lower_cols:
            lon_col = lower_cols[alias]
            break
    
    for alias in effort_aliases:
        if alias in lower_cols:
            effort_col = lower_cols[alias]
            break
    
    # Check if columns were found
    if not all([lat_col, lon_col, effort_col]):
        print("Error: Required columns not found in CSV")
        print(f"Available columns: {list(df.columns)}")
        if lat_col:
            print(f"  ✓ Latitude column found: {lat_col}")
        else:
            print(f"  ✗ Latitude column not found (searched: {lat_aliases})")
        if lon_col:
            print(f"  ✓ Longitude column found: {lon_col}")
        else:
            print(f"  ✗ Longitude column not found (searched: {lon_aliases})")
        if effort_col:
            print(f"  ✓ Fishing effort column found: {effort_col}")
        else:
            print(f"  ✗ Fishing effort column not found (searched: {effort_aliases})")
        return None
    
    # Create a copy and rename columns
    df_standard = df.copy()
    rename_map = {
        lat_col: 'lat',
        lon_col: 'lon',
        effort_col: 'fishing_hours'
    }
    df_standard.rename(columns=rename_map, inplace=True)
    
    # Keep only the columns we need
    df_standard = df_standard[['lat', 'lon', 'fishing_hours']]
    
    return df_standard


def filter_indian_ocean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter data to Indian Ocean bounding box.
    
    Bounding box:
      Latitude:  0 to 30 N
      Longitude: 55 to 100 E
    """
    bbox = INDIAN_OCEAN_BBOX
    
    # Apply filters
    df_filtered = df[
        (df['lat'] >= bbox['lat_min']) &
        (df['lat'] <= bbox['lat_max']) &
        (df['lon'] >= bbox['lon_min']) &
        (df['lon'] <= bbox['lon_max'])
    ]
    
    print(f"  Original rows: {len(df)}")
    print(f"  After Indian Ocean filter: {len(df_filtered)}")
    
    return df_filtered


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values by dropping rows with NaN in required columns.
    """
    required_cols = ['lat', 'lon', 'fishing_hours']
    
    # Check for missing values before dropping
    missing_counts = df[required_cols].isnull().sum()
    if missing_counts.sum() > 0:
        print(f"  Missing values detected:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"    {col}: {count} rows")
    
    # Drop rows with NaN
    df_clean = df.dropna(subset=required_cols)
    
    print(f"  After removing NaN: {len(df_clean)}")
    
    return df_clean


def aggregate_fishing_effort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group data by spatial grid (lat, lon) and sum fishing_hours.
    """
    df_grouped = df.groupby(['lat', 'lon'], as_index=False)['fishing_hours'].sum()
    
    print(f"  Unique grid cells: {len(df_grouped)}")
    
    return df_grouped


def select_top_seed_points(df: pd.DataFrame, top_n: int = 8) -> List[List[float]]:
    """
    Sort by fishing_hours (descending) and select top N grid cells.
    
    Returns: List of [lat, lon] pairs
    """
    # Sort by fishing_hours in descending order
    df_sorted = df.sort_values(['fishing_hours'], ascending=False)
    
    # Select top N
    top_n_actual = min(top_n, len(df_sorted))
    df_top = df_sorted.head(top_n_actual)
    
    # Convert to list of [lat, lon] pairs
    seed_points = df_top[['lat', 'lon']].values.tolist()
    
    print(f"  Selected {len(seed_points)} seed points")
    for i, (lat, lon) in enumerate(seed_points, 1):
        effort = df_top.iloc[i-1]['fishing_hours']
        print(f"    {i}. ({lat:.3f}, {lon:.3f}) - {effort:.1f} hours")
    
    return seed_points


def save_seed_points(seed_points: List[List[float]], output_path: str) -> bool:
    """
    Save seed points to JSON file.
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        data = {
            "seed_points": seed_points,
            "count": len(seed_points),
            "region": "Indian Ocean",
            "bbox": INDIAN_OCEAN_BBOX,
        }
        
        # Write JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving seed points: {e}")
        return False


def main(csv_path: Optional[str] = None, output_path: Optional[str] = None, 
         lat_min: float = 0, lat_max: float = 30, 
         lon_min: float = 55, lon_max: float = 100) -> int:
    """
    Main pipeline: Load → Filter → Aggregate → Select → Save
    
    Args:
        csv_path: Path to CSV file or ZIP archive
        output_path: Output JSON file path
        lat_min, lat_max: Latitude bounds for filtering
        lon_min, lon_max: Longitude bounds for filtering
    """
    # Use default output path if not provided
    if output_path is None:
        output_path = "./data/gfw/gfw_seed_points.json"
    
    # Override INDIAN_OCEAN_BBOX with custom bounds
    global INDIAN_OCEAN_BBOX
    INDIAN_OCEAN_BBOX = {
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lon_min': lon_min,
        'lon_max': lon_max,
    }
    
    print("=" * 60)
    print("GFW SEED POINTS EXTRACTION")
    print("=" * 60)
    
    # Load CSV
    print("\n1. Loading CSV file...")
    df = load_gfw_data(csv_path)
    if df is None:
        return 1
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Standardize columns
    print("\n2. Standardizing column names...")
    df = standardize_columns(df)
    if df is None:
        return 1
    print(f"  Columns: {list(df.columns)}")
    
    # Filter to Indian Ocean
    print("\n3. Filtering to Indian Ocean...")
    df = filter_indian_ocean(df)
    if len(df) == 0:
        print("Error: No data found in Indian Ocean region")
        return 1
    
    # Handle missing values
    print("\n4. Handling missing values...")
    df = handle_missing_values(df)
    if len(df) == 0:
        print("Error: No data remaining after removing NaN values")
        return 1
    
    # Aggregate by grid cell
    print("\n5. Aggregating fishing effort by grid cell...")
    df_agg = aggregate_fishing_effort(df)
    
    # Select top 8 seed points
    print("\n6. Selecting top 8 seed points...")
    seed_points = select_top_seed_points(df_agg, top_n=8)
    
    # Save to JSON
    print("\n7. Saving seed points to JSON...")
    if save_seed_points(seed_points, output_path):
        print(f"  Saved to: {output_path}")
    else:
        return 1
    
    # Final summary
    print("\n" + "=" * 60)
    print(f"Found {len(seed_points)} seed points for ghost gear drift")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract GFW seed points from monthly CSV data"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to GFW monthly CSV file or ZIP archive"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/gfw/gfw_seed_points.json",
        help="Output JSON file path (default: ./data/gfw/gfw_seed_points.json)"
    )
    parser.add_argument(
        "--lat-min",
        type=float,
        default=0,
        help="Minimum latitude for filtering (default: 0)"
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        default=30,
        help="Maximum latitude for filtering (default: 30)"
    )
    parser.add_argument(
        "--lon-min",
        type=float,
        default=55,
        help="Minimum longitude for filtering (default: 55)"
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=100,
        help="Maximum longitude for filtering (default: 100)"
    )
    
    args = parser.parse_args()
    
    sys.exit(main(csv_path=args.csv_path, output_path=args.output,
                   lat_min=args.lat_min, lat_max=args.lat_max,
                   lon_min=args.lon_min, lon_max=args.lon_max))

