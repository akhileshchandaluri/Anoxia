"""
Filter GFW seed points to Regional Study Areas:
- Arabian Sea: lat 5-25, lon 55-75
- Bay of Bengal: lat 5-25, lon 80-95
"""

import json
from pathlib import Path
import pandas as pd


# Define regional boundaries
REGIONS = {
    'Arabian_Sea': {
        'lat_min': 5,
        'lat_max': 25,
        'lon_min': 55,
        'lon_max': 75,
    },
    'Bay_of_Bengal': {
        'lat_min': 5,
        'lat_max': 25,
        'lon_min': 80,
        'lon_max': 95,
    }
}


def load_seed_points(json_path: str) -> list:
    """Load seed points from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get('seed_points', [])
    except FileNotFoundError:
        print(f"Error: File not found: {json_path}")
        return []
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return []


def filter_points_to_region(points: list, region_name: str, region_bbox: dict) -> list:
    """Filter points to a specific region."""
    filtered = []
    for lat, lon in points:
        if (region_bbox['lat_min'] <= lat <= region_bbox['lat_max'] and
            region_bbox['lon_min'] <= lon <= region_bbox['lon_max']):
            filtered.append((lat, lon, region_name))
    return filtered


def create_combined_dataframe(json_path: str = "./data/gfw/gfw_seed_points.json") -> pd.DataFrame:
    """
    Load seed points, filter to regional study areas, and combine into DataFrame.
    """
    print("=" * 70)
    print("REGIONAL STUDY AREA FILTERING")
    print("=" * 70)
    
    # Load seed points
    print(f"\n1. Loading seed points from {json_path}...")
    seed_points = load_seed_points(json_path)
    print(f"   Loaded {len(seed_points)} seed points")
    
    if not seed_points:
        print("   Error: No seed points found")
        return pd.DataFrame()
    
    # Filter to each region
    print("\n2. Filtering to regional boundaries...\n")
    
    all_filtered = []
    
    for region_name, region_bbox in REGIONS.items():
        filtered = filter_points_to_region(seed_points, region_name, region_bbox)
        print(f"   {region_name}:")
        print(f"     Boundaries: lat {region_bbox['lat_min']}-{region_bbox['lat_max']}, "
              f"lon {region_bbox['lon_min']}-{region_bbox['lon_max']}")
        print(f"     Points found: {len(filtered)}")
        if filtered:
            for lat, lon, region in filtered:
                print(f"       • ({lat:.1f}, {lon:.1f})")
        all_filtered.extend(filtered)
    
    # Create combined DataFrame
    print(f"\n3. Creating combined DataFrame...")
    if all_filtered:
        df = pd.DataFrame(all_filtered, columns=['lat', 'lon', 'region'])
        print(f"   Total points in study regions: {len(df)}")
        print(f"\n   Summary:")
        print(f"   {df['region'].value_counts().to_string()}")
    else:
        print("   Warning: No seed points found in either study region")
        df = pd.DataFrame(columns=['lat', 'lon', 'region'])
    
    print("\n" + "=" * 70)
    print("COMBINED STUDY REGION DATA")
    print("=" * 70)
    print(df.to_string(index=False))
    
    return df


if __name__ == "__main__":
    # Filter and create combined DataFrame
    df = create_combined_dataframe()
    
    # Save to CSV for reference
    output_csv = "./data/gfw/study_region_seed_points.csv"
    if len(df) > 0:
        df.to_csv(output_csv, index=False)
        print(f"\n✓ Saved combined data to: {output_csv}")
    else:
        print(f"\n⚠ No data to save (no points in study regions)")
    
    # Print summary statistics
    if len(df) > 0:
        print("\n" + "=" * 70)
        print("STATISTICS")
        print("=" * 70)
        print(f"Total points: {len(df)}")
        print(f"Latitude range: {df['lat'].min():.1f} to {df['lat'].max():.1f}")
        print(f"Longitude range: {df['lon'].min():.1f} to {df['lon'].max():.1f}")
        print(f"\nPoints by region:\n{df['region'].value_counts()}")
