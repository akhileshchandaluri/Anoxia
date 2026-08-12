"""
Fix hypoxia labels by mapping feature matrix to nearest Argo profiles.

Pipeline:
1. Load feature matrix (satellites features + current incorrect labels)
2. Load Argo profiles with real dissolved oxygen (DO) measurements
3. Compute hypoxia labels from DO (< 62.5 µmol/kg = hypoxic)
4. For each feature matrix row, find nearest Argo profile spatially
5. Assign the Argo hypoxia label to feature matrix row
6. Save updated feature matrix with corrected labels
7. Report label distribution

This approach aligns sparse Argo oxygen measurements with satellite feature
grid using nearest-neighbor spatial mapping, enabling supervised learning.
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_feature_matrix(parquet_path: str = "./data/argo/feature_matrix.parquet") -> pd.DataFrame:
    """Load feature matrix from parquet file."""
    df = pd.read_parquet(parquet_path)
    print(f"Loaded feature matrix: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    return df


def load_argo_profiles(json_path: str = "./data/argo/argo_profiles.json") -> pd.DataFrame:
    """Load Argo profiles from JSON and create DataFrame."""
    try:
        with open(json_path, 'r') as f:
            profiles = json.load(f)
        
        print(f"Loaded {len(profiles)} Argo profiles")
        
        # Extract profile data
        profile_data = []
        for profile in profiles:
            profile_data.append({
                'argo_lat': profile['lat'],
                'argo_lon': profile['lon'],
                'date': profile['date'],
                'do': profile['do'],  # Dissolved oxygen in µmol/kg
            })
        
        argo_df = pd.DataFrame(profile_data)
        print(f"  DO range: {argo_df['do'].min():.1f} - {argo_df['do'].max():.1f} µmol/kg")
        
        return argo_df
        
    except FileNotFoundError:
        print(f"Error: File not found: {json_path}")
        raise
    except Exception as e:
        print(f"Error loading Argo profiles: {e}")
        raise


def create_hypoxia_labels(argo_df: pd.DataFrame, threshold: float = 62.5) -> pd.DataFrame:
    """
    Create hypoxia labels from Argo DO measurements.
    
    Label = 1 if DO < threshold (hypoxic)
    Label = 0 if DO >= threshold (oxic)
    
    Args:
        argo_df: DataFrame with 'do' column
        threshold: DO threshold in µmol/kg (default: 62.5)
    
    Returns:
        DataFrame with added 'argo_label' column
    """
    argo_df = argo_df.copy()
    argo_df['argo_label'] = (argo_df['do'] < threshold).astype(int)
    
    print(f"\nHypoxia Label Creation (threshold: {threshold} µmol/kg):")
    print(f"  Oxic (label=0):    {(argo_df['argo_label'] == 0).sum()} profiles")
    print(f"  Hypoxic (label=1): {(argo_df['argo_label'] == 1).sum()} profiles")
    
    return argo_df


def nearest_neighbor_distance(lat_feat: float, lon_feat: float,
                               argo_df: pd.DataFrame) -> Tuple[int, float]:
    """
    Find nearest Argo profile to a feature matrix point using Euclidean distance.
    
    Distance = sqrt((lat1 - lat2)^2 + (lon1 - lon2)^2)
    
    Args:
        lat_feat, lon_feat: Coordinates from feature matrix
        argo_df: DataFrame with Argo profiles (argo_lat, argo_lon columns)
    
    Returns:
        (nearest_index, nearest_distance)
    """
    # Compute distances to all Argo points
    distances = np.sqrt(
        (argo_df['argo_lat'].values - lat_feat) ** 2 +
        (argo_df['argo_lon'].values - lon_feat) ** 2
    )
    
    # Find nearest
    nearest_idx = np.argmin(distances)
    nearest_dist = distances[nearest_idx]
    
    return nearest_idx, nearest_dist


def assign_labels_from_nearest(feature_matrix: pd.DataFrame,
                               argo_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    For each row in feature matrix, assign label from nearest Argo profile.
    
    Returns:
        (feature_matrix_with_labels, distances_to_nearest)
    """
    print(f"\nAssigning labels from nearest Argo profiles...")
    
    n_features = len(feature_matrix)
    new_labels = np.zeros(n_features, dtype=int)
    distances = np.zeros(n_features, dtype=float)
    
    for i, row in feature_matrix.iterrows():
        lat = row['lat']
        lon = row['lon']
        
        # Find nearest Argo profile
        nearest_idx, nearest_dist = nearest_neighbor_distance(lat, lon, argo_df)
        
        # Assign label
        new_labels[i] = argo_df.iloc[nearest_idx]['argo_label']
        distances[i] = nearest_dist
        
        if (i + 1) % max(1, n_features // 10) == 0:
            print(f"  Processed {i + 1}/{n_features} rows")
    
    print(f"  ✓ Assignment complete")
    print(f"\nSpatial Distance Statistics:")
    print(f"  Mean distance: {distances.mean():.4f}°")
    print(f"  Median distance: {np.median(distances):.4f}°")
    print(f"  Max distance: {distances.max():.4f}°")
    print(f"  Min distance: {distances.min():.4f}°")
    
    # Create updated feature matrix
    feature_matrix_updated = feature_matrix.copy()
    feature_matrix_updated['label'] = new_labels
    feature_matrix_updated['dist_to_nearest_argo'] = distances
    
    return feature_matrix_updated, distances


def validate_labels(feature_matrix: pd.DataFrame) -> bool:
    """
    Validate that labels are properly assigned.
    
    Checks:
    - No missing labels
    - Both positive and negative samples exist (or reasonable explanation)
    """
    print(f"\nLabel Validation:")
    
    # Check for missing values
    missing = feature_matrix['label'].isnull().sum()
    print(f"  Missing labels: {missing}")
    
    if missing > 0:
        print(f"  ✗ ERROR: {missing} rows with missing labels")
        return False
    
    # Check label distribution
    label_counts = feature_matrix['label'].value_counts()
    print(f"  Label distribution: {dict(label_counts)}")
    
    if len(label_counts) < 2:
        print(f"  ⚠ WARNING: Only {len(label_counts)} class(es) present")
        if 0 in label_counts and label_counts[0] == len(feature_matrix):
            print(f"    All samples are oxic (label=0)")
            print(f"    This may indicate: (a) truly oxic region, or")
            print(f"                        (b) all nearest Argo profiles are oxic")
    else:
        print(f"  ✓ Both classes present")
    
    return True


def print_summary(feature_matrix: pd.DataFrame, argo_df: pd.DataFrame) -> None:
    """Print summary statistics."""
    print(f"\n{'='*70}")
    print("LABEL ASSIGNMENT SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nFeature Matrix:")
    print(f"  Total rows: {len(feature_matrix)}")
    print(f"  Oxic (label=0):    {(feature_matrix['label'] == 0).sum()}")
    print(f"  Hypoxic (label=1): {(feature_matrix['label'] == 1).sum()}")
    print(f"  Label percentage:  {100 * (feature_matrix['label'] == 1).sum() / len(feature_matrix):.1f}% hypoxic")
    
    print(f"\nOriginal Argo Data:")
    print(f"  Total profiles: {len(argo_df)}")
    print(f"  Oxic profiles:    {(argo_df['argo_label'] == 0).sum()}")
    print(f"  Hypoxic profiles: {(argo_df['argo_label'] == 1).sum()}")
    
    print(f"\nSample rows from updated feature matrix:")
    print(feature_matrix[['lat', 'lon', 'label', 'dist_to_nearest_argo', 'chl_t', 'delta_chlor']].head(10).to_string())


def main() -> int:
    """Main pipeline."""
    print("="*70)
    print("HYPOXIA LABEL FIXER")
    print("="*70)
    
    try:
        # 1. Load data
        print("\n1. Loading data...")
        feature_matrix = load_feature_matrix()
        argo_df = load_argo_profiles()
        
        # 2. Create hypoxia labels from Argo DO
        print("\n2. Creating hypoxia labels from Argo DO...")
        argo_df = create_hypoxia_labels(argo_df, threshold=62.5)
        
        # 3. Assign labels using nearest neighbor
        print("\n3. Assigning labels using nearest-neighbor spatial mapping...")
        feature_matrix_updated, distances = assign_labels_from_nearest(feature_matrix, argo_df)
        
        # 4. Validate
        print("\n4. Validating labels...")
        is_valid = validate_labels(feature_matrix_updated)
        
        if not is_valid:
            print("\n✗ Label validation failed")
            return 1
        
        # 5. Print summary
        print_summary(feature_matrix_updated, argo_df)
        
        # 6. Save updated feature matrix
        print(f"\n5. Saving updated feature matrix...")
        output_path = "./data/argo/feature_matrix.parquet"
        
        # Drop the temporary distance column before saving (keep only original columns + label)
        feature_matrix_to_save = feature_matrix_updated.drop('dist_to_nearest_argo', axis=1)
        feature_matrix_to_save.to_parquet(output_path)
        print(f"  ✓ Saved to: {output_path}")
        
        print(f"\n{'='*70}")
        print("✓ LABEL ASSIGNMENT COMPLETE")
        print(f"{'='*70}")
        print(f"\nNext steps:")
        print(f"  1. Review label distribution above")
        print(f"  2. Check feature_matrix.parquet for updated labels")
        print(f"  3. Retrain XGBoost with: python train_xgboost_dz.py")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
