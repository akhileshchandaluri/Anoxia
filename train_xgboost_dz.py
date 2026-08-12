"""
Train XGBoost model for hypoxia prediction (denitrification zone indicator).

Pipeline:
1. Load feature matrix from parquet
2. Recover temporal information from Argo JSON
3. Perform temporal train-test split (80/20)
4. Compute class weights for imbalance
5. Train XGBClassifier
6. Evaluate on test set (F1, AUROC, precision, recall)
7. Save model and feature importance
"""

import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    auc, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split


def load_feature_matrix(parquet_path: str = "./data/argo/feature_matrix.parquet") -> pd.DataFrame:
    """Load feature matrix from parquet file."""
    df = pd.read_parquet(parquet_path)
    print(f"Loaded feature matrix: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def recover_dates(feature_matrix: pd.DataFrame, 
                  argo_json_path: str = "./data/argo/argo_profiles.json") -> pd.DataFrame:
    """
    Recover temporal information from Argo JSON and add to feature matrix.
    
    The feature matrix was created by filtering Argo profiles, so we need to
    recover the dates for temporal train/test split.
    
    Falls back to row-order proxy if JSON is missing or empty.
    """
    try:
        with open(argo_json_path, 'r') as f:
            file_content = f.read().strip()
            if not file_content:
                raise ValueError("Argo JSON file is empty")
            profiles = json.loads(file_content)
        
        print(f"Loaded {len(profiles)} Argo profiles from JSON")
        
        # Extract dates and coordinates
        profile_data = []
        for profile in profiles:
            profile_data.append({
                'lat': profile['lat'],
                'lon': profile['lon'],
                'date': profile['date'],
            })
        
        argo_df = pd.DataFrame(profile_data)
        print(f"Created DataFrame from {len(argo_df)} profiles")
        
        # Merge with feature matrix on lat, lon
        # Note: Multiple profiles might have same (lat, lon), so use merge_asof after sorting
        argo_df = argo_df.sort_values('date')
        feature_matrix_sorted = feature_matrix.copy()
        
        # Create a temporary key for merging (round coordinates to avoid floating point issues)
        argo_df['lat_round'] = (argo_df['lat'] * 10).round() / 10
        argo_df['lon_round'] = (argo_df['lon'] * 10).round() / 10
        feature_matrix_sorted['lat_round'] = (feature_matrix_sorted['lat'] * 10).round() / 10
        feature_matrix_sorted['lon_round'] = (feature_matrix_sorted['lon'] * 10).round() / 10
        
        # For temporal split, use the date as temporal ordering
        # Since one feature matrix row corresponds to one Argo profile,
        # we need to match them and assign dates
        if 'date' in feature_matrix_sorted.columns:
            try:
                feature_matrix_sorted = feature_matrix_sorted.sort_values('date')
            except Exception:
                pass
        
        # Drop old date if exists and use Argo dates
        if 'date' in feature_matrix_sorted.columns:
            feature_matrix_sorted = feature_matrix_sorted.drop('date', axis=1)
        
        # Merge to add dates
        merged = feature_matrix_sorted.merge(
            argo_df[['lat_round', 'lon_round', 'date']],
            on=['lat_round', 'lon_round'],
            how='left'
        )
        
        # If merge didn't work well, just use feature matrix order
        if merged['date'].isnull().sum() > len(merged) * 0.5:
            print("Warning: Date merge failed. Using feature matrix row order as temporal proxy.")
            feature_matrix_sorted['date'] = range(len(feature_matrix_sorted))
            return feature_matrix_sorted
        
        # Clean up temporary columns
        merged = merged.drop(['lat_round', 'lon_round'], axis=1)
        
        print(f"Date recovery: {merged['date'].isnull().sum()} missing values")
        return merged
        
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load Argo JSON ({type(e).__name__}: {str(e)[:60]})...")
        print("Using feature matrix row order as temporal proxy")
        feature_matrix['date'] = range(len(feature_matrix))
        return feature_matrix


def prepare_data(feature_matrix: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepare X and y for training.
    
    X = [chl_t, chl_t8, chl_t16, delta_chlor, month_sin, month_cos]
    y = label
    """
    feature_cols = ['chl_t', 'chl_t8', 'chl_t16', 'delta_chlor', 'month_sin', 'month_cos']
    
    # Verify all features are present
    missing = [col for col in feature_cols if col not in feature_matrix.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    X = feature_matrix[feature_cols].values
    y = feature_matrix['label'].values
    
    print(f"Feature matrix prepared:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  y distribution: {np.bincount(y.astype(int))}")
    
    return X, y, feature_matrix[feature_cols + ['label', 'date']]


def temporal_train_test_split(X: np.ndarray, y: np.ndarray, 
                              feature_matrix: pd.DataFrame,
                              test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform temporal train-test split.
    
    Sorts by date and splits: first 80% train, last 20% test.
    No shuffling to maintain temporal order.
    """
    # Sort by date
    sorted_indices = np.argsort(pd.to_datetime(feature_matrix['date'], errors='coerce'))
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Use sklearn's train_test_split with shuffle=False for temporal split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sorted, y_sorted,
        test_size=test_size,
        shuffle=False,
        random_state=None
    )
    
    print(f"\nTemporal Train-Test Split:")
    print(f"  Train set: {X_train.shape[0]} samples ({len(y_train)-np.sum(y_train)} neg, {np.sum(y_train)} pos)")
    print(f"  Test set: {X_test.shape[0]} samples ({len(y_test)-np.sum(y_test)} neg, {np.sum(y_test)} pos)")
    
    return X_train, X_test, y_train, y_test


def compute_class_weight(y_train: np.ndarray) -> float:
    """
    Compute class weight for imbalanced data.
    
    scale_pos_weight = (count of negatives) / (count of positives)
    """
    n_negative = np.sum(y_train == 0)
    n_positive = np.sum(y_train == 1)
    
    if n_positive == 0:
        print("Warning: No positive samples in training set")
        scale_pos_weight = 1.0
    else:
        scale_pos_weight = n_negative / n_positive
    
    print(f"\nClass Weight Calculation:")
    print(f"  Negative samples: {n_negative}")
    print(f"  Positive samples: {n_positive}")
    print(f"  scale_pos_weight: {scale_pos_weight:.4f}")
    
    return scale_pos_weight


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  scale_pos_weight: float) -> xgb.XGBClassifier:
    """
    Train XGBClassifier with specified hyperparameters.
    """
    print(f"\nTraining XGBoost model...")
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric='aucpr',
        random_state=42,
        verbosity=0
    )
    
    model.fit(
        X_train, y_train,
        verbose=False
    )
    
    print("  ✓ Model training complete")
    
    return model


def evaluate_model(model: xgb.XGBClassifier, 
                   X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate model on test set.
    
    Metrics: F1, AUROC, precision, recall
    Handles case where only one class is present.
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics with handling for single class
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    # AUROC and AUPRC only valid if both classes present
    n_classes = len(np.unique(y_test))
    
    if n_classes < 2:
        print(f"\n{'='*60}")
        print("TEST SET EVALUATION METRICS")
        print(f"{'='*60}")
        print(f"⚠ WARNING: Test set contains only {n_classes} class(es)")
        print(f"   All samples are class {np.unique(y_test)[0]}")
        print(f"\nMetrics computed where applicable:")
        print(f"F1 Score:        {f1:.4f} (zero_division=0)")
        print(f"Precision:       {precision:.4f} (zero_division=0)")
        print(f"Recall:          {recall:.4f} (zero_division=0)")
        print(f"AUROC:           N/A (requires both classes)")
        print(f"AUPRC:           N/A (requires both classes)")
        print(f"{'='*60}")
        
        metrics = {
            'F1': float(f1),
            'AUROC': None,
            'AUPRC': None,
            'Precision': float(precision),
            'Recall': float(recall),
            'FPR': None,
            'TPR': None,
            'Warning': 'Single class in test set'
        }
    else:
        # Compute AUPRC (area under precision-recall curve)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        auprc = auc(recall_curve, precision_curve)
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auroc = roc_auc_score(y_test, y_pred_proba)
        
        # Print metrics
        print(f"\n{'='*60}")
        print("TEST SET EVALUATION METRICS")
        print(f"{'='*60}")
        print(f"F1 Score:        {f1:.4f}")
        print(f"AUROC:           {auroc:.4f}")
        print(f"AUPRC:           {auprc:.4f}")
        print(f"Precision:       {precision:.4f}")
        print(f"Recall:          {recall:.4f}")
        print(f"{'='*60}")
        
        metrics = {
            'F1': float(f1),
            'AUROC': float(auroc),
            'AUPRC': float(auprc),
            'Precision': float(precision),
            'Recall': float(recall),
            'FPR': fpr.tolist(),
            'TPR': tpr.tolist(),
        }
    
    return metrics


def save_model(model: xgb.XGBClassifier, model_path: str = "./models/dz_predictor.pkl") -> None:
    """Save model to file using joblib."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved to: {model_path}")


def save_feature_importance(model: xgb.XGBClassifier, 
                           feature_names: list,
                           importance_path: str = "./models/feature_importance.json") -> None:
    """
    Save feature importance using XGBoost Booster.
    
    Steps:
    1. Get booster from model
    2. Compute gain-based importance scores
    3. Map feature names to indices
    4. Normalize importance to [0, 1]
    5. Save and print sorted results
    """
    # ========================================================================
    # 1. Get booster and compute importance
    # ========================================================================
    booster = model.get_booster()
    importance_scores = booster.get_score(importance_type='gain')
    
    # ========================================================================
    # 2. Map feature names from indices
    # ========================================================================
    # importance_scores has keys like 'f0', 'f1', ... 'f5'
    importance_named = {}
    for key, value in importance_scores.items():
        # Extract numeric index from 'fN' format
        try:
            feat_idx = int(key[1:])  # Remove 'f' prefix and convert to int
            if 0 <= feat_idx < len(feature_names):
                importance_named[feature_names[feat_idx]] = float(value)
        except (ValueError, IndexError):
            pass
    
    # Add missing features with 0 importance
    for fname in feature_names:
        if fname not in importance_named:
            importance_named[fname] = 0.0
    
    # ========================================================================
    # 3. Normalize importance
    # ========================================================================
    total_importance = sum(importance_named.values())
    if total_importance > 0:
        importance_normalized = {
            k: v / total_importance for k, v in importance_named.items()
        }
    else:
        # If all zeros, equal weight
        importance_normalized = {
            k: 1.0 / len(feature_names) for k in feature_names
        }
    
    # ========================================================================
    # 4. Sort by importance (highest first)
    # ========================================================================
    sorted_importance = sorted(
        importance_normalized.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # ========================================================================
    # 5. Save to JSON
    # ========================================================================
    output_data = {
        'features': feature_names,
        'importance_method': 'gain',
        'importance_normalized': {k: v for k, v in sorted_importance},
        'metadata': {
            'total_importance_raw': float(total_importance),
            'normalization': 'sum to 1.0'
        }
    }
    
    Path(importance_path).parent.mkdir(parents=True, exist_ok=True)
    with open(importance_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Feature importance saved to: {importance_path}")
    
    # ========================================================================
    # 6. Print sorted importance
    # ========================================================================
    print("\nFeature Importance (Gain-Based, Normalized):")
    print(f"{'Rank':<5} {'Feature':<18} {'Normalized Score':<18} {'Percentage':<10}")
    print("-" * 55)
    for rank, (feature, importance) in enumerate(sorted_importance, 1):
        percentage = importance * 100
        print(f"{rank:<5} {feature:<18} {importance:>17.6f} {percentage:>8.2f}%")


def main() -> int:
    """Main pipeline."""
    print("="*60)
    print("XGBOOST HYPOXIA PREDICTOR TRAINING")
    print("="*60)
    
    try:
        # 1. Load feature matrix
        print("\n1. Loading feature matrix...")
        feature_matrix = load_feature_matrix()
        
        # 2. Recover dates for temporal split
        print("\n2. Recovering temporal information...")
        feature_matrix = recover_dates(feature_matrix)
        
        # 3. Prepare data
        print("\n3. Preparing data...")
        X, y, data_with_date = prepare_data(feature_matrix)
        
        # 4. Temporal train-test split
        print("\n4. Performing temporal train-test split...")
        X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, data_with_date)
        
        # 5. Compute class weights
        print("\n5. Computing class weights...")
        scale_pos_weight = compute_class_weight(y_train)
        
        # 6. Train XGBoost
        model = train_xgboost(X_train, y_train, scale_pos_weight)
        
        # 7. Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        
        # 8. Save model
        save_model(model)
        
        # 9. Save feature importance
        feature_names = ['chl_t', 'chl_t8', 'chl_t16', 'delta_chlor', 'month_sin', 'month_cos']
        save_feature_importance(model, feature_names)
        
        print("\n" + "="*60)
        print("✓ TRAINING COMPLETE")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
