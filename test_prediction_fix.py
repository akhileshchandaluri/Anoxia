"""Test the XGBoost prediction fix with corrected encoding."""

import joblib
import numpy as np

print("Testing XGBoost prediction fix...")
print("")

# Load the existing model
print("1. Loading existing model...")
try:
    model = joblib.load("./models/dz_predictor.pkl")
    print("   Model loaded successfully")
    print(f"   Model type: {type(model)}")
except Exception as e:
    print(f"   Error loading model: {e}")
    exit(1)

# Create test data
print("")
print("2. Creating test data...")
n_samples = 100
X_test = np.random.randn(n_samples, 6).astype(np.float32)
print(f"   Test data shape: {X_test.shape}")
print(f"   Test data dtype: {X_test.dtype}")

# Test prediction with the CORRECTED approach
print("")
print("3. Testing prediction with corrected approach...")
try:
    # Ensure input is float32
    X_pred = X_test.astype(np.float32)
    
    # Get predictions
    probs = model.predict_proba(X_pred)
    print(f"   Prediction shape: {probs.shape}")
    print(f"   Columns: {probs.shape[1]} (should be 2 for binary classification)")
    
    # Extract hypoxia probabilities (class 1)
    if probs.shape[1] == 2:
        hypoxia_probs = probs[:, 1]
    else:
        hypoxia_probs = probs.flatten()
    
    print(f"   Hypoxia probability shape: {hypoxia_probs.shape}")
    print(f"   Probability range: {hypoxia_probs.min():.6f} to {hypoxia_probs.max():.6f}")
    print(f"   Mean probability: {hypoxia_probs.mean():.6f}")
    
    # Check for zeros
    n_zeros = np.sum(hypoxia_probs == 0.0)
    n_valid = np.sum(~np.isnan(hypoxia_probs))
    print(f"   Valid predictions: {n_valid}/{n_samples}")
    print(f"   Zero predictions: {n_zeros}/{n_samples}")
    
    if n_valid > 0 and hypoxia_probs.max() > 0.0:
        print("")
        print("   SUCCESS: Predictions are working correctly!")
        print("   - Predictions are NOT all zeros")
        print("   - No use_label_encoder errors")
    else:
        print("")
        print("   WARNING: Predictions appear to be all zeros or NaN")
        
except AttributeError as e:
    print(f"   AttributeError (use_label_encoder issue?): {e}")
    print("   FAILED: Model may need retraining")
except Exception as e:
    print(f"   Error: {type(e).__name__}: {e}")
    print("   FAILED: Unexpected error")

print("")
print("Test complete.")
