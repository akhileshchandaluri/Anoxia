import json
import numpy as np

# Verify drift_paths.json
print("Drift Paths Structure:")
with open('outputs/drift_paths.json') as f:
    drift_data = json.load(f)

print(f"  Keys: {list(drift_data.keys())}")
print(f"  N trajectories: {drift_data['n_trajectories']}")
print(f"  Points per trajectory: {drift_data['metadata']['n_points_per_trajectory']}")
print(f"  Sample trajectory ID 0 (first 5 points):")
for pt in drift_data['trajectories'][0]['path'][:5]:
    print(f"    {pt}")

# Verify fusion_risk_map.json
print("\nFusion Risk Map Structure:")
with open('outputs/fusion_risk_map.json') as f:
    risk_data = json.load(f)

print(f"  Keys: {list(risk_data.keys())}")
print(f"  N grid points: {risk_data['metadata']['n_points']}")
print(f"  Risk range: {risk_data['metadata']['risk_min']:.6f} - {risk_data['metadata']['risk_max']:.6f}")
print(f"  Risk mean: {risk_data['metadata']['risk_mean']:.6f}")
print(f"  First 5 lats: {risk_data['lat'][:5]}")
print(f"  First 5 lons: {risk_data['lon'][:5]}")
print(f"  First 5 risks: {risk_data['risk'][:5]}")

# High risk zones
risk_arr = np.array(risk_data['risk'])
high_risk = np.sum(risk_arr >= 0.3)
print(f"\n  High-risk grid points (risk >= 0.3): {high_risk}")
