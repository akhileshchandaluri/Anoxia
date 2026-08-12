import json
import xarray as xr

# Check JSON
print("JSON File Structure:")
with open('outputs/dz_predictions.json') as f:
    data = json.load(f)

print(f"  Keys: {list(data.keys())}")
print(f"  N_points: {data['metadata']['n_points']}")
print(f"  Lat range: {data['metadata']['lat_range']}")
print(f"  Lon range: {data['metadata']['lon_range']}")
print(f"  First 5 lats: {data['dz_lats'][:5]}")
print(f"  First 5 lons: {data['dz_lons'][:5]}")
print(f"  First 5 probs: {data['dz_probs'][:5]}")
print(f"  Prob min/max: {min(data['dz_probs']):.6f} / {max(data['dz_probs']):.6f}")

# Check NetCDF
print("\nNetCDF File Structure:")
ds = xr.open_dataset('outputs/dz_prediction_30day.nc')
print(f"  Dimensions: {dict(ds.dims)}")
print(f"  Variables: {list(ds.data_vars)}")
print(f"  Coordinates: {list(ds.coords)}")
print(f"  Attributes: {dict(ds.attrs)}")
print(f"  prob_hypoxia shape: {ds['prob_hypoxia'].shape}")
print(f"  prob_hypoxia min/max: {float(ds['prob_hypoxia'].min()):.6f} / {float(ds['prob_hypoxia'].max()):.6f}")
