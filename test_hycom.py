#!/usr/bin/env python
"""Test HYCOM data and RegularGridInterpolator functionality."""

import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Test loading one file to verify structure
ds = xr.open_dataset('data/hycom/hycom_20240905.nc')
print('✓ Successfully loaded HYCOM NetCDF')
print(f'  Variables: {list(ds.data_vars)}')
print(f'  Dimensions: {dict(ds.dims)}')
print(f'  Coordinates: {list(ds.coords)}')
print()

u_data = ds['water_u'].values
v_data = ds['water_v'].values
print(f'water_u shape: {u_data.shape}')
print(f'water_v shape: {v_data.shape}')
print()

# Extract 2D slice if needed
if u_data.ndim == 4:  # (time, depth, lat, lon)
    u_data = u_data[0, 0]  # first time, first depth
    v_data = v_data[0, 0]
elif u_data.ndim == 3:  # (time, lat, lon) - use first timestep
    u_data = u_data[0]
    v_data = v_data[0]

lat = ds['lat'].values
lon = ds['lon'].values

print(f'Creating RegularGridInterpolator with u grid shape {u_data.shape}...')
u_interp = RegularGridInterpolator((lat, lon), u_data, bounds_error=False, fill_value=np.nan)
v_interp = RegularGridInterpolator((lat, lon), v_data, bounds_error=False, fill_value=np.nan)
print('✓ Interpolators created successfully')
print()

# Test interpolation at a point
test_point = np.array([[15.0, 75.0]])  # Mid-point in region
u_interp_val = u_interp(test_point)[0]
v_interp_val = v_interp(test_point)[0]
print(f'Test interpolation at (15N, 75E):')
print(f'  u velocity: {u_interp_val:.4f} m/s')
print(f'  v velocity: {v_interp_val:.4f} m/s')
print(f'  Speed: {np.sqrt(u_interp_val**2 + v_interp_val**2):.4f} m/s')
