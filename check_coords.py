import xarray as xr
ds = xr.open_dataset('data/hycom/hycom_20240905.nc')
print('Coordinate ranges:')
print(f'  lat: [{ds["lat"].values.min():.2f}, {ds["lat"].values.max():.2f}]')
print(f'  lon: [{ds["lon"].values.min():.2f}, {ds["lon"].values.max():.2f}]')
print()
print('Data shape:', ds['water_u'].shape)
print('First few lat values:', ds['lat'].values[:5])
print('First few lon values:', ds['lon'].values[:5])
