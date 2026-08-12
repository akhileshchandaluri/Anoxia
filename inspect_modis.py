import xarray as xr
import glob
import os

modis_files = sorted(glob.glob('data/modis/*.nc'))
print(f'Found {len(modis_files)} MODIS files')

if modis_files:
    filename = os.path.basename(modis_files[0])
    print(f'First file: {filename}')
    
    ds = xr.open_dataset(modis_files[0])
    print(f'Variables: {list(ds.data_vars)}')
    print(f'Coordinates: {list(ds.coords)}')
    print(f'Dimensions: {dict(ds.dims)}')
    
    lat = ds['lat'].values
    lon = ds['lon'].values
    print(f'Lat range: [{lat.min():.2f}, {lat.max():.2f}]')
    print(f'Lon range: [{lon.min():.2f}, {lon.max():.2f}]')
    
    # Check data variable
    for var in ds.data_vars:
        print(f'{var} shape: {ds[var].shape}')
        print(f'{var} dtype: {ds[var].dtype}')
        print(f'{var} range: [{ds[var].values.min():.4f}, {ds[var].values.max():.4f}]')
