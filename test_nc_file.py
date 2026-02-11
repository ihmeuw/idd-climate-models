#!/usr/bin/env python
"""Test reading a specific NetCDF file without deleting it."""

import sys
sys.path.insert(0, '/ihme/homes/bcreiner/repos/idd-climate-models/src')

from idd_climate_models.zarr_functions import validate_netcdf_file
import xarray as xr
import netCDF4 as nc

nc_file = "/mnt/team/rapidresponse/pub/tropical-storms/tc_risk/output/cmip6/MRI-ESM2-0/r2i1p1f1/historical/2011-2014/NA/tracks_NA_MRI-ESM2-0_historical_r2i1p1f1_201101_201412.nc"

print("=" * 80)
print(f"Testing file: {nc_file}")
print("=" * 80)

print("\n1. Testing with validate_netcdf_file()...")
try:
    result = validate_netcdf_file(nc_file)
    print(f"   Result: {result}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n2. Testing with netCDF4...")
try:
    ds = nc.Dataset(nc_file, 'r')
    print(f"   ✓ Opened successfully")
    print(f"   Variables: {list(ds.variables.keys())}")
    print(f"   Dimensions: {dict(ds.dimensions.items())}")
    ds.close()
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Testing with xarray...")
try:
    ds = xr.open_dataset(nc_file)
    print(f"   ✓ Opened successfully")
    print(f"   Variables: {list(ds.variables.keys())}")
    print(f"   Dimensions: {dict(ds.dims)}")
    print(f"   Number of tracks: {ds.sizes.get('n_trk', 'N/A')}")
    ds.close()
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Test complete")
print("=" * 80)
