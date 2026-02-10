import numpy as np
import xarray as xr
from pathlib import Path

from scipy.interpolate import RectBivariateSpline as interp2d

# LAZY IMPORT: import namelist_loader as namelist in each function that needs it

# Global cache - pre-loaded data shared across all calls
_bathy_cache = {}
_land_cache = {}
_drag_cache = {}
_initialized = False

def _initialize_cache(config_dict):
    """Load all data files once in the main process before workers access them."""
    global _initialized
    if _initialized:
        return
    
    fdir = config_dict['src_directory']
    
    try:
        # Load bathymetry
        fn = f'{fdir}/intensity/data/bathymetry.nc'
        ds = xr.open_dataset(fn, engine='netcdf4')
        _bathy_cache['_raw_lon'] = ds['lon'].data.copy()
        _bathy_cache['_raw_lat'] = ds['lat'].data.copy()
        _bathy_cache['_raw_bathy'] = ds['bathymetry'].data.copy()
        ds.close()
    except Exception as e:
        print(f"Warning: Could not pre-load bathymetry: {e}")
    
    try:
        # Load land mask
        fn = f'{fdir}/intensity/data/land.nc'
        ds = xr.open_dataset(fn, engine='netcdf4')
        _land_cache['_raw_lon'] = ds['lon'].data.copy()
        _land_cache['_raw_lat'] = ds['lat'].data.copy()
        _land_cache['_raw_land'] = ds['land'].data.copy()
        ds.close()
    except Exception as e:
        print(f"Warning: Could not pre-load land mask: {e}")
    
    try:
        # Load drag coefficient
        fn = f'{fdir}/intensity/data/Cd.nc'
        ds = xr.open_dataset(fn, engine='netcdf4')
        _drag_cache['_raw_lon'] = ds['longitude'].data.copy()
        _drag_cache['_raw_lat'] = ds['latitude'].data.copy()
        _drag_cache['_raw_Cd'] = ds['Cd'].data.copy()
        ds.close()
    except Exception as e:
        print(f"Warning: Could not pre-load drag coefficient: {e}")
    
    _initialized = True

# Reads in bathymetry file.
def read_bathy(basin, config_dict):
    global _bathy_cache
    
    # Initialize cache if not already done
    _initialize_cache(config_dict)
    
    # Create cache key based on basin name
    cache_key = basin.__class__.__name__
    
    # Return from cache if already processed
    if cache_key in _bathy_cache:
        return _bathy_cache[cache_key]
    
    # Get raw data from cache
    if '_raw_bathy' not in _bathy_cache:
        raise RuntimeError("Bathymetry data not loaded")
    
    lon = _bathy_cache['_raw_lon']
    lat = _bathy_cache['_raw_lat']
    bathy = _bathy_cache['_raw_bathy']
    
    lon_b, lat_b, bathy_b = basin.transform_global_field(lon, lat, bathy)
    f_bath = interp2d(lon_b, lat_b, bathy_b.T, kx=1, ky=1)
    
    # Cache the processed result
    _bathy_cache[cache_key] = f_bath
    
    return f_bath

# Reads in land mask file.
def read_land(basin, config_dict):
    global _land_cache
    
    # Initialize cache if not already done
    _initialize_cache(config_dict)
    
    # Create cache key based on basin name
    cache_key = basin.__class__.__name__
    
    # Return from cache if already processed
    if cache_key in _land_cache:
        return _land_cache[cache_key]
    
    # Get raw data from cache
    if '_raw_land' not in _land_cache:
        raise RuntimeError("Land mask data not loaded")
    
    lon = _land_cache['_raw_lon']
    lat = _land_cache['_raw_lat']
    land = _land_cache['_raw_land']
    
    lon_b, lat_b, land_b = basin.transform_global_field(lon, lat, land)
    f_land = interp2d(lon_b, lat_b, land_b.T, kx=1, ky=1)
    
    # Cache the processed result
    _land_cache[cache_key] = f_land
    
    return f_land

# Reads in drag coefficient file.
def read_drag(basin, config_dict):
    global _drag_cache
    
    # Initialize cache if not already done
    _initialize_cache(config_dict)
    
    # Create cache key based on basin name
    cache_key = basin.__class__.__name__
    
    # Return from cache if already processed
    if cache_key in _drag_cache:
        return _drag_cache[cache_key]
    
    # Get raw data from cache
    if '_raw_Cd' not in _drag_cache:
        raise RuntimeError("Drag coefficient data not loaded")
    
    lon = _drag_cache['_raw_lon']
    lat = _drag_cache['_raw_lat']
    Cd = _drag_cache['_raw_Cd']
    
    Cd_gradient = Cd / (1 + 250.0 * Cd)
    Cd_norm = Cd_gradient / np.min(Cd_gradient)
    Cd = config_dict['Cd'] * Cd_norm

    lon_b, lat_b, Cd_b = basin.transform_global_field(lon, lat, Cd)
    f_Cd = interp2d(lon_b, lat_b, Cd_b.T, kx=1, ky=1)
    
    # Cache the processed result
    _drag_cache[cache_key] = f_Cd
    
    return f_Cd