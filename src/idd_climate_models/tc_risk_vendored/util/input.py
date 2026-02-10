#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Jonathan Lin
Utility library for dealing with CMIP6 and reanalysis data files.
"""

import calendar
import cftime
import datetime
import glob
import numpy as np
import xarray as xr

# Lazy import: only import namelist when functions need it (not at module level)
# This allows Dask workers to import this module without having namelist_loader

def preprocess_grib(config_dict):
    fns = glob.glob('%s/**/*%s*.grib' % (config_dict['base_directory'], config_dict['exp_prefix']), recursive = True)
    for fn in fns:
        idx_fns = glob.glob('%s*.idx' % fn)
        if len(idx_fns) == 0:
            # Generate the .idx file in a preprocessing step,
            # which has overhead and causes issues if you
            # try to open multiple .grib files in parallel.
            ds = xr.open_dataset(fn)
            ds.close()

# # RCR: Added function to look for either lat or latitude
# def normalize_coords(ds, var_key):
#     """
#     Checks for variations of lon/lat keys ('lon'/'longitude', 'lat'/'latitude') 
#     and renames them to standard short keys ('lon', 'lat') within the dataset.
#     """
#     ds_vars = list(ds.coords)
    
#     # 1. Normalize Longitude
#     if 'longitude' in ds_vars and 'lon' not in ds_vars:
#         ds = ds.rename({'longitude': 'lon'})
#     # 2. Normalize Latitude
#     if 'latitude' in ds_vars and 'lat' not in ds_vars:
#         ds = ds.rename({'latitude': 'lat'})
    
#     # 3. Check Vertical Level Key (plev/lev)
#     plev_keys = ['plev', 'level', 'lev'] # Add common pressure level keys here
    
#     current_lvl_key = input.get_lvl_key() # e.g., 'plev'
    
#     if current_lvl_key not in ds_vars:
#         # If the input key doesn't exist, try to find a known key and rename it.
#         for pk in plev_keys:
#             if pk in ds_vars:
#                 ds = ds.rename({pk: current_lvl_key})
#                 break
        
#     return ds

def _open_fns(fns):
    # RCR: Add decode_times=True fix here as well!
    if len(fns) == 1:
        ds = xr.open_dataset(fns[0], decode_times=True)
    else:
        ds = xr.open_mfdataset(fns, concat_dim="time", combine='nested',
                               data_vars="minimal", drop_variables=['nbdate'],
                               decode_times=True) # <-- Time decoding fix applied
    
    # --- CRITICAL FIX: CALL NORMALIZATION HERE ---
    # The normalization must be called right after loading, before keys are accessed.
    # It takes the dataset and the full var_keys dictionary.
    # ds = normalize_coords(ds, namelist.var_keys)
    
    return ds

def _glob_prefix(var_prefix, namelist_dict):
    if namelist_dict['file_type'] == 'netcdf':
        ext = 'nc'
    elif namelist_dict['file_type'] == 'grib':
        ext = 'grib'
    else:
        raise RuntimeError('File type %s not supported' % namelist_dict['file_type'])

    fns = glob.glob('%s/**/*%s*.%s' % (namelist_dict['base_directory'], namelist_dict['exp_prefix'], ext), recursive = True)
    fns_var = sorted([x for x in fns if '_%s_' % var_prefix in x])
    if len(fns_var) == 0:
        fns_var = sorted([x for x in fns if '%s_' % var_prefix in x])
    return(fns_var)

def _find_in_timerange(fns, ct_start, ct_end = None):
    fns_multi = []
    for fn in fns:
        ds = xr.open_dataset(fn)
        time = ds['time']                      # time
        if ct_start is not None and ct_end is None:
            if ((ct_start >= time[0]) & (ct_start <= time[-1])):
                fns_multi.append(fn)
        else:
            if ((time >= ct_start) & (time <= ct_end)).any():
                fns_multi.append(fn)
        ds.close()

    return(fns_multi)

"""
Opens files described by "var", bounded by times ct_start and ct_end.
If only ct_start, then opens the files at time ct_start.
If both ct_start and ct_end are None, opens all files by var.
"""
def _load_var(config_dict, var_name, ct_start = None, ct_end = None):
    var_key = config_dict['var_keys'][config_dict['dataset_type']][var_name]
    if ct_start is None and ct_end is None:
        ds = _open_fns(_glob_prefix(var_key, config_dict))
    elif ct_start is not None and ct_end is None:
        ds = _open_fns(_find_in_timerange(_glob_prefix(var_key, config_dict), ct_start)).sel(time = ct_start)
    else:
        fns = _find_in_timerange(_glob_prefix(var_key, config_dict), ct_start, ct_end)
        ds = _open_fns(fns).sel(time=slice(ct_start, ct_end))
    return ds

def get_sst_key():
    # DEPRECATED: Use config_dict['var_keys'][config_dict['dataset_type']]['sst'] instead
    raise NotImplementedError("Use config_dict['var_keys'][config_dict['dataset_type']]['sst'] instead")

def get_mslp_key():
    # DEPRECATED: Use config_dict['var_keys'][config_dict['dataset_type']]['mslp'] instead
    raise NotImplementedError("Use config_dict['var_keys'][config_dict['dataset_type']]['mslp'] instead")

def get_temp_key():
    # DEPRECATED: Use config_dict['var_keys'][config_dict['dataset_type']]['temp'] instead
    raise NotImplementedError("Use config_dict['var_keys'][config_dict['dataset_type']]['temp'] instead")

def get_sp_hum_key():
    # DEPRECATED: Use config_dict['var_keys'][config_dict['dataset_type']]['sp_hum'] instead
    raise NotImplementedError("Use config_dict['var_keys'][config_dict['dataset_type']]['sp_hum'] instead")

def get_u_key():
    # DEPRECATED: Use config_dict['var_keys'][config_dict['dataset_type']]['u'] instead
    raise NotImplementedError("Use config_dict['var_keys'][config_dict['dataset_type']]['u'] instead")

def get_v_key():
    # DEPRECATED: Use config_dict['var_keys'][config_dict['dataset_type']]['v'] instead
    raise NotImplementedError("Use config_dict['var_keys'][config_dict['dataset_type']]['v'] instead")

def get_w_key():
    # DEPRECATED: Use config_dict['var_keys'][config_dict['dataset_type']]['w'] instead
    raise NotImplementedError("Use config_dict['var_keys'][config_dict['dataset_type']]['w'] instead")

def get_lvl_key():
    # DEPRECATED: Use config_dict['var_keys'][config_dict['dataset_type']]['lvl'] instead
    raise NotImplementedError("Use config_dict['var_keys'][config_dict['dataset_type']]['lvl'] instead")

def get_lon_key():
    # DEPRECATED: Use config_dict['var_keys'][config_dict['dataset_type']]['lon'] instead
    raise NotImplementedError("Use config_dict['var_keys'][config_dict['dataset_type']]['lon'] instead")

def get_lat_key():
    # DEPRECATED: Use config_dict['var_keys'][config_dict['dataset_type']]['lat'] instead
    raise NotImplementedError("Use config_dict['var_keys'][config_dict['dataset_type']]['lat'] instead")

def load_sst(config_dict, ct_start = None, ct_end = None):
    return _load_var(config_dict, 'sst', ct_start, ct_end)

def load_mslp(config_dict, ct_start = None, ct_end = None):
    return _load_var(config_dict, 'mslp', ct_start, ct_end)

def load_w(config_dict, ct_start = None, ct_end = None):
    return _load_var(config_dict, 'w', ct_start, ct_end)

def load_temp(config_dict, ct_start = None, ct_end = None):
    return _load_var(config_dict, 'temp', ct_start, ct_end)

def load_sp_hum(config_dict, ct_start = None, ct_end = None):
    return _load_var(config_dict, 'sp_hum', ct_start, ct_end)

def _load_var_daily(fn):
    # Daily variables are large cannot be loaded into memory as easily.
    # So this is an internal function that loads a file directly.
    ds = xr.open_dataset(fn)
    return ds

def convert_from_datetime(ds, dts):
    # Convert the datetime array dts to the timestamps used by ds.
    # Only supports np.datetime64 or cftime.DatetimeNoLeap to datetimes.
    # Necessary to convert between non-standard calendars (like no leap).
    if isinstance(np.array(ds['time'])[0], np.datetime64):
        adt = np.array([np.datetime64(str(x)) for x in np.array(dts)])
    elif isinstance(np.array(ds['time'])[0], cftime.DatetimeNoLeap):
        adt = np.array([cftime.DatetimeNoLeap(x.year, x.month, x.day, x.hour) for x in np.array(dts)])
    else:
        raise Exception("Did not understand type of time.")
    return adt

def convert_to_datetime(ds, dts):
    # Convert the timestamps types of ds to datetime timestamps.
    # Only supports np.datetime64 or cftime.DatetimeNoLeap to datetimes.
    # Necessary to convert between non-standard calendars (like no leap).
    if isinstance(np.array(ds['time'])[0], np.datetime64):
        adt = np.array(dts.astype('datetime64[s]').tolist())
    elif isinstance(np.array(ds['time'])[0], cftime.DatetimeNoLeap):
        adt = np.array([datetime.datetime(x.year, x.month, x.day, x.hour) for x in np.array(dts)])
    else:
        raise Exception("Did not understand type of time.")
    return adt

def get_bounding_times(config_dict):
    s_dt = datetime.datetime(config_dict['start_year'], config_dict['start_month'], 1)
    N_day = calendar.monthrange(config_dict['end_year'], config_dict['end_month'])[1]
    e_dt = datetime.datetime(config_dict['end_year'], config_dict['end_month'], N_day)
    return (s_dt, e_dt)
