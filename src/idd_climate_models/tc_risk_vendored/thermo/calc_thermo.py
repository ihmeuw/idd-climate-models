#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jzlin@mit.edu
"""

import dask
import datetime
import os
# LAZY IMPORT: import namelist_loader as namelist in each function that needs it
import numpy as np
import xarray as xr

from dask.distributed import LocalCluster, Client
from ..util import input, mat
from ..thermo import thermo

def get_fn_thermo(config_dict):
    fn_th = '%s/thermo_%s_%d%02d_%d%02d.nc' % (config_dict['output_directory'], config_dict['exp_prefix'],
                                               config_dict['start_year'], config_dict['start_month'],
                                               config_dict['end_year'], config_dict['end_month'])
    return(fn_th)




def compute_thermo(dt_start, dt_end, config_dict):
    # RCR: Added 1
    # 1. Load datasets. Normalization now happens INSIDE input._open_fns()
    #    We also load the data (.load()) to make sure it's in memory before dask manipulation
    ds_sst = input.load_sst(config_dict, dt_start, dt_end).load()
    ds_psl = input.load_mslp(config_dict, dt_start, dt_end).load()
    ds_ta = input.load_temp(config_dict, dt_start, dt_end).load()
    ds_hus = input.load_sp_hum(config_dict, dt_start, dt_end).load()

    # ----------------------------------------------------------------------
    # FIX: Apply pressure level rounding to atmospheric data (ds_ta, ds_hus) 
    # to prevent floating-point key errors later in the loop.
    # ----------------------------------------------------------------------
    plev_key = config_dict['var_keys'][config_dict['dataset_type']]['lvl']
    if plev_key in ds_ta.coords:
        rounded_plev = ds_ta[plev_key].round(0) 
        ds_ta = ds_ta.assign_coords({plev_key: rounded_plev})
        ds_hus = ds_hus.assign_coords({plev_key: rounded_plev})
    # ----------------------------------------------------------------------

    # 2. Use Standardized Keys
    # After normalization, all longitude/latitude keys are guaranteed to be 'lon'/'lat' 
    # (or whatever short key input.get_lon_key() returns, IF we adjust the function)
    # 
    # NOTE: Assuming input.get_lon_key() returns 'lon' and input.get_lat_key() returns 'lat'
    lon_ky = config_dict['var_keys'][config_dict['dataset_type']]['lon']
    lat_ky = config_dict['var_keys'][config_dict['dataset_type']]['lat']
    sst_ky = config_dict['var_keys'][config_dict['dataset_type']]['sst']

    nTime = len(ds_sst['time'])
    mslp_key = config_dict['var_keys'][config_dict['dataset_type']]['mslp']
    vmax = np.zeros(ds_psl[mslp_key].shape)
    chi = np.zeros(ds_psl[mslp_key].shape)
    rh_mid = np.zeros(ds_psl[mslp_key].shape)
    
    for i in range(nTime):
        # Convert all variables to the atmospheric grid.
        sst_interp = mat.interp_2d_grid(ds_sst[lon_ky], ds_sst[lat_ky], 
                                        np.nan_to_num(ds_sst[sst_ky][i, :, :].data),
                                        ds_ta[lon_ky], ds_ta[lat_ky])
        if 'C' in ds_sst[sst_ky].units:
            sst_interp = sst_interp + 273.15

        temp_key = config_dict['var_keys'][config_dict['dataset_type']]['temp']
        sp_hum_key = config_dict['var_keys'][config_dict['dataset_type']]['sp_hum']
        
        psl = ds_psl[mslp_key][i, :, :]
        ta = ds_ta[temp_key][i, :, :, :]
        hus = ds_hus[sp_hum_key][i, :, :, :]
        lvl = ds_ta[plev_key]
        lvl_d = np.copy(ds_ta[plev_key].data)

        # Ensure lowest model level is first.
        # Here we assume the model levels are in pressure.
        if (lvl[0] - lvl[1]) < 0:
            ta = ta.reindex({plev_key: lvl[::-1]})
            hus = hus.reindex({plev_key: lvl[::-1]})
            lvl_d = lvl_d[::-1]
    
        p_midlevel = config_dict.get('p_midlevel', 50000)                    # Pa
        if lvl.units in ['millibars', 'hPa']:
            lvl_d *= 100                                    # needs to be in Pa
            p_midlevel_hpa = p_midlevel / 100          # hPa
            lvl_mid = lvl.sel({plev_key: p_midlevel_hpa}, method = 'nearest')
        # RCR: I added the next two lines
        # There used to be no definition of lvl_mid if the levels were in Pa already.
        # This would cause an UnboundLocalError later on when trying to use lvl_mid
        # I wanted to leave it in an if/elif in case units are something very different so things break
        elif lvl.units in ['Pa', 'pascal', 'Pascals']:
            lvl_mid = lvl.sel({plev_key: p_midlevel}, method = 'nearest')
        
        

        # TODO: Check units of psl, ta, and hus
        vmax_args = (sst_interp, psl.data, lvl_d, ta.data, hus.data, config_dict)
        vmax[i, :, :] = thermo.CAPE_PI_vectorized(*vmax_args)
        ta_midlevel = ta.sel({plev_key: p_midlevel if lvl.units in ['Pa', 'pascal', 'Pascals'] else p_midlevel_hpa}, method = 'nearest').data
        hus_midlevel = hus.sel({plev_key: p_midlevel if lvl.units in ['Pa', 'pascal', 'Pascals'] else p_midlevel_hpa}, method = 'nearest').data

        p_midlevel_Pa = float(lvl_mid) * 100 if lvl_mid.units in ['millibars', 'hPa'] else float(lvl_mid)
        chi_args = (sst_interp, psl.data, ta_midlevel,
                    p_midlevel_Pa, hus_midlevel, config_dict)
        chi[i, :, :] = np.minimum(np.maximum(thermo.sat_deficit(*chi_args), 0), 10)
        rh_mid[i, :, :] = thermo.conv_q_to_rh(ta_midlevel, hus_midlevel, p_midlevel_Pa)

    return (vmax, chi, rh_mid)

def gen_thermo(config_dict):
    # TODO: Assert all of the datasets have the same length in time.
    if os.path.exists(get_fn_thermo(config_dict)):
        return

    # Load datasets metadata. Since SST is split into multiple files and can
    # cause parallel reads with open_mfdataset to hang, save as a single file.
    dt_start, dt_end = input.get_bounding_times(config_dict)
    ds = input.load_mslp(config_dict)

    ct_bounds = [dt_start, dt_end]
    ds_times = input.convert_from_datetime(ds,
                   np.array([x for x in input.convert_to_datetime(ds, ds['time'].values)
                             if x >= ct_bounds[0] and x <= ct_bounds[1]]))

    n_chunks = config_dict['n_procs']
    chunks = np.array_split(ds_times, np.minimum(n_chunks, np.floor(len(ds_times) / 2)))

    cl_args = {'n_workers': config_dict['n_procs'],
               'processes': True,
               'threads_per_worker': 1}
    lazy_results = []
    with LocalCluster(**cl_args) as cluster, Client(cluster) as client:
        for i in range(n_chunks):
            lazy_result = dask.delayed(compute_thermo)(chunks[i][0], chunks[i][-1], config_dict)
            lazy_results.append(lazy_result)
        out = dask.compute(*lazy_results, scheduler = 'processes', num_workers = n_chunks)

    # Clean up and process output.
    # Ensure monthly timestamps have middle-of-the-month days.
    # EDIT BY BCR: Changed to check if the time step was month. If so, then make sure the date is the middle
    # of the month. If not, leave the data as it is.
    t0 = np.datetime64(ds['time'][0].values, 'M')
    t1 = np.datetime64(ds['time'][1].values, 'M')
    dt_months = t1 - t0

    t0_day = np.datetime64(ds['time'][0].values, 'D')
    t1_day = np.datetime64(ds['time'][1].values, 'D')
    dt_days = t1_day - t0_day

    if dt_months == np.timedelta64(1, 'M'):
        # X: monthly logic
        ds_times = input.convert_from_datetime(ds,
            np.array([datetime.datetime(x.year, x.month, 15) for x in
                [x for x in input.convert_to_datetime(ds, ds['time'].values)
                    if x >= ct_bounds[0] and x <= ct_bounds[1]]]))
        print("Detected monthly time steps; setting day to 15th of month.")
    elif dt_days == np.timedelta64(1, 'D'):
        # Y: daily logic
        ds_times = input.convert_from_datetime(ds,
            np.array([x for x in input.convert_to_datetime(ds, ds['time'].values)
                if x >= ct_bounds[0] and x <= ct_bounds[1]]))
        print("Detected daily time steps; keeping original days.")
    else:
        raise RuntimeError("Time step is neither 1 month nor 1 day!")
        
    vmax = np.concatenate([x[0] for x in out], axis = 0)
    chi = np.concatenate([x[1] for x in out], axis = 0)
    rh_mid = np.concatenate([x[2] for x in out], axis = 0)
    lon_key = config_dict['var_keys'][config_dict['dataset_type']]['lon']
    lat_key = config_dict['var_keys'][config_dict['dataset_type']]['lat']
    ds_thermo = xr.Dataset(data_vars = dict(vmax = (['time', 'lat', 'lon'], vmax),
                                            chi = (['time', 'lat', 'lon'], chi),
                                            rh_mid = (['time', 'lat', 'lon'], rh_mid)),
                           coords = dict(lon = ("lon", ds[lon_key].data),
                                         lat = ("lat", ds[lat_key].data),
                                         time = ("time", ds_times.astype('datetime64[ns]'))))
    ds_thermo.to_netcdf(get_fn_thermo(config_dict))
    print('Saved %s' % get_fn_thermo(config_dict))
