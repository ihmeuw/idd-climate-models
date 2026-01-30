"""
Process a single variable for a given model/variant/scenario/time_period.

This script:
  1. Finds all raw files for the variable that overlap with the time period
  2. Reads them (lazy loading)
  3. Clips to the time period years
  4. Optionally regrids to target grid (r360x180)
  5. Fills NaNs using nearest neighbor interpolation (AFTER regridding)
  6. Saves the combined file to TC_RISK_INPUT_PATH

Usage:
  python 02_process_variable.py --model MRI-ESM2-0 --variant r2i1p1f1 --scenario historical \
      --time_period 1970-1999 --variable tos --grid gn --frequency Omon --data_source cmip6
"""

import argparse
import sys
import subprocess
import os
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
from scipy.ndimage import distance_transform_edt

import idd_climate_models.constants as rfc
from idd_climate_models.climate_file_functions import should_regrid, regrid_with_cdo
from idd_climate_models.validation_functions import is_monthly


# ============================================================================
# NAN FILLING FUNCTIONS
# ============================================================================

def fill_nans_nearest(data):
    """Fills NaNs using nearest neighbor interpolation on a numpy array.

    Optimized: if NaN pattern is constant across time (e.g., land mask),
    compute indices once and apply to all timesteps.
    """
    if not np.any(np.isnan(data)):
        return data

    data_filled = data.copy()

    # Handle 2D arrays
    if data.ndim == 2:
        mask = np.isnan(data)
        ind = distance_transform_edt(mask, return_distances=False, return_indices=True)
        data_filled[mask] = data[tuple(ind[:, mask])]
        return data_filled

    # Handle 3D+ arrays (time, lat, lon)
    # Check if NaN mask is constant across time (common for land/ocean masks)
    first_mask = np.isnan(data[0, ...])

    # Quick check on first few timesteps
    mask_is_constant = True
    for t in range(min(5, data.shape[0])):
        if not np.array_equal(first_mask, np.isnan(data[t, ...])):
            mask_is_constant = False
            break

    if mask_is_constant and np.any(first_mask):
        # Optimized path: compute indices once, apply to all timesteps
        print(f"        NaN mask is constant - using optimized single-index computation")
        ind = distance_transform_edt(first_mask, return_distances=False, return_indices=True)
        for t in range(data.shape[0]):
            data_filled[t, ...][first_mask] = data[t, ...][tuple(ind[:, first_mask])]
    else:
        # Fallback: per-timestep computation (for varying NaN patterns)
        print(f"        NaN mask varies - processing {data.shape[0]} timesteps")
        for t in range(data.shape[0]):
            slice_data = data[t, ...]
            mask = np.isnan(slice_data)
            if np.any(mask):
                ind = distance_transform_edt(mask, return_distances=False, return_indices=True)
                data_filled[t, ...][mask] = slice_data[tuple(ind[:, mask])]

    return data_filled


def fill_nans_xarray(ds):
    """Applies the numpy-based nearest neighbor fill to an xarray Dataset."""
    ds_filled = xr.Dataset(coords=ds.coords, attrs=ds.attrs)

    for var_name in ds.data_vars:
        variable_ds = ds[var_name]
        if variable_ds.isnull().any():
            print(f"        Filling NaNs in variable '{var_name}' using nearest neighbor...")
            filled_np_data = fill_nans_nearest(variable_ds.values)
            filled_var = xr.DataArray(
                filled_np_data,
                coords=variable_ds.coords,
                dims=variable_ds.dims,
                attrs=variable_ds.attrs
            )
            ds_filled[var_name] = filled_var
        else:
            ds_filled[var_name] = variable_ds
    return ds_filled


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def parse_date_from_filename(filename, frequency):
    """Extract start and end dates from a NetCDF filename.

    Expects format like: variable_freq_model_scenario_variant_grid_YYYYMMDD-YYYYMMDD.nc
    or: variable_freq_model_scenario_variant_grid_YYYYMM-YYYYMM.nc
    """
    import re

    if is_monthly(frequency):
        # Try YYYYMM-YYYYMM format (monthly)
        match = re.search(r'_(\d{6})-(\d{6})\.nc$', filename)
        if match:
            start_date = match.group(1)
            end_date = match.group(2)
            return int(start_date[:4]), int(end_date[:4])
    else:
        # Try YYYYMMDD-YYYYMMDD format (daily)
        match = re.search(r'_(\d{8})-(\d{8})\.nc$', filename)
        if match:
            start_date = match.group(1)
            end_date = match.group(2)
            return int(start_date[:4]), int(end_date[:4])

    return None, None


def find_overlapping_files(raw_dir, start_year, end_year, frequency):
    """Find all raw files in raw_dir that overlap with [start_year, end_year]."""
    if not raw_dir.exists():
        return []

    overlapping = []
    for f in sorted(raw_dir.glob("*.nc")):
        file_start, file_end = parse_date_from_filename(f.name, frequency)
        if file_start is None:
            print(f"    Warning: Could not parse dates from {f.name}, including anyway")
            overlapping.append(f)
            continue

        # Check overlap: file overlaps if file_end >= start_year AND file_start <= end_year
        if file_end >= start_year and file_start <= end_year:
            overlapping.append(f)

    return overlapping

def get_output_filename(args, grid=None):
    if grid is None:
        grid = args.grid

    target_dir = (
        rfc.TC_RISK_INPUT_PATH / args.data_source / args.model / args.variant /
        args.scenario / args.time_period
    )
    start_year, end_year = map(int, args.time_period.split('-'))
        # Construct filename
    if is_monthly(args.frequency):
        date_start = f"{start_year}01"
        date_end = f"{end_year}12"
    else:
        date_start = f"{start_year}0101"
        date_end = f"{end_year}1231"

    output_filename = (
        f"{args.variable}_{args.frequency}_{args.model}_{args.scenario}_"
        f"{args.variant}_{grid}_{date_start}-{date_end}.nc"
    )
    return target_dir, output_filename

def run_cdo(command):
    """Helper to run CDO commands and check for errors."""
    try:
        subprocess.run(f"cdo -O {command}", shell=True, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR in CDO: {e.stderr.decode()}")
        return False

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', default='cmip6')
    parser.add_argument('--model', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--time_period', required=True)
    parser.add_argument('--variable', required=True)
    parser.add_argument('--grid', required=True)
    parser.add_argument('--frequency', required=True)
    args = parser.parse_args()

    start_time = datetime.now()
    start_year, end_year = map(int, args.time_period.split('-'))

    # Output Paths
    target_dir, output_filename = get_output_filename(args) # Use your existing func
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # We always work toward 'gr' (regridded) for the final output path
    _, final_name = get_output_filename(args, 'gr')
    final_output_path = target_dir / final_name
    
    # Temporary files
    tmp_merged = target_dir / f"tmp_merged_{args.variable}.nc"
    tmp_regridded = target_dir / f"tmp_regrid_{args.variable}.nc"

    # 1. Find Files
    raw_dir = rfc.RAW_DATA_PATH / args.data_source / args.model / args.variant / args.scenario / args.variable / args.grid / args.frequency
    raw_files = find_overlapping_files(raw_dir, start_year, end_year, args.frequency) # Use your existing func

    print(f"Found {len(raw_files)} files. Starting CDO pipeline...")

    # 2. CDO: MERGE and SELECT DATE in one pass
    file_list = " ".join([str(f) for f in raw_files])
    
    print(f"Step 1: Merging and Clipping time...")
    # CDO seldate also accepts year ranges, which is safer across frequencies
    merge_cmd = f"-P 8 selyear,{start_year}/{end_year} -mergetime {file_list} {tmp_merged}"
    if not run_cdo(merge_cmd): sys.exit(1)

    # Check if regridding is needed (using your existing function)
    needs_regridding = should_regrid(str(raw_files[0]), target_grid='r360x180')

    if needs_regridding:
        print(f"Step 2: Regridding {args.grid} to r360x180 via CDO...")
        regrid_cmd = f"-P 8 remapbil,r360x180 {tmp_merged} {tmp_regridded}"
        if not run_cdo(regrid_cmd): sys.exit(1)
        input_for_python = tmp_regridded
        output_grid_label = 'gr'
    else:
        print(f"Step 2: Skipping regridding (already on target grid)...")
        input_for_python = tmp_merged
        output_grid_label = args.grid

    # 4. PYTHON: FINAL POLISH
    print(f"Step 3: Loading into xarray for final NaN check/fill...")
    ds = xr.open_dataset(input_for_python)

    # Convert non-standard calendars to standard calendar
    if 'time' in ds.coords:
        calendar = ds.time.encoding.get('calendar', 'standard')
        if calendar not in ['standard', 'gregorian', 'proleptic_gregorian']:
            print(f"    Converting calendar from '{calendar}' to 'standard'...")
            import pandas as pd
            new_times = pd.to_datetime([
                f"{t.year:04d}-{t.month:02d}-{t.day:02d} {t.hour:02d}:{t.minute:02d}:{t.second:02d}"
                for t in ds.time.values
            ])
            ds = ds.assign_coords(time=new_times)

    # 5. SAVE FINAL
    _, final_name = get_output_filename(args, output_grid_label)
    final_output_path = target_dir / final_name

    # Check for NaNs
    has_nans = any(ds[var].isnull().any() for var in ds.data_vars)
    
    if has_nans:
        print(f"    NaNs detected. Filling and re-saving with compression...")
        ds = fill_nans_xarray(ds)
        
        encoding = {v: {'zlib': True, 'complevel': 5, 'shuffle': True} for v in ds.data_vars}
        
        # Add time encoding to ensure TC-risk compatibility
        if 'time' in ds.coords:
            encoding['time'] = {
                'units': 'days since 1850-01-01',
                'calendar': 'standard',
                'dtype': 'float64'
            }
        
        ds.to_netcdf(final_output_path, encoding=encoding)
        ds.close()
    else:
        print(f"    No NaNs detected. Re-encoding time coordinate...")
        
        # Even with no NaNs, we need to ensure proper time encoding
        # Load the CDO output and re-save with explicit time encoding
        encoding = {
            v: {'zlib': True, 'complevel': 5, 'shuffle': True} 
            for v in ds.data_vars
        }
        if 'time' in ds.coords:
            encoding['time'] = {
                'units': 'days since 1850-01-01',
                'calendar': 'standard',
                'dtype': 'float64'
            }
        
        ds.to_netcdf(final_output_path, encoding=encoding)
        ds.close()

    # 6. SET PERMISSIONS TO 775
    try:
        os.chmod(final_output_path, 0o775)
        print(f"  ✓ Permissions set to 775 for: {final_output_path.name}")
    except Exception as e:
        print(f"  ! Warning: Could not set permissions: {e}")

    # 7. CLEANUP
    for tmp in [tmp_merged, tmp_regridded]:
        if tmp.exists():
            tmp.unlink()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"SUCCESS: Processed in {elapsed:.1f}s | Output: {final_output_path}")

if __name__ == "__main__":
    main()