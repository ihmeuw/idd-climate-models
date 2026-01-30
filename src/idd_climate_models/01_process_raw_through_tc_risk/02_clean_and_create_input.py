"""
Clean raw data and create time-period input files for TC-risk.

This script is called by the orchestrator as Level 2 of the workflow.
It:
  1. Reads raw NetCDF files for a variable
  2. Applies NaN filling using nearest neighbor interpolation
  3. Slices to the specified time period
  4. Regrids to target grid if needed
  5. Writes output to TC_RISK_INPUT_PATH

This combines the "fill" logic from fill_and_yearly_split.py with the
time-chunk creation logic from 1b_process_time_chunk.py.
"""

import argparse
import sys
import os
from pathlib import Path
import xarray as xr
import numpy as np
from scipy.ndimage import distance_transform_edt

import idd_climate_models.constants as rfc
from idd_climate_models.climate_file_functions import (
    should_regrid,
    regrid_with_cdo,
)
from idd_climate_models.validation_functions import is_monthly, filter_files_by_time_period


# ============================================================================
# NAN FILLING FUNCTIONS (from fill_and_yearly_split.py)
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
    mask_is_constant = all(np.array_equal(first_mask, np.isnan(data[t, ...])) for t in range(min(5, data.shape[0])))

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
        variable = ds[var_name]
        if variable.isnull().any():
            print(f"        Filling NaNs in variable '{var_name}' using nearest neighbor...")

            # Get numpy data, fill it, and create a new DataArray
            filled_np_data = fill_nans_nearest(variable.values)
            filled_var = xr.DataArray(
                filled_np_data,
                coords=variable.coords,
                dims=variable.dims,
                attrs=variable.attrs
            )
            ds_filled[var_name] = filled_var
        else:
            # If no NaNs, just add the original variable
            ds_filled[var_name] = variable

    return ds_filled


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Clean raw data and create time-period input files for TC-risk'
    )
    parser.add_argument('--data_source', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--time_period', required=True)
    parser.add_argument('--variable', required=True)
    parser.add_argument('--grid', required=True)
    parser.add_argument('--frequency', required=True)

    args = parser.parse_args()

    # --- 1. DEFINE PATHS AND SETTINGS ---
    TARGET_GRID = 'r360x180'  # 1x1 global grid

    raw_data_path = Path(rfc.RAW_DATA_PATH)
    target_base_path = rfc.TC_RISK_INPUT_PATH / args.data_source

    source_dir = (
        raw_data_path / args.data_source / args.model / args.variant /
        args.scenario / args.variable / args.grid / args.frequency
    )
    target_dir = (
        target_base_path / args.model / args.variant /
        args.scenario / args.time_period
    )

    print(f"\n{'='*80}")
    print(f"Processing {args.variable}/{args.frequency} for {args.model}/{args.variant}")
    print(f"Scenario: {args.scenario}")
    print(f"Time period: {args.time_period}")
    print(f"Source grid: {args.grid}")
    print(f"Target grid: {TARGET_GRID}")
    print(f"Source dir: {source_dir}")
    print(f"Target dir: {target_dir}")
    print(f"{'='*80}\n")

    target_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. PARSE TIME PERIOD ---
    bin_start_str, bin_end_str = args.time_period.split('-')
    bin_start = int(bin_start_str)
    bin_end = int(bin_end_str)

    # Construct date strings based on frequency
    if is_monthly(args.frequency):
        date_start = f"{bin_start}01"
        date_end = f"{bin_end}12"
    else:
        date_start = f"{bin_start}0101"
        date_end = f"{bin_end}1231"

    # --- 3. FIND FILES COVERING TIME PERIOD ---
    files_in_range = filter_files_by_time_period(
        str(source_dir), args.variable, bin_start, bin_end, args.frequency,
        args.model, args.scenario, args.variant, args.grid
    )

    if not files_in_range:
        print(f"  No files found for {args.variable} in {bin_start}-{bin_end}")
        sys.exit(1)

    print(f"  Found {len(files_in_range)} files covering time period")

    # --- 4. LOAD AND COMBINE FILES ---
    print(f"  Loading and combining files...")

    try:
        # Open all files as a single dataset
        ds = xr.open_mfdataset(
            sorted(set(files_in_range)),
            combine='by_coords',
            engine='netcdf4'
        )

        # Slice to exact time period
        ds = ds.sel(time=slice(f"{bin_start}-01-01", f"{bin_end}-12-31"))

        actual_years = np.unique(ds.time.dt.year.values)
        print(f"  Loaded data for years: {actual_years[0]}-{actual_years[-1]} ({len(actual_years)} years)")
        print(f"  Time steps: {len(ds.time)}")

    except Exception as e:
        print(f"  Failed to load files: {e}")
        sys.exit(1)

    # --- 5. APPLY NAN FILLING ---
    print(f"  Checking for NaN values...")

    has_nans = any(ds[var].isnull().any() for var in ds.data_vars)

    if has_nans:
        print(f"  Applying NaN filling...")
        ds_filled = fill_nans_xarray(ds)
    else:
        print(f"  No NaN values found")
        ds_filled = ds

    # --- 6. CHECK IF REGRIDDING IS NEEDED ---
    # Save to a temporary file first if regridding is needed
    needs_regridding = should_regrid(files_in_range[0], target_grid=TARGET_GRID)

    # Set output grid label: 'gr' if regridded, keep original if not
    output_grid = 'gr' if needs_regridding else args.grid

    # Construct output filename
    output_filename = (
        f"{args.variable}_{args.frequency}_{args.model}_{args.scenario}_"
        f"{args.variant}_{output_grid}_{date_start}-{date_end}.nc"
    )
    output_path = target_dir / output_filename

    # --- 7. WRITE OUTPUT ---
    print(f"  Writing output...")

    if needs_regridding:
        print(f"  Will regrid from {args.grid} to {output_grid} ({TARGET_GRID})")

        # Write to temporary file first
        temp_path = target_dir / f"temp_{output_filename}"

        # Compute to load into memory (avoids chunked datetime encoding issues)
        ds_filled = ds_filled.compute()

        # Encoding for compression (data variables)
        encoding = {
            var: {
                'zlib': True,
                'complevel': 7,
                'shuffle': True,
            }
            for var in ds_filled.data_vars
        }

        # Add proper encoding for time_bnds if present (fixes datetime encoding error)
        if 'time_bnds' in ds_filled:
            encoding['time_bnds'] = {'units': 'days since 1850-01-01', 'dtype': 'float64'}

        ds_filled.to_netcdf(temp_path, encoding=encoding, engine='netcdf4')
        ds_filled.close()
        ds.close()

        # Regrid using CDO
        success = regrid_with_cdo(temp_path, output_path, TARGET_GRID)

        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()

        if not success:
            print(f"  Failed to regrid to {TARGET_GRID}")
            sys.exit(1)

        print(f"  Successfully regridded to {output_filename}")

    else:
        print(f"  Files already on target grid, no regridding needed")

        # Compute to load into memory (avoids chunked datetime encoding issues)
        ds_filled = ds_filled.compute()

        # Encoding for compression (data variables)
        encoding = {
            var: {
                'zlib': True,
                'complevel': 7,
                'shuffle': True,
            }
            for var in ds_filled.data_vars
        }

        # Add proper encoding for time_bnds if present (fixes datetime encoding error)
        if 'time_bnds' in ds_filled:
            encoding['time_bnds'] = {'units': 'days since 1850-01-01', 'dtype': 'float64'}

        ds_filled.to_netcdf(output_path, encoding=encoding, engine='netcdf4')
        ds_filled.close()
        ds.close()

        print(f"  Successfully wrote {output_filename}")

    # --- 8. SUMMARY ---
    print(f"\n{'='*80}")
    print(f"Completed {args.variable}/{args.frequency}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
