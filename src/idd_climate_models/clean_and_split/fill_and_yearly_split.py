import os
import re
import numpy as np
import xarray as xr
from datetime import datetime
import argparse
from pathlib import Path
from scipy.ndimage import distance_transform_edt  # <-- ADD THIS LINE

# Assuming your constants file is accessible
import idd_climate_models.constants as rfc
from idd_climate_models.validate_model_functions import int_to_date, is_monthly

PROCESSED_DATA_PATH = rfc.PROCESSED_DATA_PATH

# Year filtering constants
MIN_YEAR = 1950
MAX_YEAR = 2100


def define_dest_dir(model, variant, scenario, variable, grid, time_period):
    """Define the destination directory based on model parameters."""
    dest_dir = os.path.join(PROCESSED_DATA_PATH, model, variant, scenario, variable, grid, time_period)
    os.makedirs(dest_dir, exist_ok=True)
    return dest_dir


def fill_nans_nearest(data):
    """Fills NaNs using nearest neighbor interpolation on a numpy array."""
    if not np.any(np.isnan(data)):
        return data
    
    data_filled = data.copy()
    
    # Handle multi-dimensional arrays (like time, lat, lon)
    if data.ndim > 2:
        for t in range(data.shape[0]):
            slice_data = data[t, ...]
            mask = np.isnan(slice_data)
            if np.any(mask):
                ind = distance_transform_edt(mask, return_distances=False, return_indices=True)
                data_filled[t, ...][mask] = slice_data[tuple(ind[:, mask])]
    # Handle 2D arrays
    else:
        mask = np.isnan(data)
        ind = distance_transform_edt(mask, return_distances=False, return_indices=True)
        data_filled[mask] = data[tuple(ind[:, mask])]
    
    return data_filled


def fill_nans_xarray(ds):
    """Applies the numpy-based nearest neighbor fill to an xarray Dataset."""
    ds_filled = xr.Dataset(coords=ds.coords, attrs=ds.attrs)
    
    for var_name in ds.data_vars:
        variable = ds[var_name]
        if variable.isnull().any():
            print(f"    Filling NaNs in variable '{var_name}' using nearest neighbor...")
            
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


def write_yearly_files(ds, src_file, dest_dir):
    """Write dataset split into yearly files."""
    time_folder = os.path.basename(os.path.dirname(src_file))
    is_monthly = 'mon' in time_folder.lower()
    is_daily = time_folder.lower() == 'day'
    
    # The dataset is already trimmed, so we just get the years present
    years = np.unique(ds["time.year"].values)
    
    print(f"  Years to process: {len(years)} ({years[0]}-{years[-1]})")
    
    for year in years:
        try:
            ds_year = ds.sel(time=ds.time.dt.year == year)
            if len(ds_year.time) == 0:
                continue
            
            base_name = os.path.basename(src_file)
            if is_monthly:
                out_fname = re.sub(r'_(\d{6})-(\d{6})\.nc$', f'_{year}01-{year}12.nc', base_name)
            elif is_daily:
                out_fname = re.sub(r'_(\d{8})-(\d{8})\.nc$', f'_{year}0101-{year}1231.nc', base_name)
            else:
                out_fname = re.sub(r'_(\d{6,8})-(\d{6,8})\.nc$', f'_{year}.nc', base_name)
            
            out_path = os.path.join(dest_dir, out_fname)
            encoding = {var: {'zlib': True, 'complevel': 4, 'shuffle': True} for var in ds_year.data_vars}
            ds_year.to_netcdf(out_path, encoding=encoding, engine='netcdf4')
            print(f"    Wrote: {out_fname}")
            
        except Exception as e:
            print(f"    ✗ Failed to write year {year}: {str(e)}")
            raise


def process_file(file_path, dest_dir):
    """Process a single NetCDF file: trim, fill NaNs, and split into yearly files."""
    try:
        start_time = datetime.now()
        print(f"Processing: {os.path.basename(file_path)}")
        
        # 1. Open dataset (NO CHUNKING)
        ds = xr.open_dataset(file_path, engine='netcdf4')
        
        # 2. Trim data to be between MIN_YEAR and MAX_YEAR
        original_years = np.unique(ds.time.dt.year.values)
        ds = ds.sel(time=ds.time.dt.year.isin(range(MIN_YEAR, MAX_YEAR + 1)))
        trimmed_years = np.unique(ds.time.dt.year.values)
        print(f"  Trimmed years to {MIN_YEAR}-{MAX_YEAR}. Kept {len(trimmed_years)}/{len(original_years)} years.")

        if len(trimmed_years) == 0:
            print("  No data left after trimming to year range. Stopping.")
            return True

        print(f"  Dataset info: {len(ds.time)} time steps, {list(ds.data_vars.keys())} variables")
        
        # Check for NaNs
        has_nans = any(ds[var].isnull().any() for var in ds.data_vars)
        
        if has_nans:
            ds_filled = fill_nans_xarray(ds)
        else:
            print("  No NaNs found.")
            ds_filled = ds
        
        print("  Writing yearly files...")
        write_yearly_files(ds_filled, file_path, dest_dir)
        
        ds.close()
        ds_filled.close()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"  ✓ Completed in {elapsed:.1f}s")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {os.path.basename(file_path)}: {str(e)}")
        return False


def main():
    """Main function to process climate model data."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Fill and yearly split climate model data")
    parser.add_argument("--model", type=str, required=True, help="Climate model name")
    parser.add_argument("--variant", type=str, required=True, help="Model variant")
    parser.add_argument("--scenario", type=str, required=True, help="Climate scenario")
    parser.add_argument("--variable", type=str, required=True, help="Climate variable")
    parser.add_argument("--grid", type=str, required=True, help="Grid type")
    parser.add_argument("--time_period", type=str, required=True, help="Time period of the data")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.file_path):
        print(f"Error: Input file does not exist: {args.file_path}")
        return 1
    
    # Create destination directory
    dest_dir = define_dest_dir(
        args.model, args.variant, args.scenario, 
        args.variable, args.grid, args.time_period
    )
    
    print(f"Processing climate model data:")
    print(f"  Model: {args.model}")
    print(f"  Variant: {args.variant}")
    print(f"  Scenario: {args.scenario}")
    print(f"  Variable: {args.variable}")
    print(f"  Grid: {args.grid}")
    print(f"  Time period: {args.time_period}")
    print(f"  Input file: {args.file_path}")
    print(f"  Output directory: {dest_dir}")
    print(f"  Year range: {MIN_YEAR}-{MAX_YEAR}")
    print()
    
    # Process the file
    success = process_file(args.file_path, dest_dir)
    
    if success:
        print("\n✓ Processing completed successfully!")
        return 0
    else:
        print("\n✗ Processing failed!")
        return 1


if __name__ == "__main__":
    exit(main())