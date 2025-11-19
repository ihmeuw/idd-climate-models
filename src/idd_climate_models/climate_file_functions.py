import argparse
import sys
import os
import subprocess
from pathlib import Path
import xarray as xr
import numpy as np
from xarray.backends.file_manager import FILE_CACHE
import idd_climate_models.constants as rfc

# --- PREPROCESSING CONSTANTS ---
NON_STANDARD_COORDS = {'latitude': 'lat', 'longitude': 'lon'}
# -----------------------------------------------------------------------

def is_curvilinear_grid(file_path):
    """
    Check if a NetCDF file has a curvilinear grid (i/j dimensions with 2D lat/lon).
    """
    try:
        ds = xr.open_dataset(Path(file_path))
        is_curvi = ('i' in ds.dims and 'j' in ds.dims) or \
                   ('latitude' in ds.coords and ds['latitude'].ndim == 2)
        ds.close()
        return is_curvi
    except Exception as e:
        print(f"    ⚠️  Error checking grid type: {e}")
        return False


def regrid_with_cdo(input_path, output_path, target_grid='r360x180'):
    """
    Use CDO to regrid curvilinear grid to regular lon/lat grid.
    
    Args:
        input_path: Path to curvilinear NetCDF file
        output_path: Path for regridded output
        target_grid: CDO grid specification (default: r360x180 = 1°x1° global)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"    → Regridding with CDO (target: {target_grid})...")
        
        result = subprocess.run([
            'cdo', '-f', 'nc',
            f'remapbil,{target_grid}',
            str(input_path),
            str(output_path)
        ], check=True, capture_output=True, text=True)
        
        print(f"    ✅ Regridded to regular grid")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"    ❌ CDO regridding failed: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"    ❌ CDO not found. Install with: conda install -c conda-forge cdo")
        return False

def fix_and_save_dataset(ds, output_path, variable, was_regridded=False, compression_level=7):
    """
    Applies coordinate normalization and saves the dataset.
    
    Args:
        was_regridded: If True, skip coordinate renaming (CDO already standardized names)
    """
    
    # 1. Apply Coordinate Normalization (only if NOT regridded by CDO)
    if not was_regridded:
        renames = {
            old: new 
            for old, new in NON_STANDARD_COORDS.items() 
            if old in ds.coords and old not in ds.dims
        }
        
        if renames:
            ds = ds.rename(renames)
            print(f"    → Renamed coordinates: {renames}")
            
            # Drop any leftover 2D coordinate arrays
            coords_to_drop = [old for old in NON_STANDARD_COORDS.keys() 
                             if old in ds.coords and ds[old].ndim == 2]
            if coords_to_drop:
                ds = ds.drop_vars(coords_to_drop)
                print(f"    → Dropped 2D coordinates: {coords_to_drop}")
    else:
        print(f"    → Skipping coordinate normalization (already handled by CDO)")
    
    # Define compression encoding
    encoding = {
        var: {
            'zlib': True, 
            'complevel': compression_level,
            'shuffle': True,
            'chunksizes': None
        } 
        for var in ds.data_vars
    }
    
    # Write to output file
    ds.to_netcdf(output_path, encoding=encoding, engine='netcdf4')
    
    # Make file group-writable
    try:
        os.chmod(output_path, 0o775)
    except (OSError, PermissionError) as e:
        print(f"    ⚠️  Could not chmod file: {e}")

    try:
        ds.close()
    except Exception:
        pass


def recombine_variable_files(file_paths, output_path, variable, compression_level=7):
    """
    Combine multiple NetCDF files into a single file.
    Automatically regrids curvilinear grids to regular grids using CDO.
    
    Returns:
        tuple: (success: bool, was_regridded: bool)
    """
    if not file_paths:
        return False, False
    
    try:
        # Check if first file is curvilinear
        needs_regridding = is_curvilinear_grid(file_paths[0])
        
        # Regrid all files if needed
        processing_paths = file_paths
        temp_dir = None
        
        if needs_regridding:
            print(f"    → Curvilinear grid detected, regridding {len(file_paths)} files...")
            temp_dir = output_path.parent / f"temp_regrid_{output_path.stem}"
            temp_dir.mkdir(exist_ok=True)
            
            regridded_paths = []
            for i, fpath in enumerate(file_paths):
                regrid_path = temp_dir / f"regrid_{i}_{Path(fpath).name}"
                
                if regrid_with_cdo(fpath, regrid_path):
                    regridded_paths.append(regrid_path)
                else:
                    print(f"    ⚠️  Failed to regrid {fpath}, skipping...")
            
            if not regridded_paths:
                print(f"    ❌ No files successfully regridded")
                return False, False
            
            processing_paths = regridded_paths
            print(f"    ✅ Regridded {len(regridded_paths)}/{len(file_paths)} files")
        
        # Open and combine datasets with time decoding enabled
        datasets = [xr.open_dataset(fpath, decode_times=True) for fpath in processing_paths]
        combined = xr.concat(datasets, dim='time', data_vars='all').sortby('time')
        
        # Close source datasets immediately after concatenation
        for ds in datasets:
            ds.close()
        
        # Apply fixes (renaming, coordinate cleanup) and save with compression
        fix_and_save_dataset(combined, output_path, variable, 
                    was_regridded=needs_regridding, 
                    compression_level=compression_level)
        
        # Clean up temporary regridded files
        if temp_dir and temp_dir.exists():
            for temp_file in temp_dir.glob('*'):
                temp_file.unlink()
            temp_dir.rmdir()
            print(f"    → Cleaned up temporary regridded files")
        
        # Clear file cache after successful write
        FILE_CACHE.clear()
        
        return True, needs_regridding
        
    except Exception as e:
        print(f"    ❌ Error combining files: {e}")
        import traceback
        traceback.print_exc()
        FILE_CACHE.clear()
        return False, False