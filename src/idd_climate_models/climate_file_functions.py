import argparse
import sys
import os
import subprocess
from pathlib import Path
import xarray as xr
import numpy as np
from typing import Any, Optional
from xarray.backends.file_manager import FILE_CACHE
import idd_climate_models.constants as rfc

# --- PREPROCESSING CONSTANTS ---
NON_STANDARD_COORDS = {'latitude': 'lat', 'longitude': 'lon'}
# -----------------------------------------------------------------------

def should_regrid(file_path, target_grid='r360x180'):
    """
    Check if a NetCDF file needs regridding to target grid.
    Returns True if regridding is needed.
    
    Args:
        file_path: Path to NetCDF file
        target_grid: Target grid specification (e.g., 'r360x180' for 1°×1°)
    
    Returns:
        bool: True if regridding needed, False if already on target grid
    """
    try:
        ds = xr.open_dataset(Path(file_path))
        
        # Parse target grid to get expected dimensions
        # r360x180 means 360 lon points, 180 lat points
        if target_grid.startswith('r'):
            parts = target_grid[1:].split('x')
            target_lon = int(parts[0])
            target_lat = int(parts[1])
        else:
            print(f"    ⚠️  Unknown grid format: {target_grid}")
            ds.close()
            return True  # Assume regridding needed
        
        # Check if it's already on target grid
        has_correct_dims = (
            'lon' in ds.dims and ds.sizes.get('lon') == target_lon and
            'lat' in ds.dims and ds.sizes.get('lat') == target_lat
        )
        
        ds.close()
        
        # Regrid if NOT already on target grid
        return not has_correct_dims
        
    except Exception as e:
        print(f"    ⚠️  Error checking grid: {e}")
        return True  # Assume regridding needed if check fails


def is_curvilinear_grid(file_path):
    """
    Check if a NetCDF file has a curvilinear grid (i/j dimensions with 2D lat/lon).
    
    Note: This function is kept for backwards compatibility, but should_regrid()
    is now the preferred method as it catches both curvilinear grids AND 
    rectangular grids with wrong dimensions.
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
    Use CDO to regrid to regular lon/lat grid.
    
    Args:
        input_path: Path to NetCDF file
        output_path: Path for regridded output
        target_grid: CDO grid specification (default: r360x180 = 1°×1° global)
    
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
        
        print(f"    ✅ Regridded to {target_grid}")
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
        ds: xarray Dataset
        output_path: Path for output file
        variable: Variable name being processed
        was_regridded: If True, skip coordinate renaming (CDO already standardized names)
        compression_level: Compression level for NetCDF output (1-9)
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


def recombine_variable_files(file_paths, output_path, variable, 
                            target_grid='r360x180', compression_level=7):
    """
    Combine multiple NetCDF files into a single file.
    Automatically regrids to target grid if needed (checks dimensions, not just curvilinear).
    
    Args:
        file_paths: List of input file paths
        output_path: Path for combined output file
        variable: Variable name being processed
        target_grid: CDO grid specification (default: r360x180 = 1°×1°)
        compression_level: Compression level for output (1-9)
    
    Returns:
        tuple: (success: bool, was_regridded: bool)
    """
    if not file_paths:
        return False, False
    
    try:
        # Check if first file needs regridding (dimension check catches both 
        # curvilinear grids AND rectangular grids with wrong dimensions)
        needs_regridding = should_regrid(file_paths[0], target_grid)
        
        # Regrid all files if needed
        processing_paths = file_paths
        temp_dir = None
        
        if needs_regridding:
            print(f"    → Grid mismatch detected, regridding {len(file_paths)} files to {target_grid}...")
            temp_dir = output_path.parent / f"temp_regrid_{output_path.stem}"
            temp_dir.mkdir(exist_ok=True)
            
            regridded_paths = []
            for i, fpath in enumerate(file_paths):
                regrid_path = temp_dir / f"regrid_{i}_{Path(fpath).name}"
                
                if regrid_with_cdo(fpath, regrid_path, target_grid):
                    regridded_paths.append(regrid_path)
                else:
                    print(f"    ⚠️  Failed to regrid {fpath}, skipping...")
            
            if not regridded_paths:
                print(f"    ❌ No files successfully regridded")
                return False, False
            
            processing_paths = regridded_paths
            print(f"    ✅ Regridded {len(regridded_paths)}/{len(file_paths)} files")
        else:
            print(f"    ✓ Files already on {target_grid} grid")
        
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

def get_dir(
    args: Any, 
    source: bool = True, 
    until_key: Optional[str] = None
) -> Path:
    """
    Constructs the directory path based on FOLDER_STRUCTURE, allowing 
    for partial path construction up to a specified structure key.
    
    Args:
        args (Any): The object containing the path component attributes.
        source (bool): If True, uses input_data_type/input_io_data_type. 
                       Otherwise, uses output_data_type/output_io_data_type.
        until_key (Optional[str]): If provided, path construction stops 
                                   after appending the value for this key. 
                                   Defaults to None (full path).
    
    Returns:
        Path: The constructed directory path.
    """
    
    # 1. Determine data_type and io_data_type
    if source:
        data_type = args.input_data_type
        io_data_type = args.input_io_data_type
    else:
        data_type = args.output_data_type
        io_data_type = args.output_io_data_type

    # 2. Retrieve the base path and structure keys from FOLDER_STRUCTURE
    structure_dict = rfc.FOLDER_STRUCTURE[data_type][io_data_type]
    base_path: Path = structure_dict['base']
    structure_keys: list[str] = structure_dict['structure']

    # 3. Sequentially join the components from the args object
    current_path = base_path / args.data_source
    for key in structure_keys:
        component = getattr(args, key)
        current_path = current_path / str(component)
        # Check for the stopping condition
        if until_key is not None and key == until_key:
            break

    return current_path


def get_track_path(args: Any, source=True, extension=".nc") -> Path:
    """Constructs the full file path for a track file."""    
    # Draws are 1-based, array indices are 0-based (e.g., draw 1 is _e0)
    draw = args.draw
    draw_text = f'_e{draw - 1}' if draw > 0 else ''
    
    # Use f-strings for time strings, ensuring 4-digit years
    time_period = tuple(map(int, args.time_period.split('-')))
    time_start_str = f'{time_period[0]:04d}01' # Added :04d for explicit 4-digit formatting
    time_end_str = f'{time_period[1]:04d}12'
    
    # File name construction
    track_file = f'tracks_{args.basin}_{args.model}_{args.scenario}_{args.variant}_{time_start_str}_{time_end_str}{draw_text}{extension}'
    
    # The basin directory is a subdirectory of the time_period directory
    return get_dir(args, source=source) / track_file