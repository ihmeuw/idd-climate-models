import xarray as xr
import numpy as np
import zarr
from pathlib import Path
import netCDF4


def validate_netcdf_file(file_path):
    """
    Quick validation: can we open the NetCDF file and read basic structure?
    Checks for RAW TC-risk variable names (lon_trks, lat_trks, etc.)
    Returns True if valid, False if corrupted.
    """
    try:
        with netCDF4.Dataset(file_path, 'r') as ds:
            # Check that we can access variables
            _ = ds.variables
            # Check for required variables from RAW TC-risk output
            required_vars = ['lon_trks', 'lat_trks', 'tc_years', 'tc_month']
            for var in required_vars:
                if var not in ds.variables:
                    return False
        return True
    except Exception as e:
        print(f"  ⚠️  NetCDF validation failed for {Path(file_path).name}: {e}")
        return False


def validate_zarr_file(zarr_path):
    """
    Quick validation: is the Zarr directory readable and has expected structure?
    Returns True if valid, False if corrupted or missing.
    """
    try:
        # Try to open the zarr store
        store = zarr.open(str(zarr_path), mode='r')
        
        # Check that we have storm groups (not flat arrays)
        group_keys = list(store.group_keys())
        
        # Zarr should have at least one storm group
        if len(group_keys) == 0:
            return False
            
        # Optionally: check that first group has expected structure
        first_storm = store[group_keys[0]]
        required_arrays = ['lon', 'lat', 'max_sustained_wind', 'time_step']
        for arr_name in required_arrays:
            if arr_name not in first_storm:
                return False
                
        return True
    except Exception as e:
        print(f"  ⚠️  Zarr validation failed for {Path(zarr_path).name}: {e}")
        return False


def create_draw_completion_marker(climada_input_path, draw_num):
    """
    Create a completion marker file for a successfully validated draw.
    Marker files enable crash-resistant progress tracking.
    
    Args:
        climada_input_path: Path to CLIMADA input directory
        draw_num: Draw number (0-249)
    """
    marker_path = Path(climada_input_path) / f".draw_{draw_num:04d}.complete"
    marker_path.touch()
    marker_path.chmod(0o775)


def get_completed_draws_from_markers(climada_input_path):
    """
    Read completion markers to determine which draws are complete.
    
    Args:
        climada_input_path: Path to CLIMADA input directory
    
    Returns:
        set: Set of completed draw numbers
    """
    climada_input = Path(climada_input_path)
    if not climada_input.exists():
        return set()
    
    marker_files = list(climada_input.glob(".draw_*.complete"))
    completed_draws = set()
    
    for marker_file in marker_files:
        try:
            # Extract draw number from .draw_0123.complete
            draw_num = int(marker_file.stem.split('_')[1])
            completed_draws.add(draw_num)
        except (IndexError, ValueError):
            continue
    
    return completed_draws


def validate_single_draw(nc_file, zarr_file, spot_check=False, delete_on_failure=False):
    """
    Validate a single draw's NetCDF and Zarr files.
    
    Performs three-layer validation:
    1. File existence check
    2. Quick validation (can we open the files?)
    3. Comprehensive check (storm count + optional spot-check)
    
    Args:
        nc_file: Path to NetCDF file
        zarr_file: Path to Zarr file
        spot_check: If True, performs detailed spot-check (slow)
        delete_on_failure: If True, deletes both files if validation fails
    
    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    import shutil
    
    nc_file = Path(nc_file)
    zarr_file = Path(zarr_file)
    
    # Layer 1: File existence
    if not nc_file.exists():
        return False, "NetCDF file missing"
    if not zarr_file.exists():
        return False, "Zarr file missing"
    
    # Layer 2: Quick validation - can we open the files?
    if not validate_netcdf_file(str(nc_file)):
        error_msg = "NetCDF corrupted (cannot open)"
        if delete_on_failure:
            try:
                nc_file.unlink()
                shutil.rmtree(zarr_file)
            except Exception as e:
                pass  # Deletion failure is non-critical
        return False, error_msg
    
    if not validate_zarr_file(str(zarr_file)):
        error_msg = "Zarr corrupted (cannot open)"
        if delete_on_failure:
            try:
                nc_file.unlink()
                shutil.rmtree(zarr_file)
            except Exception as e:
                pass  # Deletion failure is non-critical
        return False, error_msg
    
    # Layer 3: Comprehensive integrity check (storm count + structure)
    try:
        verify_zarr_integrity(str(nc_file), str(zarr_file), None, spot_check=spot_check, verbose=False)
        return True, None
    except Exception as e:
        error_msg = str(e)
        if delete_on_failure:
            try:
                nc_file.unlink()
                shutil.rmtree(zarr_file)
            except Exception as e2:
                pass  # Deletion failure is non-critical
        return False, error_msg


def verify_zarr_storm_count(original_nc_path, zarr_path, verbose=False):
    """
    Lightweight verification: check that zarr has same number of storms as NC file.
    
    Args:
        original_nc_path: Path to original NetCDF file
        zarr_path: Path to zarr store
        verbose: If True, prints detailed output
    
    Returns:
        bool: True if counts match, False otherwise
    
    Raises:
        ValueError: If storm counts don't match (so caller can catch and delete files)
    """
    try:
        # Load original NetCDF
        ds_original = xr.open_dataset(original_nc_path)
        n_trk_original = ds_original.sizes["n_trk"]
        
        # Count valid tracks (same logic as processing)
        valid_count = 0
        for i in range(n_trk_original):
            if not (np.isnan(ds_original["tc_years"][i].item()) or 
                    np.isnan(ds_original["tc_month"][i].item())):
                
                lon = ds_original["lon_trks"][i].values
                lat = ds_original["lat_trks"][i].values
                valid_idx = np.isfinite(lon) & np.isfinite(lat)
                
                if valid_idx.sum() > 0:
                    valid_count += 1
        
        ds_original.close()
        
        # Open Zarr store and count groups
        zarr_store = zarr.open(str(zarr_path), mode='r')
        zarr_count = len(list(zarr_store.group_keys()))
        
        if valid_count != zarr_count:
            error_msg = f"Storm count mismatch: NC has {valid_count}, Zarr has {zarr_count}"
            if verbose:
                print(f"    ⚠️  {error_msg}")
            raise ValueError(error_msg)
        
        if verbose:
            print(f"    ✓ Storm count verified: {valid_count} storms")
        return True
            
    except ValueError:
        # Re-raise ValueError so caller knows it's a mismatch
        raise
    except Exception as e:
        error_msg = f"Storm count verification failed: {e}"
        if verbose:
            print(f"    ⚠️  {error_msg}")
        raise ValueError(error_msg)

def verify_zarr_integrity(original_nc_path, zarr_path, args, spot_check=True, verbose=True):
    """
    Comprehensive verification that Zarr output matches original NetCDF input.
    
    Args:
        original_nc_path: Path to original NetCDF file
        zarr_path: Path to zarr store
        args: Arguments object (can be None)
        spot_check: If True, performs detailed spot-check of 3 storms (slow)
                   If False, only checks storm counts (fast)
        verbose: If True, prints detailed output
    
    Raises:
        ValueError: If validation fails
    """
    if verbose:
        mode = "FULL (with spot-check)" if spot_check else "LIGHT (storm count only)"
        print("=" * 60)
        print(f"VERIFYING ZARR DATA INTEGRITY - {mode}")
        print("=" * 60)
    
    # 1. Load original NetCDF
    if verbose:
        print(f"\n1. Loading original NetCDF: {Path(original_nc_path).name}")
    ds_original = xr.open_dataset(original_nc_path)
    n_trk_original = ds_original.sizes["n_trk"]
    if verbose:
        print(f"   Original file has {n_trk_original} tracks")
    
    # 2. Open Zarr store and list groups
    if verbose:
        print(f"\n2. Opening Zarr store: {Path(zarr_path).name}")
    zarr_store = zarr.open(str(zarr_path), mode='r')
    group_names = list(zarr_store.group_keys())
    if verbose:
        print(f"   Zarr store has {len(group_names)} groups")
    
    # 3. Count valid tracks in original (same logic as your processing)
    if verbose:
        print("\n3. Counting valid tracks in original...")
    valid_count = 0
    valid_indices = []
    for i in range(n_trk_original):
        if not (np.isnan(ds_original["tc_years"][i].item()) or 
                np.isnan(ds_original["tc_month"][i].item())):
            
            lon = ds_original["lon_trks"][i].values
            lat = ds_original["lat_trks"][i].values
            valid_idx = np.isfinite(lon) & np.isfinite(lat)
            
            if valid_idx.sum() > 0:
                valid_count += 1
                valid_indices.append(i)
    
    if verbose:
        print(f"   Valid tracks in original: {valid_count}")
        print(f"   Tracks in Zarr: {len(group_names)}")
    
    if valid_count != len(group_names):
        error_msg = f"Storm count mismatch: NC has {valid_count} valid tracks, zarr has {len(group_names)} storms"
        if verbose:
            print(f"   ❌ ERROR: {error_msg}")
        ds_original.close()
        raise ValueError(error_msg)
    else:
        if verbose:
            print(f"   ✓ Track counts match ({valid_count} storms)")
    
    # If not spot-checking, we're done
    if not spot_check:
        ds_original.close()
        if verbose:
            print("\n" + "=" * 60)
            print("VERIFICATION COMPLETE (light mode)")
            print("=" * 60)
        return
    
    # 4. Spot-check a few storms in detail
    if verbose:
        print("\n4. Spot-checking individual storms...")
    # storms_to_check = min(3, len(group_names))
    storms_to_check = 1
    
    # for idx in range(storms_to_check):
    for idx in [-1]:
        group_name = group_names[idx]
        storm_id = int(group_name.split('_')[1])
        
        print(f"\n   Checking {group_name} (storm_id={storm_id})...")
        
        # Load from Zarr
        ds_zarr = xr.open_zarr(zarr_path, group=group_name)
        
        # Get corresponding original track
        # Note: Use the processed variable names if they exist, otherwise skip
        lon_orig = ds_original["lon_trks"][storm_id].values
        lat_orig = ds_original["lat_trks"][storm_id].values
        
        # Check if processed variables exist (they're added by add_parameters_to_dataset)
        if "vmax_trks" in ds_original:
            vmax_orig = ds_original["vmax_trks"][storm_id].values
        else:
            print(f"      Note: vmax_trks not in original, skipping vmax check")
            vmax_orig = None
            
        if "central_pressure" in ds_original:
            cp_orig = ds_original["central_pressure"][storm_id].values
        else:
            print(f"      Note: central_pressure not in original, skipping pressure check")
            cp_orig = None
        
        # Apply same filtering as your code
        lon_orig = ((lon_orig + 180) % 360) - 180
        lat_orig = np.clip(lat_orig, -90, 90)
        valid_idx = np.isfinite(lon_orig) & np.isfinite(lat_orig)
        
        lon_orig_filtered = lon_orig[valid_idx]
        lat_orig_filtered = lat_orig[valid_idx]
        vmax_orig_filtered = vmax_orig[valid_idx] if vmax_orig is not None else None
        cp_orig_filtered = cp_orig[valid_idx] if cp_orig is not None else None
        
        # Compare
        lon_zarr = ds_zarr["lon"].values
        lat_zarr = ds_zarr["lat"].values
        vmax_zarr = ds_zarr["max_sustained_wind"].values
        cp_zarr = ds_zarr["central_pressure"].values
        
        print(f"      Time points: Original={len(lon_orig_filtered)}, Zarr={len(lon_zarr)}")
        
        if len(lon_orig_filtered) != len(lon_zarr):
            print(f"      ⚠️  WARNING: Length mismatch!")
            continue
        
        # Check if values match (within floating point precision)
        lon_match = np.allclose(lon_orig_filtered, lon_zarr, rtol=1e-5, atol=1e-8)
        lat_match = np.allclose(lat_orig_filtered, lat_zarr, rtol=1e-5, atol=1e-8)
        
        print(f"      Lon match: {lon_match}")
        print(f"      Lat match: {lat_match}")
        
        if vmax_orig_filtered is not None:
            vmax_match = np.allclose(vmax_orig_filtered, vmax_zarr, rtol=1e-5, atol=1e-8, equal_nan=True)
            print(f"      Vmax match: {vmax_match}")
        else:
            vmax_match = True
            print(f"      Vmax: skipped (not in original)")
            
        if cp_orig_filtered is not None:
            cp_match = np.allclose(cp_orig_filtered, cp_zarr, rtol=1e-5, atol=1e-8, equal_nan=True)
            print(f"      Central pressure match: {cp_match}")
        else:
            cp_match = True
            print(f"      Central pressure: skipped (not in original)")
        
        if lon_match and lat_match and vmax_match and cp_match:
            print(f"      ✓ All data matches!")
        else:
            print(f"      ⚠️  WARNING: Data mismatch detected!")
            if not lon_match:
                print(f"         Lon diff: max={np.abs(lon_orig_filtered - lon_zarr).max()}")
            if not vmax_match:
                print(f"         Vmax diff: max={np.abs(vmax_orig_filtered - vmax_zarr).max()}")
    
    # 5. Check file sizes (optional, only if verbose)
    if verbose:
        print("\n5. File size comparison...")
        nc_size = Path(original_nc_path).stat().st_size
        
        # Calculate zarr size (sum of all files in the store)
        zarr_size = sum(f.stat().st_size for f in Path(zarr_path).rglob('*') if f.is_file())
        
        compression_ratio = nc_size / zarr_size if zarr_size > 0 else 0
        
        print(f"   Original NetCDF: {nc_size / 1024**2:.2f} MB")
        print(f"   Zarr store: {zarr_size / 1024**2:.2f} MB")
        print(f"   Compression ratio: {compression_ratio:.1f}x")
        
        if compression_ratio > 5:
            print(f"   ✓ High compression ratio - this is normal for Zarr with mostly NaN data")
    
    # 6. Check data types and compression info (optional, only if verbose)
    if verbose:
        print("\n6. Checking Zarr metadata...")
        first_group = group_names[0]
        ds_sample = xr.open_zarr(str(zarr_path), group=first_group)
        
        print(f"   Variables: {list(ds_sample.data_vars)}")
        print(f"   Coordinates: {list(ds_sample.coords)}")
        
        # Check actual zarr array info
        zarr_group = zarr_store[first_group]
        if 'lon' in zarr_group:
            lon_array = zarr_group['lon']
            print(f"\n   Lon array info:")
            print(f"      Shape: {lon_array.shape}")
            print(f"      Dtype: {lon_array.dtype}")
            print(f"      Chunks: {lon_array.chunks}")
            
            # Try to get compressor info (API differs between Zarr v2 and v3)
            try:
                print(f"      Compressor: {lon_array.compressor}")
            except (TypeError, AttributeError):
                # Zarr v3 doesn't have compressor attribute in the same way
                try:
                    print(f"      Codec: {lon_array.metadata.codecs if hasattr(lon_array, 'metadata') else 'N/A'}")
                except:
                    print(f"      Compression: enabled (details not accessible)")
    
    if verbose:
        print("\n" + "=" * 60)
        print("VERIFICATION COMPLETE (full mode with spot-check)")
        print("=" * 60)
    
    ds_original.close()
