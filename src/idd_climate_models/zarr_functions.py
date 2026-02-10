import xarray as xr
import numpy as np
import zarr
from pathlib import Path

def verify_zarr_storm_count(original_nc_path, zarr_path):
    """
    Lightweight verification: check that zarr has same number of storms as NC file.
    Returns True if counts match, False otherwise.
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
        zarr_store = zarr.open(zarr_path, mode='r')
        zarr_count = len(list(zarr_store.group_keys()))
        
        if valid_count == zarr_count:
            print(f"    ✓ Storm count verified: {valid_count} storms")
            return True
        else:
            print(f"    ⚠️  Storm count mismatch: NC has {valid_count}, Zarr has {zarr_count}")
            return False
            
    except Exception as e:
        print(f"    ⚠️  Verification error: {e}")
        return False

def verify_zarr_integrity(original_nc_path, zarr_path, args):
    """
    Comprehensive verification that Zarr output matches original NetCDF input.
    """
    print("=" * 60)
    print("VERIFYING ZARR DATA INTEGRITY")
    print("=" * 60)
    
    # 1. Load original NetCDF
    print(f"\n1. Loading original NetCDF: {original_nc_path}")
    ds_original = xr.open_dataset(original_nc_path)
    n_trk_original = ds_original.sizes["n_trk"]
    print(f"   Original file has {n_trk_original} tracks")
    
    # 2. Open Zarr store and list groups
    print(f"\n2. Opening Zarr store: {zarr_path}")
    zarr_store = zarr.open(zarr_path, mode='r')
    group_names = list(zarr_store.group_keys())
    print(f"   Zarr store has {len(group_names)} groups: {group_names[:5]}...")
    
    # 3. Count valid tracks in original (same logic as your processing)
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
    
    print(f"   Valid tracks in original: {valid_count}")
    print(f"   Tracks in Zarr: {len(group_names)}")
    
    if valid_count != len(group_names):
        error_msg = f"Storm count mismatch: NC has {valid_count} valid tracks, zarr has {len(group_names)} storms"
        print(f"   ❌ ERROR: {error_msg}")
        raise ValueError(error_msg)
    else:
        print(f"   ✓ Track counts match")
    
    # 4. Spot-check a few storms in detail
    print("\n4. Spot-checking individual storms...")
    storms_to_check = min(3, len(group_names))
    
    for idx in range(storms_to_check):
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
    
    # 5. Check file sizes
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
    
    # 6. Check data types and compression info
    print("\n6. Checking Zarr metadata...")
    first_group = group_names[0]
    ds_sample = xr.open_zarr(zarr_path, group=first_group)
    
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
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    
    ds_original.close()
