"""
Diagnostic script to figure out why parameter estimation is returning NaN values.
"""

import numpy as np
import xarray as xr
from pathlib import Path

def diagnose_track_data(file_path, year_idx=0, track_idx=0):
    """
    Inspect the raw track data to diagnose issues.
    
    Parameters
    ----------
    file_path : str
        Path to NetCDF file
    year_idx : int
        Year index
    track_idx : int
        Track index
    """
    print("="*70)
    print("DIAGNOSTIC: INSPECTING TRACK DATA")
    print("="*70)
    
    # Load dataset
    ds = xr.open_dataset(file_path)
    
    print(f"\nDataset dimensions:")
    for dim, size in ds.dims.items():
        print(f"  {dim}: {size}")
    
    print(f"\nDataset coordinates:")
    for coord in ds.coords:
        print(f"  {coord}: {ds[coord].values}")
    
    print(f"\nDataset variables:")
    for var in ds.data_vars:
        print(f"  {var}: shape={ds[var].shape}, dtype={ds[var].dtype}")
    
    # Select track
    print(f"\n" + "-"*70)
    print(f"Examining track at year_idx={year_idx}, track_idx={track_idx}")
    print("-"*70)
    
    track = ds.isel(year=year_idx, n_trk=track_idx)
    
    # Check each required variable
    required_vars = ['lon_trks', 'lat_trks', 'vmax_trks', 'u850_trks', 'v850_trks', 'u250_trks', 'v250_trks']
    
    print("\nRequired variables check:")
    for var in required_vars:
        if var in track:
            data = track[var].values
            n_valid = np.sum(~np.isnan(data))
            n_total = len(data)
            print(f"  {var}:")
            print(f"    Shape: {data.shape}")
            print(f"    Valid values: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
            print(f"    Range: [{np.nanmin(data):.4f}, {np.nanmax(data):.4f}]")
            print(f"    Mean: {np.nanmean(data):.4f}")
            
            # Check for all zeros or all same value
            non_nan_data = data[~np.isnan(data)]
            if len(non_nan_data) > 0:
                if np.all(non_nan_data == 0):
                    print(f"    ⚠️  WARNING: All values are zero!")
                elif np.std(non_nan_data) < 1e-10:
                    print(f"    ⚠️  WARNING: No variance (all same value)!")
        else:
            print(f"  {var}: ❌ MISSING!")
    
    # Extract valid data
    print("\n" + "-"*70)
    print("Extracting valid data for estimation:")
    print("-"*70)
    
    lon = track.lon_trks.values
    lat = track.lat_trks.values
    vmax_ms = track.vmax_trks.values
    
    # Find valid mask
    valid_mask = ~np.isnan(lat)
    print(f"\nValid time steps: {np.sum(valid_mask)}/{len(lat)}")
    
    if np.sum(valid_mask) == 0:
        print("❌ ERROR: No valid data points found!")
        ds.close()
        return
    
    # Extract valid data
    lon_valid = lon[valid_mask]
    lat_valid = lat[valid_mask]
    vmax_valid = vmax_ms[valid_mask]
    
    print(f"\nValid data ranges:")
    print(f"  Longitude: [{np.min(lon_valid):.2f}, {np.max(lon_valid):.2f}]°E")
    print(f"  Latitude: [{np.min(lat_valid):.2f}, {np.max(lat_valid):.2f}]°N")
    print(f"  Vmax: [{np.min(vmax_valid):.2f}, {np.max(vmax_valid):.2f}] m/s")
    print(f"  Vmax: [{np.min(vmax_valid)*1.943844:.2f}, {np.max(vmax_valid)*1.943844:.2f}] kt")
    
    # Test pressure estimation
    print("\n" + "-"*70)
    print("Testing pressure estimation:")
    print("-"*70)
    
    from idd_climate_models.tc_estimation import _estimate_pressure
    
    vmax_kt = vmax_valid * 1.943844
    
    print(f"\nInput to _estimate_pressure:")
    print(f"  cen_pres: all NaN (shape={vmax_kt.shape})")
    print(f"  lat: min={np.min(lat_valid):.2f}, max={np.max(lat_valid):.2f}")
    print(f"  lon: min={np.min(lon_valid):.2f}, max={np.max(lon_valid):.2f}")
    print(f"  v_max (kt): min={np.min(vmax_kt):.2f}, max={np.max(vmax_kt):.2f}")
    
    # Test the function
    try:
        pressure_est = _estimate_pressure(
            np.full(vmax_kt.shape, np.nan),
            lat_valid,
            lon_valid,
            vmax_kt
        )
        
        n_valid_pressure = np.sum(~np.isnan(pressure_est))
        print(f"\n✓ Estimation completed")
        print(f"  Valid pressure values: {n_valid_pressure}/{len(pressure_est)}")
        
        if n_valid_pressure > 0:
            print(f"  Pressure range: [{np.nanmin(pressure_est):.1f}, {np.nanmax(pressure_est):.1f}] hPa")
            print(f"  Mean pressure: {np.nanmean(pressure_est):.1f} hPa")
        else:
            print(f"  ❌ All pressure values are NaN!")
            
            # Debug the estimation function
            print("\n  Debugging _estimate_pressure logic:")
            cen_pres = np.where(np.isnan(np.full(vmax_kt.shape, np.nan)), -1, np.full(vmax_kt.shape, np.nan))
            v_max_check = np.where(np.isnan(vmax_kt), -1, vmax_kt)
            lat_check = np.where(np.isnan(lat_valid), -999, lat_valid)
            lon_check = np.where(np.isnan(lon_valid), -999, lon_valid)
            msk = (cen_pres <= 0) & (v_max_check > 0) & (lat_check > -999) & (lon_check > -999)
            
            print(f"    cen_pres <= 0: {np.sum(cen_pres <= 0)}")
            print(f"    v_max > 0: {np.sum(v_max_check > 0)}")
            print(f"    lat > -999: {np.sum(lat_check > -999)}")
            print(f"    lon > -999: {np.sum(lon_check > -999)}")
            print(f"    Final mask (all conditions): {np.sum(msk)}")
            
    except Exception as e:
        print(f"\n❌ Error during estimation: {e}")
        import traceback
        traceback.print_exc()
    
    # Test RMW estimation
    print("\n" + "-"*70)
    print("Testing RMW estimation:")
    print("-"*70)
    
    from idd_climate_models.tc_estimation import estimate_rmw
    
    if n_valid_pressure > 0:
        try:
            rmw_est = estimate_rmw(
                np.full(pressure_est.shape, np.nan),
                pressure_est
            )
            
            n_valid_rmw = np.sum(~np.isnan(rmw_est))
            print(f"✓ RMW estimation completed")
            print(f"  Valid RMW values: {n_valid_rmw}/{len(rmw_est)}")
            
            if n_valid_rmw > 0:
                print(f"  RMW range: [{np.nanmin(rmw_est):.1f}, {np.nanmax(rmw_est):.1f}] nm")
        except Exception as e:
            print(f"❌ Error during RMW estimation: {e}")
    
    # Check if track has new parameters already
    print("\n" + "-"*70)
    print("Checking for existing parameters:")
    print("-"*70)
    
    new_params = ['central_pressure', 'environmental_pressure', 'rmw', 'roci']
    for param in new_params:
        if param in track:
            data = track[param].values
            n_valid = np.sum(~np.isnan(data))
            print(f"  {param}: {n_valid}/{len(data)} valid values")
            if n_valid > 0:
                print(f"    Range: [{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]")
        else:
            print(f"  {param}: Not in dataset")
    
    ds.close()
    
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)


def test_estimation_directly(file_path, year_idx=0, track_idx=0):
    """
    Test the estimation functions directly on a track.
    """
    print("\n" + "="*70)
    print("DIRECT ESTIMATION TEST")
    print("="*70)
    
    ds = xr.open_dataset(file_path)
    track = ds.isel(year=year_idx, n_trk=track_idx)
    
    # Extract data
    lon = track.lon_trks.values
    lat = track.lat_trks.values
    vmax_ms = track.vmax_trks.values
    u850 = track.u850_trks.values
    v850 = track.v850_trks.values
    u250 = track.u250_trks.values
    v250 = track.v250_trks.values
    
    # Remove NaN
    valid = ~np.isnan(lat)
    lon = lon[valid]
    lat = lat[valid]
    vmax_ms = vmax_ms[valid]
    u850 = u850[valid]
    v850 = v850[valid]
    u250 = u250[valid]
    v250 = v250[valid]
    
    print(f"\nProcessing {len(lat)} valid points")
    
    # Convert to knots
    vmax_kt = vmax_ms * 1.943844
    
    print(f"Wind speed range: {np.min(vmax_kt):.1f} - {np.max(vmax_kt):.1f} kt")
    
    # Try ensemble estimation
    from idd_climate_models.tc_estimation import ensemble_estimates
    
    try:
        print("\nRunning ensemble_estimates...")
        estimates = ensemble_estimates(lat, lon, vmax_kt, u850, v850, u250, v250)
        
        print("\n✓ Estimation successful!")
        
        for param in ['central_pressure', 'rmw', 'roci']:
            data = estimates[param]['mean']
            n_valid = np.sum(~np.isnan(data))
            print(f"\n{param}:")
            print(f"  Valid values: {n_valid}/{len(data)}")
            if n_valid > 0:
                print(f"  Range: [{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]")
                print(f"  Mean: {np.nanmean(data):.2f}")
            else:
                print(f"  ❌ All NaN!")
        
        return estimates
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        ds.close()


# Example usage
if __name__ == "__main__":
    print("USAGE:")
    print("="*70)
    print("from diagnose_issue import diagnose_track_data, test_estimation_directly")
    print("")
    print("# Diagnose what's in the file")
    print("diagnose_track_data('your_file.nc', year_idx=0, track_idx=0)")
    print("")
    print("# Test estimation directly")
    print("estimates = test_estimation_directly('your_file.nc', year_idx=0, track_idx=0)")