"""
Add estimated TC parameters (central pressure, environmental pressure, RMW, ROCI)
to tropical cyclone track NetCDF files from the tropical_cyclone_risk model.

This script processes tracks from:
https://github.com/linjonathan/tropical_cyclone_risk
"""

import numpy as np
import xarray as xr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import the estimation functions (assumes tc_estimation.py is in the same directory)
from idd_climate_models.tc_estimation import (
    _estimate_pressure,
    estimate_rmw,
    estimate_roci,
    estimate_pressure_dvorak,
    estimate_rmw_willoughby,
    estimate_rmw_shear,
    estimate_roci_knaff,
    ensemble_estimates
)


def estimate_environmental_pressure(lat, roci_nm=None, default_method='standard'):
    """
    Estimate environmental pressure at the storm's outer boundary.
    
    Environmental pressure is typically estimated by 
    azimuthal-mean pressure at a distance dependent on storm size, ranging from 
    900-2100 km from the storm center.
    
    Parameters
    ----------
    lat : array-like
        Latitude (degrees)
    roci_nm : array-like, optional
        Radius of outermost closed isobar (nautical miles)
        Used to determine appropriate distance for environmental pressure
    default_method : str
        'standard': Use simple latitude-based estimate (1010-1015 hPa range)
        'gradient': Include latitude gradient effect
    
    Returns
    -------
    p_env : np.array
        Environmental pressure (hPa)
        
    Notes
    -----
    Ambient environmental pressure P∞ is typically ~101.3 kPa 
    (1013 hPa) at sea level, but varies with latitude and season.
    
    Without access to full pressure field analysis, we use a simplified approach:
    - Higher latitudes: slightly lower environmental pressure
    - Tropics: near standard sea-level pressure
    """
    lat_abs = np.abs(lat)
    
    if default_method == 'standard':
        # Simple approach: environmental pressure decreases slightly with latitude
        # Based on typical subtropical high pressure patterns
        p_env = 1013.0 - 0.15 * lat_abs  # Ranges from ~1013 at equator to ~1008 at 30°
        
    elif default_method == 'gradient':
        # More sophisticated: account for subtropical high pressure belt
        # Peak pressure at ~30° latitude (subtropical highs)
        subtropical_peak = 30.0
        
        # Pressure is higher in subtropics, lower in deep tropics and mid-latitudes
        lat_effect = -0.5 * (lat_abs - subtropical_peak)**2 / subtropical_peak**2
        p_env = 1015.0 + lat_effect  # Ranges from ~1015 at 30° to ~1013 elsewhere
        
    elif default_method == 'size_aware' and roci_nm is not None:
        # Larger storms may have different environmental pressure characteristics
        # This is a simplified proxy
        NM_TO_KM = 1.852
        roci_km = roci_nm * NM_TO_KM
        
        # Base environmental pressure
        p_env = 1013.0 - 0.1 * lat_abs
        
        # Small adjustment for very large storms (may interact with larger-scale patterns)
        size_adjustment = np.clip((roci_km - 200) / 1000, -2, 2)
        p_env = p_env - size_adjustment
    else:
        # Default: standard sea-level pressure
        p_env = np.full_like(lat, 1013.0)
    
    return np.clip(p_env, 1005, 1020)  # Reasonable bounds


def add_tc_parameters_to_track(ds, track_year, track_num, 
                               use_ensemble=True,
                               env_pressure_method='standard',
                               verbose=False):
    """
    Add estimated TC parameters to a single track in the dataset.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing track data
    track_year : int
        Year index (0-based)
    track_num : int
        Track number index (0-based)
    use_ensemble : bool
        If True, use ensemble estimation. If False, use CLIMADA only.
    env_pressure_method : str
        Method for environmental pressure estimation
    verbose : bool
        Print progress information
    
    Returns
    -------
    track_data : dict
        Dictionary with estimated parameters
    """
    # Select the track using isel (index-based selection) instead of sel
    track = ds.isel(year=track_year, n_trk=track_num)
    
    # Extract data
    lon = track.lon_trks.values
    lat = track.lat_trks.values
    vmax_ms = track.vmax_trks.values
    u850 = track.u850_trks.values
    v850 = track.v850_trks.values
    u250 = track.u250_trks.values
    v250 = track.v250_trks.values
    
    # Remove NaN values (track padding)
    valid_mask = ~np.isnan(lat)
    lon = lon[valid_mask]
    lat = lat[valid_mask]
    vmax_ms = vmax_ms[valid_mask]
    u850 = u850[valid_mask]
    v850 = v850[valid_mask]
    u250 = u250[valid_mask]
    v250 = v250[valid_mask]
    
    if len(lat) == 0:
        return None
    
    # Convert wind speed from m/s to knots
    MS_TO_KN = 1.943844
    vmax_kt = vmax_ms * MS_TO_KN
    
    if verbose:
        year_value = ds.year.values[track_year] if 'year' in ds.coords else track_year
        print(f"  Processing year_idx={track_year} (year={year_value}), track={track_num}: {len(lat)} points, "
              f"max wind={np.max(vmax_kt):.1f} kt")
    
    if use_ensemble:
        # Use ensemble estimation
        estimates = ensemble_estimates(lat, lon, vmax_kt, u850, v850, u250, v250)
        
        central_pressure = estimates['central_pressure']['mean']
        rmw = estimates['rmw']['mean']
        roci = estimates['roci']['mean']
        
        # Also store uncertainties
        central_pressure_std = estimates['central_pressure'].get('std', np.zeros_like(central_pressure))
        rmw_std = estimates['rmw'].get('std', np.zeros_like(rmw))
        roci_std = estimates['roci'].get('std', np.zeros_like(roci))
    else:
        # Use CLIMADA methods only
        central_pressure = _estimate_pressure(
            np.full(vmax_kt.shape, np.nan), lat, lon, vmax_kt
        )
        rmw = estimate_rmw(np.full(vmax_kt.shape, np.nan), central_pressure)
        roci = estimate_roci(np.full(vmax_kt.shape, np.nan), central_pressure)
        
        central_pressure_std = np.zeros_like(central_pressure)
        rmw_std = np.zeros_like(rmw)
        roci_std = np.zeros_like(roci)
    
    # Estimate environmental pressure
    environmental_pressure = estimate_environmental_pressure(
        lat, roci_nm=roci, default_method=env_pressure_method
    )
    
    # Calculate pressure deficit
    pressure_deficit = environmental_pressure - central_pressure
    
    # Prepare output with original array shape (including NaN padding)
    n_time = len(track.time.values)
    
    def pad_to_original(data):
        """Pad data back to original shape with NaN"""
        result = np.full(n_time, np.nan)
        result[valid_mask] = data
        return result
    
    track_data = {
        'central_pressure': pad_to_original(central_pressure),
        'central_pressure_std': pad_to_original(central_pressure_std),
        'environmental_pressure': pad_to_original(environmental_pressure),
        'pressure_deficit': pad_to_original(pressure_deficit),
        'rmw': pad_to_original(rmw),
        'rmw_std': pad_to_original(rmw_std),
        'roci': pad_to_original(roci),
        'roci_std': pad_to_original(roci_std),
    }
    
    return track_data


def add_parameters_to_dataset(input_file, output_file=None,
                              use_ensemble=True,
                              env_pressure_method='standard',
                              overwrite=False,
                              verbose=True):
    """
    Add TC parameters to all tracks in a NetCDF file.
    
    Parameters
    ----------
    input_file : str or Path
        Path to input NetCDF file
    output_file : str or Path, optional
        Path to output NetCDF file. If None, appends '_with_params' to input filename
    use_ensemble : bool
        Use ensemble estimation methods
    env_pressure_method : str
        Environmental pressure estimation method
    overwrite : bool
        Overwrite output file if it exists
    verbose : bool
        Print progress information
    
    Returns
    -------
    output_path : Path
        Path to output file
    """
    input_path = Path(input_file)
    
    if output_file is None:
        output_path = input_path.parent / f"{input_path.stem}_with_params{input_path.suffix}"
    else:
        output_path = Path(output_file)
    
    if output_path.exists() and not overwrite:
        print(f"Output file exists: {output_path}")
        print("Use overwrite=True to replace it.")
        return output_path
    
    if verbose:
        print(f"\nProcessing: {input_path.name}")
        print(f"Output: {output_path.name}")
        print("="*70)
    
    # Load dataset
    ds = xr.open_dataset(input_path)
    
    # Get dimensions - NOTE: data shape is (n_tracks, n_time) not (n_years, n_tracks, n_time)!
    n_tracks = len(ds.n_trk)
    n_time = len(ds.time)
    
    if verbose:
        print(f"Dataset: {n_tracks} tracks, {n_time} time steps")
        print(f"Data shape: ({n_tracks}, {n_time})")
    
    # Initialize arrays for new variables - match the actual data shape!
    central_pressure = np.full((n_tracks, n_time), np.nan)
    central_pressure_std = np.full((n_tracks, n_time), np.nan)
    environmental_pressure = np.full((n_tracks, n_time), np.nan)
    pressure_deficit = np.full((n_tracks, n_time), np.nan)
    rmw = np.full((n_tracks, n_time), np.nan)
    rmw_std = np.full((n_tracks, n_time), np.nan)
    roci = np.full((n_tracks, n_time), np.nan)
    roci_std = np.full((n_tracks, n_time), np.nan)
    
    # Process each track - NOTE: tracks are NOT organized by year in this dataset!
    n_processed = 0
    n_skipped = 0
    
    if verbose:
        print("\nProcessing tracks:")
    
    for track_idx in range(n_tracks):
        try:
            # Get this track directly
            track = ds.isel(n_trk=track_idx)
            
            # Extract data
            lon = track.lon_trks.values
            lat = track.lat_trks.values
            vmax_ms = track.vmax_trks.values
            u850 = track.u850_trks.values
            v850 = track.v850_trks.values
            u250 = track.u250_trks.values
            v250 = track.v250_trks.values
            
            # Remove NaN values (track padding)
            valid_mask = ~np.isnan(lat)
            
            if np.sum(valid_mask) == 0:
                n_skipped += 1
                continue
            
            lon_valid = lon[valid_mask]
            lat_valid = lat[valid_mask]
            vmax_valid = vmax_ms[valid_mask]
            u850_valid = u850[valid_mask]
            v850_valid = v850[valid_mask]
            u250_valid = u250[valid_mask]
            v250_valid = v250[valid_mask]
            
            # Convert wind speed from m/s to knots
            MS_TO_KN = 1.943844
            vmax_kt = vmax_valid * MS_TO_KN
            
            if verbose and track_idx % 20 == 0:
                print(f"  Track {track_idx}: {len(lat_valid)} points, max wind={np.max(vmax_kt):.1f} kt")
            
            # Run estimation
            if use_ensemble:
                from idd_climate_models.tc_estimation import ensemble_estimates
                estimates = ensemble_estimates(lat_valid, lon_valid, vmax_kt, 
                                              u850_valid, v850_valid, u250_valid, v250_valid)
                
                cp = estimates['central_pressure']['mean']
                cp_std = estimates['central_pressure'].get('std', np.zeros_like(cp))
                rmw_est = estimates['rmw']['mean']
                rmw_std_est = estimates['rmw'].get('std', np.zeros_like(rmw_est))
                roci_est = estimates['roci']['mean']
                roci_std_est = estimates['roci'].get('std', np.zeros_like(roci_est))
            else:
                from idd_climate_models.tc_estimation import _estimate_pressure, estimate_rmw, estimate_roci
                cp = _estimate_pressure(np.full(vmax_kt.shape, np.nan), lat_valid, lon_valid, vmax_kt)
                cp_std = np.zeros_like(cp)
                rmw_est = estimate_rmw(np.full(vmax_kt.shape, np.nan), cp)
                rmw_std_est = np.zeros_like(rmw_est)
                roci_est = estimate_roci(np.full(vmax_kt.shape, np.nan), cp)
                roci_std_est = np.zeros_like(roci_est)
            
            # Estimate environmental pressure
            ep = estimate_environmental_pressure(lat_valid, roci_nm=roci_est, 
                                                default_method=env_pressure_method)
            
            # Calculate pressure deficit
            pd = ep - cp
            
            # Pad back to original shape
            central_pressure[track_idx, valid_mask] = cp
            central_pressure_std[track_idx, valid_mask] = cp_std
            environmental_pressure[track_idx, valid_mask] = ep
            pressure_deficit[track_idx, valid_mask] = pd
            rmw[track_idx, valid_mask] = rmw_est
            rmw_std[track_idx, valid_mask] = rmw_std_est
            roci[track_idx, valid_mask] = roci_est
            roci_std[track_idx, valid_mask] = roci_std_est
            
            n_processed += 1
                    
        except Exception as e:
            if verbose:
                print(f"  Error processing track={track_idx}: {e}")
            n_skipped += 1
    
    # Add new variables to dataset - CORRECT DIMENSIONS!
    ds_out = ds.copy()
    
    # Central pressure
    ds_out['central_pressure'] = (('n_trk', 'time'), central_pressure)
    ds_out['central_pressure'].attrs = {
        'long_name': 'Estimated central pressure',
        'units': 'hPa',
        'description': 'Minimum sea-level pressure at storm center',
        'estimation_method': 'ensemble' if use_ensemble else 'climada'
    }
    
    # Central pressure uncertainty
    ds_out['central_pressure_std'] = (('n_trk', 'time'), central_pressure_std)
    ds_out['central_pressure_std'].attrs = {
        'long_name': 'Central pressure estimation uncertainty',
        'units': 'hPa',
        'description': 'Standard deviation of central pressure estimates across methods'
    }
    
    # Environmental pressure
    ds_out['environmental_pressure'] = (('n_trk', 'time'), environmental_pressure)
    ds_out['environmental_pressure'].attrs = {
        'long_name': 'Environmental pressure',
        'units': 'hPa',
        'description': 'Ambient pressure at outer radius of storm influence',
        'estimation_method': env_pressure_method,
        'note': 'Estimated from latitude; ideally should use reanalysis data'
    }
    
    # Pressure deficit
    ds_out['pressure_deficit'] = (('n_trk', 'time'), pressure_deficit)
    ds_out['pressure_deficit'].attrs = {
        'long_name': 'Central pressure deficit',
        'units': 'hPa',
        'description': 'Environmental pressure minus central pressure',
        'formula': 'P_env - P_central'
    }
    
    # RMW
    ds_out['rmw'] = (('n_trk', 'time'), rmw)
    ds_out['rmw'].attrs = {
        'long_name': 'Radius of maximum wind',
        'units': 'nm',
        'description': 'Radius from storm center to maximum wind speed',
        'estimation_method': 'ensemble' if use_ensemble else 'climada'
    }
    
    # RMW uncertainty
    ds_out['rmw_std'] = (('n_trk', 'time'), rmw_std)
    ds_out['rmw_std'].attrs = {
        'long_name': 'RMW estimation uncertainty',
        'units': 'nm',
        'description': 'Standard deviation of RMW estimates across methods'
    }
    
    # ROCI
    ds_out['roci'] = (('n_trk', 'time'), roci)
    ds_out['roci'].attrs = {
        'long_name': 'Radius of outermost closed isobar',
        'units': 'nm',
        'description': 'Outer radius of closed circulation',
        'estimation_method': 'ensemble' if use_ensemble else 'climada'
    }
    
    # ROCI uncertainty
    ds_out['roci_std'] = (('n_trk', 'time'), roci_std)
    ds_out['roci_std'].attrs = {
        'long_name': 'ROCI estimation uncertainty',
        'units': 'nm',
        'description': 'Standard deviation of ROCI estimates across methods'
    }
    
    # Add processing metadata
    ds_out.attrs['tc_parameters_added'] = 'true'
    ds_out.attrs['tc_estimation_date'] = str(np.datetime64('now'))
    ds_out.attrs['tc_estimation_ensemble'] = str(use_ensemble)
    ds_out.attrs['tc_env_pressure_method'] = env_pressure_method
    
    # Save to file
    if verbose:
        print(f"\nSaving to: {output_path}")
    
    ds_out.to_netcdf(output_path)
    
    if verbose:
        print(f"\nSummary:")
        print(f"  Total tracks processed: {n_processed}")
        print(f"  Tracks skipped: {n_skipped}")
        print(f"  Output file size: {output_path.stat().st_size / 1e6:.1f} MB")
        print("="*70)
    
    ds.close()
    ds_out.close()
    
    return output_path


def batch_process_directory(input_dir, output_dir=None, 
                           pattern='*.nc',
                           **kwargs):
    """
    Process all NetCDF files in a directory.
    
    Parameters
    ----------
    input_dir : str or Path
        Directory containing input files
    output_dir : str or Path, optional
        Directory for output files. If None, uses input_dir
    pattern : str
        Glob pattern for finding NetCDF files
    **kwargs : dict
        Additional arguments passed to add_parameters_to_dataset
    
    Returns
    -------
    output_files : list
        List of output file paths
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    input_files = sorted(input_path.glob(pattern))
    
    if len(input_files) == 0:
        print(f"No files found matching pattern '{pattern}' in {input_dir}")
        return []
    
    print(f"Found {len(input_files)} files to process")
    print("="*70)
    
    output_files = []
    
    for i, input_file in enumerate(input_files, 1):
        print(f"\n[{i}/{len(input_files)}] Processing {input_file.name}")
        
        try:
            output_file = output_path / f"{input_file.stem}_with_params{input_file.suffix}"
            
            result = add_parameters_to_dataset(
                input_file,
                output_file=output_file,
                **kwargs
            )
            
            output_files.append(result)
            
        except Exception as e:
            print(f"ERROR processing {input_file.name}: {e}")
            continue
    
    print("\n" + "="*70)
    print(f"Batch processing complete: {len(output_files)}/{len(input_files)} successful")
    print("="*70)
    
    return output_files


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("ADD TC PARAMETERS TO TROPICAL_CYCLONE_RISK OUTPUT")
    print("="*70)
    
    print("\n1. PROCESS A SINGLE FILE")
    print("-"*70)
    print("from add_tc_params import add_parameters_to_dataset")
    print("")
    print("input_file = 'path/to/tracks_SI_ACCESS-CM2_historical_r1i1p1f1_197001_198912_e9.nc'")
    print("output_file = add_parameters_to_dataset(")
    print("    input_file,")
    print("    use_ensemble=True,")
    print("    env_pressure_method='standard',")
    print("    verbose=True")
    print(")")
    print("")
    
    print("\n2. BATCH PROCESS A DIRECTORY")
    print("-"*70)
    print("from add_tc_params import batch_process_directory")
    print("")
    print("output_files = batch_process_directory(")
    print("    input_dir='/path/to/tracks/',")
    print("    output_dir='/path/to/output/',")
    print("    pattern='tracks_*.nc',")
    print("    use_ensemble=True,")
    print("    overwrite=False,")
    print("    verbose=True")
    print(")")
    print("")
    
    print("\n3. LOAD AND USE THE ENHANCED DATA")
    print("-"*70)
    print("import xarray as xr")
    print("")
    print("ds = xr.open_dataset('tracks_..._with_params.nc')")
    print("")
    print("# Access new variables")
    print("central_pressure = ds.central_pressure")
    print("environmental_pressure = ds.environmental_pressure")
    print("pressure_deficit = ds.pressure_deficit")
    print("rmw = ds.rmw")
    print("roci = ds.roci")
    print("")
    print("# With uncertainties")
    print("rmw_mean = ds.rmw")
    print("rmw_uncertainty = ds.rmw_std")
    print("")
    
    print("\n4. ENVIRONMENTAL PRESSURE METHODS")
    print("-"*70)
    print("  - 'standard': Simple latitude-based (default)")
    print("  - 'gradient': Includes subtropical high effect")
    print("  - 'size_aware': Adjusts for storm size (requires ROCI)")
    print("")
    
    print("="*70)