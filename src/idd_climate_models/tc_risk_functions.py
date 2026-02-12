import os
import sys
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
import argparse
import json

import idd_climate_models.constants as rfc
from idd_climate_models.add_tc_params import add_parameters_to_dataset
from idd_climate_models.climate_file_functions import get_track_path

# Import the vendored config utilities
from idd_climate_models.tc_risk_vendored.config_utils import create_tc_risk_config

TC_RISK_REPO_ROOT_DIR = rfc.TC_RISK_REPO_ROOT_DIR
repo_name = rfc.repo_name
package_name = rfc.package_name

# Cache for time bins dataframe to avoid re-reading on every call
_TIME_BINS_WIDE_DF_CACHE = None


def _load_time_bins_wide_df():
    """Load and cache the time bins wide dataframe.
    
    Tries to load the chunked wide file (max_bin_5) first if it exists,
    otherwise falls back to the original wide file.
    """
    global _TIME_BINS_WIDE_DF_CACHE
    if _TIME_BINS_WIDE_DF_CACHE is None:
        # Try loading the chunked wide file first (created by get_time_bins_path with max_duration=5)
        chunked_wide_path = rfc.TIME_BINS_WIDE_DF_PATH.parent / 'bayespoisson_time_bins_wide_max_bin_5.csv'
        
        if chunked_wide_path.exists():
            print(f"Loading chunked time bins from {chunked_wide_path}")
            _TIME_BINS_WIDE_DF_CACHE = pd.read_csv(chunked_wide_path)
        else:
            print(f"Chunked file not found, using original: {rfc.TIME_BINS_WIDE_DF_PATH}")
            _TIME_BINS_WIDE_DF_CACHE = pd.read_csv(rfc.TIME_BINS_WIDE_DF_PATH)
    return _TIME_BINS_WIDE_DF_CACHE


def get_tracks_per_year(model, variant, scenario, time_period, basin):
    """
    Look up tracks_per_year (int_storms) from the time_bins_wide CSV.

    Args:
        model: Climate model name (e.g., 'CMCC-ESM2')
        variant: Model variant (e.g., 'r1i1p1f1')
        scenario: Scenario name (e.g., 'historical', 'ssp126')
        time_period: Time period string (e.g., '1970-1986')
        basin: Basin code (e.g., 'EP', 'NA', 'GL')

    Returns:
        int: The integer storm count for this combination

    Raises:
        ValueError: If no matching row is found in the time bins file
    """
    df = _load_time_bins_wide_df()

    # Parse time_period to get start_year and end_year
    start_year, end_year = map(int, time_period.split('-'))

    # Filter to matching row
    mask = (
        (df['model'] == model) &
        (df['variant'] == variant) &
        (df['scenario'] == scenario) &
        (df['start_year'] == start_year) &
        (df['end_year'] == end_year)
    )

    matching_rows = df[mask]

    if len(matching_rows) == 0:
        raise ValueError(
            f"No time bins found for {model}/{variant}/{scenario}/{time_period}. "
            f"Check that this combination exists in {rfc.TIME_BINS_WIDE_DF_PATH}"
        )

    if len(matching_rows) > 1:
        print(f"Warning: Multiple rows found for {model}/{variant}/{scenario}/{time_period}, using first")

    row = matching_rows.iloc[0]

    # Get the int_storms column for this basin
    basin_col = f"{basin}_int"

    if basin_col not in row:
        raise ValueError(
            f"Basin '{basin}' not found in time bins file. "
            f"Expected column '{basin_col}'"
        )

    tracks_per_year = int(row[basin_col])

    return tracks_per_year


def create_tc_risk_config_dict(args):
    """
    Creates a complete TC-risk configuration dictionary.
    
    NO namelist files created! Returns dictionary ready to pass to vendored functions.
    
    Args:
        args: Namespace with model, variant, scenario, time_period, basin, data_source
    
    Returns:
        dict: Complete configuration dictionary
    """
    # Set up paths
    target_base_path = rfc.TC_RISK_INPUT_PATH / args.data_source
    target_output_path = rfc.TC_RISK_OUTPUT_PATH / args.data_source
    
    # Full path to input data
    base_dir = str(target_base_path / args.model / args.variant / args.scenario / args.time_period)
    
    # Full path to output folder
    output_dir_parent = str(target_output_path / args.model / args.variant / args.scenario / args.time_period)
    
    # Look up tracks_per_year from time_bins_wide.csv
    tracks_per_year = get_tracks_per_year(
        model=args.model,
        variant=args.variant,
        scenario=args.scenario,
        time_period=args.time_period,
        basin=args.basin
    )
    
    # Create config using helper
    dataset_type = 'GCM' if args.data_source.lower() == 'cmip6' else 'ERA5'
    
    config_dict = create_tc_risk_config(
        model=args.model,
        variant=args.variant,
        scenario=args.scenario,
        time_period=args.time_period,
        basin=args.basin,
        base_directory=base_dir,
        output_directory=output_dir_parent,
        n_procs=rfc.tc_risk_n_procs,
        dataset_type=dataset_type
    )
    
    # Add additional parameters specific to this workflow
    config_dict.update({
        'file_type': rfc.tc_risk_file_type,
        'tracks_per_year': tracks_per_year,
        'total_track_time_days': rfc.tc_risk_total_track_time_days,
        'src_directory': str(TC_RISK_REPO_ROOT_DIR),
    })
    
    print(f"✅ Created config dict for {args.model}/{args.variant}/{args.scenario}/{args.time_period}/{args.basin}")
    return config_dict


def execute_tc_risk_with_config(config_dict, script_name='compute', args=None, delete_nc_post_process=False, total_memory=None):
    """
    Execute TC-risk using configuration dictionary (NO NAMELIST FILES!).
    
    Args:
        config_dict: Complete configuration dictionary
        script_name: 'compute' or 'run_downscaling'
        args: Original args namespace (for draw batch info)
        delete_nc_post_process: If True, delete .nc files after creating climada input file (default: False)
    
    Returns:
        bool: True if successful
    """
    print(f"🚀 Executing TC-risk with dict-based config...")
    
    # Save config JSON to output directory (same location as old namelist.py)
    if args:
        config_save_dir = (rfc.TC_RISK_OUTPUT_PATH / args.data_source / args.model / 
                          args.variant / args.scenario / args.time_period / args.basin)
        config_save_dir.mkdir(parents=True, exist_ok=True)
        config_file_path = config_save_dir / 'config.json'
        
        with open(config_file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"✅ Saved config to: {config_file_path}")
    
    # Get draws to run - either from explicit list or from range
    if hasattr(args, 'draws_to_run') and args.draws_to_run:
        # Explicit list of draws provided (skips already-completed draws)
        draws_to_run = args.draws_to_run
    else:
        # Fall back to range-based approach
        draw_start = getattr(args, 'draw_start_batch', 0) if args else 0
        draw_end = getattr(args, 'draw_end_batch', rfc.NUM_DRAWS - 1) if args else rfc.NUM_DRAWS - 1
        draws_to_run = list(range(draw_start, draw_end + 1))
    
    try:
        # Import vendored TC-risk modules
        from idd_climate_models.tc_risk_vendored.scripts import generate_land_masks
        from idd_climate_models.tc_risk_vendored.util import compute
        
        if script_name == 'compute':
            # Pass config_dict directly - NO imports of namelist_loader!
            compute.compute_downscaling_inputs(config_dict)
            
        elif script_name == 'run_downscaling':
            import gc
            import os
            from pathlib import Path
            from dask.distributed import LocalCluster, Client
            
            # Create worker log directory
            worker_log_dir = Path(config_dict['output_directory']) / 'worker_logs'
            worker_log_dir.mkdir(exist_ok=True, parents=True)
            
            # Calculate memory per worker from total allocation
            n_workers = config_dict['n_procs']
            if total_memory is not None:
                # Parse memory string (e.g., "20G" -> 20GB)
                memory_str = total_memory.upper().replace('G', '')
                total_memory_gb = float(memory_str)
                # Allocate 90% of total memory to workers (10% for scheduler overhead)
                memory_per_worker_gb = (total_memory_gb * 0.9) / n_workers
                memory_per_worker = f"{memory_per_worker_gb:.2f}GB"
                print(f"📊 Calculated memory: {total_memory_gb}G total → {memory_per_worker} per worker ({n_workers} workers)")
            else:
                # Fallback: use 4GB per worker
                memory_per_worker = "4GB"
                print(f"⚠️  No total_memory provided, using fallback: {memory_per_worker} per worker")
            
            cl_args = {
                'n_workers': n_workers,
                'processes': True,
                'threads_per_worker': 1,
                'memory_limit': memory_per_worker,
                'silence_logs': 'ERROR',  # Reduce logging noise at scheduler level
                'local_directory': str(worker_log_dir),  # Worker scratch space
            }
            
            print(f"🔍 Worker logs will be written to: {worker_log_dir}")
            
            cluster = None
            client = None
            try:
                cluster = LocalCluster(**cl_args)
                client = Client(cluster)
                
                # Print worker info for debugging
                print(f"📊 Dask cluster started with {len(client.scheduler_info()['workers'])} workers")
                for worker_addr, worker_info in client.scheduler_info()['workers'].items():
                    mem_limit = worker_info.get('memory_limit', 'unknown')
                    mem_limit_gb = mem_limit / (1024**3) if isinstance(mem_limit, (int, float)) else mem_limit
                    print(f"  Worker {worker_addr}: memory_limit={mem_limit} ({mem_limit_gb:.2f} GB)" if isinstance(mem_limit_gb, float) else f"  Worker {worker_addr}: memory_limit={mem_limit}")
                
                # Note: config_dict will be captured in the closure and sent to workers
                def init_worker_cache():
                    from idd_climate_models.tc_risk_vendored.intensity import geo
                    geo._initialize_cache(config_dict)
                
                client.run(init_worker_cache)
                print("✅ Initialized cache on all workers")
                
                # Pass config_dict to downscaling function
                print(f"\nRunning {len(draws_to_run)} draws: {draws_to_run}")
                
                # Track draw-level timing
                draw_timings = []
                
                # Define callback to process TC-risk output to climada input format
                def post_process_draw(ds, draw_num, fn_trk_out, tc_risk_time):
                    """Create climada input file from TC-risk output (in-memory)"""
                    import time
                    post_process_start = time.time()
                    
                    print(f"\n  📊 Draw {draw_num} timing:")
                    print(f"      TC-risk computation: {tc_risk_time:.1f}s")
                    
                    # Convert to climada input format (netCDF)
                    nc_conversion_start = time.time()
                    print(f"      Converting to climada format (in-memory)...")
                    try:
                        process_tc_risk_dataset_for_climada(args, ds, draw_num)
                        nc_conversion_time = time.time() - nc_conversion_start
                        print(f"      ✅ Conversion: {nc_conversion_time:.1f}s")
                        
                        # Create completion marker for this draw
                        climada_input_path = (rfc.CLIMADA_INPUT_PATH / args.data_source / args.model / 
                                            args.variant / args.scenario / args.time_period / args.basin)
                        marker_path = climada_input_path / f".nc_draw_{draw_num:04d}.complete"
                        marker_path.touch()
                        try:
                            marker_path.chmod(0o775)
                        except:
                            pass  # Permissions failure is non-critical
                        
                    except Exception as e:
                        nc_conversion_time = time.time() - nc_conversion_start
                        print(f"      ❌ Conversion failed after {nc_conversion_time:.1f}s")
                        raise ValueError(f"Draw {draw_num} conversion failed: {e}")
                    
                    # Total times
                    post_process_time = time.time() - post_process_start
                    total_draw_time = tc_risk_time + post_process_time
                    
                    print(f"      ⏱️  DRAW {draw_num} TOTAL: {total_draw_time:.1f}s")
                    
                    # Store timing data
                    draw_timings.append({
                        'draw': draw_num,
                        'tc_risk': tc_risk_time,
                        'nc_conversion': nc_conversion_time,
                        'post_process_total': post_process_time,
                        'total': total_draw_time
                    })
                    
                    # Conditionally delete TC-risk .nc file to save space
                    if delete_nc_post_process:
                        try:
                            os.remove(fn_trk_out)
                            print(f"      Deleted TC-risk .nc file: {fn_trk_out}")
                        except Exception as e:
                            print(f"      ⚠️  Could not delete TC-risk .nc file {fn_trk_out}: {e}")
                
                compute.run_downscaling(
                    basin_id=config_dict.get('basin', 'GL'),
                    config_dict=config_dict,
                    client=client,
                    draws_to_run=draws_to_run,
                    post_process_callback=post_process_draw
                )
                
                # Print timing summary after all draws complete
                if draw_timings:
                    print(f"\n{'=' * 80}")
                    print("⏱️  PER-DRAW TIMING SUMMARY:")
                    print(f"{'=' * 80}")
                    tc_risk_times = [t['tc_risk'] for t in draw_timings]
                    nc_conversion_times = [t['nc_conversion'] for t in draw_timings]
                    total_times = [t['total'] for t in draw_timings]
                    
                    print(f"  TC-risk computation:  min={min(tc_risk_times):.1f}s, max={max(tc_risk_times):.1f}s, avg={sum(tc_risk_times)/len(tc_risk_times):.1f}s")
                    print(f"  NC conversion:        min={min(nc_conversion_times):.1f}s, max={max(nc_conversion_times):.1f}s, avg={sum(nc_conversion_times)/len(nc_conversion_times):.1f}s")
                    print(f"  Total per draw:       min={min(total_times):.1f}s, max={max(total_times):.1f}s, avg={sum(total_times)/len(total_times):.1f}s")
                    print(f"  Grand total:          {sum(total_times):.1f}s ({sum(total_times)/60:.1f} min)")
                    print(f"{'=' * 80}")
                
                # Set permissions on output files
                if args:
                    # TC-risk output directory
                    tc_risk_output_dir = (rfc.TC_RISK_OUTPUT_PATH / args.data_source / args.model / 
                                         args.variant / args.scenario / args.time_period / args.basin)
                    if tc_risk_output_dir.exists():
                        for nc_file in tc_risk_output_dir.glob("*.nc"):
                            try:
                                os.chmod(nc_file, 0o775)
                            except Exception as e:
                                print(f"Warning: Could not set permissions on {nc_file}: {e}")
                    
                    # Climada input directory
                    climada_input_dir = (rfc.CLIMADA_INPUT_PATH / args.data_source / args.model / 
                                        args.variant / args.scenario / args.time_period / args.basin)
                    if climada_input_dir.exists():
                        for nc_file in climada_input_dir.glob("*.nc"):
                            try:
                                os.chmod(nc_file, 0o775)
                            except Exception as e:
                                print(f"Warning: Could not set permissions on {nc_file}: {e}")
            finally:
                # Graceful cleanup with proper timeout handling
                if client is not None:
                    try:
                        client.close(timeout=10)
                    except Exception as e:
                        print(f"⚠️ Warning during client close: {e}")
                
                if cluster is not None:
                    try:
                        cluster.close(timeout=10)
                    except Exception as e:
                        print(f"⚠️ Warning during cluster close: {e}")
                
                # Force garbage collection to help cleanup
                gc.collect()
        
        else:
            raise ValueError(f"Unknown script_name: {script_name}")
        
        print(f"✅ Successfully executed TC-risk ({script_name})")
        return True
        
    except Exception as e:
        print(f"❌ ERROR executing TC-risk: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# POST-PROCESSING FUNCTIONS (TC-risk output → CLIMADA input format)
# ============================================================================

def create_storm_list_file(ds_multi, args):
    """
    Processes tc-risk output and creates Climada input files in .nc format.
    Uses vectorized operations on the original (n_trk, n_time) structure for efficiency.
    """
    n_trk = ds_multi.sizes["n_trk"]
    n_time = ds_multi.sizes["time"]
    raw_time = ds_multi["time"].values
    dt_hours = np.diff(raw_time).mean() / 3600.0 if len(raw_time) > 1 else 1.0
    
    # Filter out storms with invalid metadata
    valid_storm_mask = ~(np.isnan(ds_multi["tc_years"].values) | np.isnan(ds_multi["tc_month"].values))
    
    if not valid_storm_mask.any():
        print("Warning: No valid storms found in dataset")
        return
    
    # Apply vectorized transformations to all storms at once
    lon = ds_multi["lon_trks"].values.copy()
    lat = ds_multi["lat_trks"].values.copy()
    
    # Normalize longitude to [-180, 180]
    lon = ((lon + 180) % 360) - 180
    lat = np.clip(lat, -90, 90)
    
    # Mark invalid positions as NaN (they'll be filtered during reading)
    lon = np.where(np.isfinite(lon) & np.isfinite(lat), lon, np.nan)
    lat = np.where(np.isfinite(lon) & np.isfinite(lat), lat, np.nan)
    
    # Calculate storm categories (vectorized)
    vmax_max = np.nanmax(ds_multi["vmax_trks"].values, axis=1)
    categories = np.select(
        [vmax_max < 33, vmax_max < 43, vmax_max < 50, vmax_max < 58, vmax_max < 70],
        [0, 1, 2, 3, 4],
        default=5
    )
    
    # Create output dataset with rectangular structure
    output_ds = xr.Dataset(
        coords={
            "n_trk": np.arange(n_trk),
            "time": raw_time,
        },
        data_vars={
            "lon": (("n_trk", "time"), lon),
            "lat": (("n_trk", "time"), lat),
            "max_sustained_wind": (("n_trk", "time"), ds_multi["vmax_trks"].values),
            "central_pressure": (("n_trk", "time"), ds_multi["central_pressure"].values),
            "environmental_pressure": (("n_trk", "time"), ds_multi["environmental_pressure"].values),
            "time_step": (("n_trk", "time"), np.full((n_trk, n_time), dt_hours)),
            # Storm metadata
            "tc_years": (("n_trk",), ds_multi["tc_years"].values),
            "tc_month": (("n_trk",), ds_multi["tc_month"].values),
            "tc_basins": (("n_trk",), ds_multi["tc_basins"].values),
            "category": (("n_trk",), categories),
            "valid_storm": (("n_trk",), valid_storm_mask),
        },
        attrs={
            "draw": args.draw,
            "data_provider": "custom",
            "max_sustained_wind_unit": "kn",
        }
    )
    
    # Filter to only valid storms
    output_ds = output_ds.sel(n_trk=valid_storm_mask)
    
    num_valid_storms = output_ds.sizes["n_trk"]
    print(f"Saving {num_valid_storms} valid storms (of {n_trk} total)")
    
    output_path = get_track_path(args, source=False, extension=".nc")
    
    # Set umask to create files with 775 permissions from the start
    old_umask = os.umask(0o002)
    
    try:
        # Save as netCDF file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_ds.to_netcdf(str(output_path), engine='netcdf4')
    finally:
        # Always restore original umask
        os.umask(old_umask)
    
    # Set file permissions
    try:
        os.chmod(str(output_path), 0o775)
    except Exception as e:
        print(f"Warning: Could not set file permissions: {e}")
    
    # Validate output file
    try:
        with xr.open_dataset(str(output_path)) as test_ds:
            if 'lon' not in test_ds or 'lat' not in test_ds:
                raise ValueError("Required variables missing from output")
        print(f"Successfully saved and validated {num_valid_storms} storms to netCDF.")
    except Exception as e:
        print(f"❌ ERROR: Output file validation failed: {e}")
        # Delete corrupted file
        try:
            output_path.unlink()
        except:
            pass
        raise


def process_tc_risk_dataset_for_climada(args, ds, draw):
    """
    Post-process a TC-risk Dataset to climada input format (in-memory, no disk re-read).
    Called immediately after each draw completes with the Dataset still in memory.
    
    Args:
        args: Arguments namespace
        ds: xarray Dataset (already in memory from TC-risk)
        draw: Draw number
    """
    # Create args namespace compatible with get_track_path
    process_args = argparse.Namespace(
        input_data_type='tc_risk',
        input_io_data_type='output',
        output_data_type='climada',
        output_io_data_type='input',
        data_source=args.data_source,
        model=args.model,
        variant=args.variant,
        scenario=args.scenario,
        time_period=args.time_period,
        basin=args.basin,
        draw=draw,
        use_ensemble=True,
        env_pressure_method='standard',
        verbose=False,
        verify=False  # Skip verification since we're not reading from disk
    )
    
    # Add parameters and create climada input file
    ds_processed = add_parameters_to_dataset(ds, process_args)
    create_storm_list_file(ds_processed, process_args)
    
    return True


def process_tc_risk_output_to_netcdf(args, draw):
    """
    Post-process a single TC-risk output file to climada input format (reads from disk).
    Use this only when processing existing .nc files after the fact.
    For new runs, use process_tc_risk_dataset_for_climada() with callback instead.
    """
    # Create args namespace compatible with get_track_path
    process_args = argparse.Namespace(
        input_data_type='tc_risk',
        input_io_data_type='output',
        output_data_type='climada',
        output_io_data_type='input',
        data_source=args.data_source,
        model=args.model,
        variant=args.variant,
        scenario=args.scenario,
        time_period=args.time_period,
        basin=args.basin,
        draw=draw,
        use_ensemble=True,
        env_pressure_method='standard',
        verbose=False,
        verify=False
    )
    
    input_track_path = get_track_path(process_args, source=True)
    
    if not input_track_path.exists():
        print(f"  ⚠️  TC-risk output not found: {input_track_path}")
        return False
    
    print(f"  Post-processing draw {draw} to climada format...")
    
    # Set permissions on input file
    try:
        os.chmod(input_track_path, 0o775)
    except Exception as e:
        print(f"Warning: Could not set permissions on {input_track_path}: {e}")
    
    # Read and process
    ds = xr.open_dataset(input_track_path)
    ds = add_parameters_to_dataset(ds, process_args)
    create_storm_list_file(ds, process_args)
    ds.close()
    
    print(f"  ✅ Post-processing complete for draw {draw}")
    return True