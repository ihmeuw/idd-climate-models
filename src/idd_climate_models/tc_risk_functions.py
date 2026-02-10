import os
import shutil
import sys
from pathlib import Path
import importlib.util
import importlib
import pandas as pd
import xarray as xr
import numpy as np
import argparse
import json

import idd_climate_models.constants as rfc
from idd_climate_models.add_tc_params import add_parameters_to_dataset
from idd_climate_models.climate_file_functions import get_track_path
from idd_climate_models.zarr_functions import verify_zarr_integrity

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
        delete_nc_post_process: If True, delete .nc files after converting to zarr (default: False)
    
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
                
                # Define callback to process .nc to .zarr while ds is still in memory
                def post_process_draw(ds, draw_num, fn_trk_out, tc_risk_time):
                    """Process Dataset to zarr immediately, avoiding re-read from disk"""
                    import time
                    post_process_start = time.time()
                    
                    print(f"\n  📊 Draw {draw_num} timing:")
                    print(f"      TC-risk computation: {tc_risk_time:.1f}s")
                    
                    # Step 1: Zarr conversion
                    zarr_start = time.time()
                    print(f"      [1/2] Converting to zarr format (in-memory)...")
                    zarr_created = process_tc_risk_dataset_to_zarr(args, ds, draw_num)
                    zarr_time = time.time() - zarr_start
                    print(f"            ✅ Zarr conversion: {zarr_time:.1f}s")
                    
                    # Step 2: Validation
                    validation_time = 0
                    if zarr_created:
                        validation_start = time.time()
                        print(f"      [2/2] Validating zarr integrity...")
                        from idd_climate_models.zarr_functions import verify_zarr_integrity
                        # Create complete process_args with ALL required attributes for get_track_path
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
                            draw=draw_num
                        )
                        zarr_path = get_track_path(process_args, source=False, extension=".zarr")
                        try:
                            verify_zarr_integrity(str(fn_trk_out), str(zarr_path), None)
                            validation_time = time.time() - validation_start
                            print(f"            ✅ Validation: {validation_time:.1f}s")
                        except Exception as e:
                            validation_time = time.time() - validation_start
                            print(f"            ❌ Validation failed after {validation_time:.1f}s")
                            raise ValueError(f"Draw {draw_num} zarr validation failed: {e}")
                    
                    # Total times
                    post_process_time = time.time() - post_process_start
                    total_draw_time = tc_risk_time + post_process_time
                    
                    print(f"      Post-processing: {post_process_time:.1f}s (zarr: {zarr_time:.1f}s + validation: {validation_time:.1f}s)")
                    print(f"      ⏱️  DRAW {draw_num} TOTAL: {total_draw_time:.1f}s")
                    
                    # Store timing data
                    draw_timings.append({
                        'draw': draw_num,
                        'tc_risk': tc_risk_time,
                        'zarr_conversion': zarr_time,
                        'validation': validation_time,
                        'post_process_total': post_process_time,
                        'total': total_draw_time
                    })
                    
                    # Conditionally delete .nc file to save space
                    if delete_nc_post_process:
                        try:
                            os.remove(fn_trk_out)
                            print(f"         Deleted .nc file: {fn_trk_out}")
                        except Exception as e:
                            print(f"         ⚠️  Could not delete .nc file {fn_trk_out}: {e}")
                
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
                    zarr_times = [t['zarr_conversion'] for t in draw_timings]
                    validation_times = [t['validation'] for t in draw_timings]
                    total_times = [t['total'] for t in draw_timings]
                    
                    print(f"  TC-risk computation:  min={min(tc_risk_times):.1f}s, max={max(tc_risk_times):.1f}s, avg={sum(tc_risk_times)/len(tc_risk_times):.1f}s")
                    print(f"  Zarr conversion:      min={min(zarr_times):.1f}s, max={max(zarr_times):.1f}s, avg={sum(zarr_times)/len(zarr_times):.1f}s")
                    print(f"  Validation:           min={min(validation_times):.1f}s, max={max(validation_times):.1f}s, avg={sum(validation_times)/len(validation_times):.1f}s")
                    print(f"  Total per draw:       min={min(total_times):.1f}s, max={max(total_times):.1f}s, avg={sum(total_times)/len(total_times):.1f}s")
                    print(f"  Grand total:          {sum(total_times):.1f}s ({sum(total_times)/60:.1f} min)")
                    print(f"{'=' * 80}")
                
                # Set permissions on output files (both .nc and .zarr)
                if args:
                    output_dir = (rfc.TC_RISK_OUTPUT_PATH / args.data_source / args.model / 
                                 args.variant / args.scenario / args.time_period / args.basin)
                    
                    if output_dir.exists():
                        for nc_file in output_dir.glob("*.nc"):
                            try:
                                os.chmod(nc_file, 0o775)
                            except Exception as e:
                                print(f"Warning: Could not set permissions on {nc_file}: {e}")
                        for zarr_dir in output_dir.glob("*.zarr"):
                            try:
                                set_zarr_permissions_recursive(zarr_dir, 0o775)
                            except Exception as e:
                                print(f"Warning: Could not set permissions on {zarr_dir}: {e}")
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


# =============================================================================
# OLD FUNCTIONS BELOW - DEPRECATED BUT KEPT FOR BACKWARD COMPATIBILITY
# These create Python namelist files (causes race conditions with parallel jobs)
# Use create_tc_risk_config_dict() and execute_tc_risk_with_config() instead
# =============================================================================

def create_custom_namelist_path(args):
    """
    Constructs the custom namelist path based on input arguments, 
    using the full nested output directory structure.
    """
    target_output_path_root = rfc.TC_RISK_OUTPUT_PATH / args.data_source
    output_dir = target_output_path_root / args.model / args.variant / args.scenario / args.time_period
    exp_name_folder = f'{args.basin}' 
    custom_namelist_path = output_dir / exp_name_folder / 'namelist.py'
    
    return custom_namelist_path

def create_replacement_line(var_name, new_value, original_lines):
    """
    Finds the original line for a variable and returns the new, complete line, 
    preserving any comments or trailing spaces.
    """
    new_line_prefix = f"{var_name} = {new_value}"
    for line in original_lines:
        if line.strip().startswith(f"{var_name} ="):
            comment_index = line.find('#')
            if comment_index != -1:
                trailing_comment = line[comment_index:].strip()
                return f"{new_line_prefix}    # {trailing_comment}\n"
            else:
                return f"{new_line_prefix}\n"
    return f"{new_line_prefix}\n"

def modify_and_save_config(reference_namelist_path, custom_namelist_path, replacements):
    """
    Reads the reference config, applies replacements, and saves it to custom_namelist_path.
    Returns the path of the newly created file.
    """
    try:
        with open(reference_namelist_path, 'r') as f:
            original_content = f.read()
    except FileNotFoundError:
        print(f"❌ ERROR: Original configuration file not found at {reference_namelist_path}")
        return None

    lines = original_content.splitlines()
    new_lines = []

    for line in lines:
        is_replaced = False
        for var_name, new_value in replacements.items():
            if line.strip().startswith(f"{var_name} ="):
                new_line = create_replacement_line(var_name, new_value, lines).strip()
                new_lines.append(new_line)
                is_replaced = True
                break
        
        if not is_replaced:
            new_lines.append(line)

    modified_content = '\n'.join(new_lines) + '\n'

    os.makedirs(custom_namelist_path.parent, exist_ok=True)
    with open(custom_namelist_path, 'w') as f:
        f.write(modified_content)

    print(f"✅ Configuration successfully created at: {custom_namelist_path}")
    return custom_namelist_path

def create_custom_namelist(args, verbose=False):
    """
    Creates a custom namelist configuration file for TC-risk with specified parameters.
    Ensures output paths point to the correct OUTPUT location.
    """

    target_base_path = rfc.TC_RISK_INPUT_PATH / args.data_source
    target_output_path = rfc.TC_RISK_OUTPUT_PATH / args.data_source

    # Full path to the specific input data folder for base_directory
    base_dir = target_base_path / args.model / args.variant / args.scenario / args.time_period
    
    # Full path to the parent output folder for the specific time bin
    # This value becomes namelist.output_directory
    output_dir_parent = target_output_path / args.model / args.variant / args.scenario / args.time_period

    exp_name = f'{args.basin}' # This value becomes namelist.exp_name (e.g., 'GL')

    # Look up tracks_per_year from time_bins_wide.csv for this specific combination
    tracks_per_year = get_tracks_per_year(
        model=args.model,
        variant=args.variant,
        scenario=args.scenario,
        time_period=args.time_period,
        basin=args.basin
    )

    start_year, end_year = map(int, args.time_period.split('-'))
    dataset_type = 'GCM' if args.data_source.lower() == 'cmip6' else 'era5'

    replacements = {}
    
    replacements['src_directory'] = f"'{TC_RISK_REPO_ROOT_DIR}'"
    replacements['base_directory'] = f"'{base_dir}'"
    replacements['output_directory'] = f"'{output_dir_parent}'" 
    replacements['exp_name'] = f"'{exp_name}'"
    replacements['dataset_type'] = f"'{dataset_type}'"
    replacements['exp_prefix'] = f"'{args.model}_{args.scenario}_{args.variant}'"
    replacements['file_type'] = f"'{rfc.tc_risk_file_type}'"
    replacements['n_procs'] = str(rfc.tc_risk_n_procs)
    replacements['tracks_per_year'] = str(tracks_per_year)
    replacements['start_year'] = str(start_year)
    replacements['end_year'] = str(end_year)
    replacements['total_track_time_days'] = str(rfc.tc_risk_total_track_time_days)
    
    if verbose:
        print("🔧 Creating custom namelist with the following parameters:")
        for key, value in replacements.items():
            print(f"   - {key}: {value}")
    custom_namelist_path = create_custom_namelist_path(args)
    
    return modify_and_save_config(REFERENCE_CONFIG_PATH, custom_namelist_path, replacements)

# ============================================================================
# POST-PROCESSING FUNCTIONS (TC-risk output → Zarr for CLIMADA)
# ============================================================================

def set_input_file_permissions(file_path: Path, mode: int = 0o775):
    """Sets permissions on input .nc file before reading."""
    try:
        if file_path.is_file():
            os.chmod(str(file_path), mode)
        elif file_path.is_dir():
            os.chmod(str(file_path), mode)
            for root, dirs, files in os.walk(file_path):
                for d in dirs:
                    os.chmod(os.path.join(root, d), mode)
                for f in files:
                    os.chmod(os.path.join(root, f), mode)
    except Exception as e:
        print(f"Warning: Could not set permissions on {file_path}: {e}")


def set_zarr_permissions_recursive(zarr_path: Path, mode: int = 0o775):
    """Recursively sets permissions for all files and directories in the Zarr store."""
    try:
        os.chmod(str(zarr_path), mode)
        for dirpath, dirnames, filenames in os.walk(zarr_path):
            for dirname in dirnames:
                os.chmod(Path(dirpath) / dirname, mode)
            for filename in filenames:
                os.chmod(Path(dirpath) / filename, mode)
    except Exception as e:
        print(f"Warning: Could not set zarr permissions on {zarr_path}: {e}")


def create_storm_list_file(ds_multi, args):
    """
    Processes and saves each individual track as a Zarr Group within a single 
    Zarr store, preserving the list-of-datasets structure.
    """
    n_trk = ds_multi.sizes["n_trk"]
    raw_time = ds_multi["time"].values
    dt_hours = np.diff(raw_time).mean() / 3600.0 if len(raw_time) > 1 else 1.0

    storms = []
    for i in range(n_trk):
        if np.isnan(ds_multi["tc_years"][i].item()) or np.isnan(ds_multi["tc_month"][i].item()):
             continue
        
        start_year = int(ds_multi["tc_years"][i].item())
        start_month = int(ds_multi["tc_month"][i].item())
        basin = ds_multi["tc_basins"][i].item()
        
        lon = ds_multi["lon_trks"][i].values
        lat = ds_multi["lat_trks"][i].values
        cp = ds_multi["central_pressure"][i].values
        env = ds_multi["environmental_pressure"][i].values
        vmax = ds_multi["vmax_trks"][i].values
        
        start_date = np.datetime64(f"{start_year:04d}-{start_month:02d}-01T00:00", "s")
        time_seconds = raw_time.astype("timedelta64[s]")
        time_dt = start_date + time_seconds
        time_dt = time_dt.astype("datetime64[h]")
        
        lon = ((lon + 180) % 360) - 180
        lat = np.clip(lat, -90, 90)

        valid_idx = np.isfinite(lon) & np.isfinite(lat)
        valid_lon, valid_lat, valid_vmax, valid_cp, valid_env, valid_time_dt = (
            arr[valid_idx] for arr in [lon, lat, vmax, cp, env, time_dt]
        )
        n_time = len(valid_lon)
        
        if n_time == 0:
            continue

        vmax_max = valid_vmax.max().item() if valid_vmax.size > 0 else 0 
        category = 0 if vmax_max < 33 else 1 if vmax_max < 43 else 2 if vmax_max < 50 else 3 if vmax_max < 58 else 4 if vmax_max < 70 else 5

        track_ds = xr.Dataset(
            coords={"time": valid_time_dt, "storm_id": i}, 
            data_vars={
                "lon": (("time",), valid_lon),
                "lat": (("time",), valid_lat),
                "max_sustained_wind": (("time",), valid_vmax),
                "central_pressure": (("time",), valid_cp),
                "environmental_pressure": (("time",), valid_env),
                "basin": (("time",), np.repeat(basin, n_time)),
                "time_step": (("time",), np.full(n_time, dt_hours)),
            },
            attrs={
                "name": f"{basin}_{args.draw}_{start_year}_{i:04d}",
                "sid": int(i),
                "id_no": int(i),
                "category": category,
                "data_provider": "custom",
                "max_sustained_wind_unit": "kn",
            }
        )
        storms.append(track_ds)
    
    if len(storms) == 0:
        print("Warning: No valid storms found in dataset")
        return
    
    output_zarr_path = get_track_path(args, source=False, extension=".zarr")
    
    # Remove existing zarr file to avoid conflicts from partial previous runs
    if Path(output_zarr_path).exists():
        import shutil
        shutil.rmtree(output_zarr_path)
    
    print(f"Saving {len(storms)} storms to Zarr at: {output_zarr_path}")

    first_ds = storms[0]
    first_ds.to_zarr(
        output_zarr_path,
        group=f'storm_{first_ds.storm_id.item():04d}', 
        mode='w',
        consolidated=True
    )

    for ds in storms[1:]:
        group_name = f'storm_{ds.storm_id.item():04d}'
        ds.to_zarr(output_zarr_path, group=group_name, mode='a')
    
    set_zarr_permissions_recursive(output_zarr_path, 0o775)
    print(f"Successfully saved {len(storms)} storms to Zarr with 775 permissions.")


def process_tc_risk_dataset_to_zarr(args, ds, draw):
    """
    Post-process a TC-risk Dataset to zarr format (in-memory, no disk re-read).
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
    
    # Add parameters and create zarr
    ds_processed = add_parameters_to_dataset(ds, process_args)
    create_storm_list_file(ds_processed, process_args)
    
    return True


def process_tc_risk_output_to_zarr(args, draw):
    """
    Post-process a single TC-risk output file to zarr format (reads from disk).
    Use this only when processing existing .nc files after the fact.
    For new runs, use process_tc_risk_dataset_to_zarr() with callback instead.
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
        verify=True
    )
    
    input_track_path = get_track_path(process_args, source=True)
    
    if not input_track_path.exists():
        print(f"  ⚠️  TC-risk output not found: {input_track_path}")
        return False
    
    print(f"  Post-processing draw {draw} to zarr...")
    
    # Set permissions on input file
    set_input_file_permissions(input_track_path, 0o775)
    
    # Read and process
    ds = xr.open_dataset(input_track_path)
    ds = add_parameters_to_dataset(ds, process_args)
    create_storm_list_file(ds, process_args)
    ds.close()
    
    # Verify output
    output_zarr_path = get_track_path(process_args, source=False, extension=".zarr")
    verify_zarr_integrity(input_track_path, output_zarr_path, process_args)
    
    print(f"  ✅ Post-processing complete for draw {draw}")
    return True


# FIXED VERSION for tc_risk_functions.py - using vendored TC-risk code

def execute_tc_risk(args, script_name='compute'):
    """
    Execute TC-risk using vendored code to avoid race conditions.
    
    The vendored code has been modified to:
    1. Use relative imports within tc_risk_vendored package
    2. Accept namelist parameters as function arguments instead of importing
    3. No longer requires sys.path manipulation or os.chdir()
    """
    # Create custom namelist FIRST
    custom_namelist_path = create_custom_namelist_path(args)
    created_namelist = create_custom_namelist(args, verbose=True)
    
    if created_namelist is None:
        print("❌ ERROR: Failed to create custom namelist")
        return False
    
    # Get batch parameters (default to full range for backward compatibility)
    draw_start = getattr(args, 'draw_start_batch', 0)
    draw_end = getattr(args, 'draw_end_batch', rfc.NUM_DRAWS - 1)
    
    try:
        # Load the namelist module for this specific job
        # The vendored TC-risk code will read from this
        spec = importlib.util.spec_from_file_location("namelist_loader", custom_namelist_path)
        namelist_loader = importlib.util.module_from_spec(spec)
        
        # Register module and execute it
        sys.modules['namelist_loader'] = namelist_loader
        spec.loader.exec_module(namelist_loader)
        
        # Set alias for compatibility
        sys.modules['namelist'] = namelist_loader
        
        print(f"✅ Loaded namelist for {args.model}/{args.scenario}/{args.time_period}")
        
        # Import vendored TC-risk modules
        # These use relative imports internally, no sys.path hacks needed
        from idd_climate_models.tc_risk_vendored.scripts import generate_land_masks
        from idd_climate_models.tc_risk_vendored.util import compute

        if script_name == 'compute':
            compute.compute_downscaling_inputs()
            
        elif script_name == 'run_downscaling':
            import gc
            from dask.distributed import LocalCluster, Client
            
            cl_args = {
                'n_workers': rfc.tc_risk_n_procs,
                'processes': True,
                'threads_per_worker': 1,
                'memory_limit': 'auto', 
            }
            
            with LocalCluster(**cl_args) as cluster, Client(cluster) as client:
                def init_worker_cache():
                    from idd_climate_models.tc_risk_vendored.intensity import geo
                    geo._initialize_cache()
                
                client.run(init_worker_cache)
                print("✅ Initialized cache on all workers")
                
                output_dir = (rfc.TC_RISK_OUTPUT_PATH / args.data_source / args.model / 
                             args.variant / args.scenario / args.time_period / args.basin)
                
                num_draws_in_batch = draw_end - draw_start + 1
                
                print(f"\nRunning {num_draws_in_batch} draws ({draw_start}-{draw_end})")
                
                compute.run_downscaling(
                    args.basin, 
                    client=client,
                    draw_start=draw_start,
                    draw_end=draw_end
                )
                
                # Set permissions on output files
                if output_dir.exists():
                    for nc_file in output_dir.glob("*.nc"):
                        try:
                            os.chmod(nc_file, 0o775)
                        except Exception as e:
                            print(f"Warning: Could not set permissions on {nc_file}: {e}")
                
                # Post-process each draw in the batch
                for draw_num in range(draw_start, draw_end + 1):
                    try:
                        process_tc_risk_output_to_zarr(args, draw_num)
                    except Exception as e:
                        print(f"⚠️  Warning: Post-processing failed for draw {draw_num}: {e}")
                
                gc.collect()
        
        print("✅ Run complete.")
        
    finally:
        # Clean up the module from sys.modules
        if 'namelist_loader' in sys.modules:
            del sys.modules['namelist_loader']
        if 'namelist' in sys.modules:
            del sys.modules['namelist']