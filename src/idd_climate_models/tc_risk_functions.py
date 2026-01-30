import os
import shutil
import sys
from pathlib import Path
import importlib.util
import importlib
import pandas as pd

import idd_climate_models.constants as rfc

TC_RISK_REPO_ROOT_DIR = rfc.TC_RISK_REPO_ROOT_DIR
repo_name = rfc.repo_name
package_name = rfc.package_name

REFERENCE_CONFIG_PATH = rfc.REPO_ROOT / repo_name / "src" / package_name / "01_run_tc_risk" / "reference_namelist.py"

# Cache for time bins dataframe to avoid re-reading on every call
_TIME_BINS_WIDE_DF_CACHE = None


def _load_time_bins_wide_df():
    """Load and cache the time bins wide dataframe."""
    global _TIME_BINS_WIDE_DF_CACHE
    if _TIME_BINS_WIDE_DF_CACHE is None:
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

def execute_tc_risk(args, script_name='compute'):
    custom_namelist_path = create_custom_namelist_path(args)
    
    # Just set the environment variable!
    os.environ['TC_RISK_NAMELIST_PATH'] = str(custom_namelist_path)
    
    original_cwd = os.getcwd()
    tc_risk_repo_path_str = str(TC_RISK_REPO_ROOT_DIR)
    original_sys_path = sys.path.copy()
    
    try:
        os.chdir(tc_risk_repo_path_str) 
        
        if tc_risk_repo_path_str not in sys.path:
            sys.path.insert(0, tc_risk_repo_path_str)
        
        from scripts import generate_land_masks
        from util import compute

        if script_name == 'compute':
            # generate_land_masks.generate_land_masks() Seems to be needed once for the repo, not every run
            compute.compute_downscaling_inputs()
        elif script_name == 'run_downscaling':
            import gc
            from dask.distributed import LocalCluster, Client
            
            # Create ONE cluster for all draws
            cl_args = {
                'n_workers': rfc.tc_risk_n_procs,
                'processes': True,
                'threads_per_worker': 1
            }
            
            with LocalCluster(**cl_args) as cluster, Client(cluster) as client:
                # Initialize cache on ALL worker processes (not in main process!)
                def init_worker_cache():
                    """Initialize geo cache on each worker when it starts"""
                    from intensity import geo
                    geo._initialize_cache()
                
                # Run this function on every worker process
                client.run(init_worker_cache)
                print("✅ Initialized cache on all workers")
                
                for draw in range(0, args.num_draws):
                    print(f"Starting draw {draw + 1}/{args.num_draws}")
                    
                    # Run tc_risk (pass the client so it reuses the cluster)
                    compute.run_downscaling(args.basin, client=client)
                    
                    # Force garbage collection between draws
                    gc.collect()
                    
                    print(f"✅ Completed draw {draw + 1}/{args.num_draws}")
        
        print("✅ Run complete.")
    finally:
        os.chdir(original_cwd)
        sys.path = original_sys_path
        del os.environ['TC_RISK_NAMELIST_PATH']