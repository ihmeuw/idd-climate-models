import os
import shutil
import sys
from pathlib import Path
import importlib.util
import importlib

import idd_climate_models.constants as rfc

TC_RISK_REPO_ROOT_DIR = rfc.TC_RISK_REPO_ROOT_DIR
repo_name = rfc.repo_name
package_name = rfc.package_name

REFERENCE_CONFIG_PATH = rfc.REPO_ROOT / repo_name / "src" / package_name / "01_run_tc_risk" / "reference_namelist.py"

def create_custom_namelist_path(args):
    """
    Constructs the custom namelist path based on input arguments, 
    using the full nested output directory structure.
    """
    target_output_path_root = rfc.TC_RISK_OUTPUT_PATH / args.data_source
    output_dir = target_output_path_root / args.model / args.variant / args.scenario / args.time_bin
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
    base_dir = target_base_path / args.model / args.variant / args.scenario / args.time_bin
    
    # Full path to the parent output folder for the specific time bin
    # This value becomes namelist.output_directory
    output_dir_parent = target_output_path / args.model / args.variant / args.scenario / args.time_bin

    exp_name = f'{args.basin}' # This value becomes namelist.exp_name (e.g., 'GL')
    tracks_per_year = rfc.tc_risk_tracks_per_basin[args.basin]

    start_year, end_year = map(int, args.time_bin.split('-'))
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
            for draw in range(0, args.num_draws):
                # Run tc_risk
                compute.run_downscaling(args.basin)
                # Post-process to 
            
        print("✅ Run complete.")
    finally:
        os.chdir(original_cwd)
        sys.path = original_sys_path
        del os.environ['TC_RISK_NAMELIST_PATH']