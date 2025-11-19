import os
import shutil
import sys
from pathlib import Path

import idd_climate_models.constants as rfc
TC_RISK_REPO_ROOT_DIR = rfc.TC_RISK_REPO_ROOT_DIR

# --- Execution Function ---
def execute_tc_risk(tc_risk_repo_path, basin_name):
    """
    Sets up the environment and runs the TC-risk model components.

    Args:
        tc_risk_repo_path (str): The absolute path to the root of the
                                 TC-risk repository (e.g., ~/repos/tropical_cyclone_risk).
        basin_name (str): The basin identifier to run (e.g., 'NA').
    """
    
    if str(tc_risk_repo_path) not in sys.path:
        sys.path.insert(0, str(tc_risk_repo_path))
    
    # 2. Import Modules AFTER path modification
    try:
        import namelist
        from scripts import generate_land_masks
        # Assuming 'compute.py' exists inside the 'util' directory
        from util import compute 
    except ImportError as e:
        print(f"❌ Error importing TC-risk modules. Check if the path {tc_risk_repo_path} is correct and contains the required modules.")
        print(f"Details: {e}")
        # IMPORTANT: Remove the path we temporarily added before exiting
        sys.path.pop(0)
        return

    # --- Original Execution Logic ---
    
    # Get the absolute path to the namelist file being used
    # It is located directly in the root of the TC-risk repo.
    namelist_file_abs_path = Path(tc_risk_repo_path) / 'namelist.py'

    # Ensure output directory and copy namelist
    f_base = Path(namelist.output_directory) / namelist.exp_name
    os.makedirs(f_base, exist_ok=True)
    print(f'Saving model output to {f_base}')
    
    # 3. Use Absolute Path for shutil.copyfile
    shutil.copyfile(namelist_file_abs_path, f_base / 'namelist.py')

    # Run preliminary steps
    generate_land_masks.generate_land_masks()
    compute.compute_downscaling_inputs()

    # Run the main compute step
    print(f'Running tracks for basin {basin_name}...')
    compute.run_downscaling(basin_name)
    
    # Clean up the system path after execution
    sys.path.pop(0)


# --- Main Execution Block ---
if __name__ == '__main__':
        
    # Expand the user (~) directory for robustness
    REPO_PATH = Path(TC_RISK_REPO_ROOT_DIR).expanduser().resolve()
    
    try:
        # Reads the basin name from the command line argument (e.g., 'NA')
        BASIN_ARG = sys.argv[1]
    except IndexError:
        print("Usage: python your_wrapper_script.py <BASIN_NAME>")
        sys.exit(1)

    execute_tc_risk(REPO_PATH, BASIN_ARG)