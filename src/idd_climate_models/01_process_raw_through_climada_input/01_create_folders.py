"""
Create input/output folders for TC-risk AND CLIMADA input folders.

This script is called by the orchestrator as Level 1 of the workflow.
It creates:
  - TC_RISK_INPUT_PATH / data_source / model / variant / scenario / time_period
  - TC_RISK_OUTPUT_PATH / data_source / model / variant / scenario / time_period
  - CLIMADA_INPUT_PATH / data_source / model / variant / scenario / time_period / {basin} (for all basins)

Note: To generate a registry CSV, use 01_generate_folder_registry.py separately.
"""

import os
import argparse
import shutil
from pathlib import Path
import idd_climate_models.constants as rfc

def main():
    os.umask(0o002)

    parser = argparse.ArgumentParser(
        description='Create input and output folders for tc_risk pipeline and CLIMADA input folders'
    )
    parser.add_argument('--data_source', type=str, default='cmip6')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--variant', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--time_period', type=str, required=True)
    parser.add_argument('--delete_destination_folder', action='store_true',
                        help='Delete the output folders and all contents before creating them')
    args = parser.parse_args()

    # ========================================================================
    # PART 1: TC-RISK FOLDERS (original functionality)
    # ========================================================================
    
    # Define TC-risk paths
    tc_input_dir = rfc.TC_RISK_INPUT_PATH / args.data_source / args.model / args.variant / args.scenario / args.time_period
    tc_output_dir = rfc.TC_RISK_OUTPUT_PATH / args.data_source / args.model / args.variant / args.scenario / args.time_period
    
    tc_dirs_to_create = [tc_input_dir, tc_output_dir]

    for directory in tc_dirs_to_create:
        # Delete existing folder if requested
        if args.delete_destination_folder and directory.exists():
            print(f"Deleting existing TC-risk folder: {directory}")
            shutil.rmtree(directory)
        
        # mode=0o775 sets the permissions
        directory.mkdir(parents=True, exist_ok=True, mode=0o775)
        
        # Brute force: ensure the leaf directory is 775 
        os.chmod(directory, 0o775)
        
        print(f"Verified TC-risk directory and permissions (775): {directory}")

    # ========================================================================
    # PART 2: CLIMADA INPUT FOLDERS (new functionality)
    # ========================================================================
    
    BASINS = ['EP', 'NA', 'NI', 'SI', 'AU', 'SP', 'WP']
    
    # Fast delete strategy: rename entire time_period directory if it exists
    climada_time_period_dir = rfc.CLIMADA_INPUT_PATH / args.data_source / args.model / args.variant / args.scenario / args.time_period
    
    if args.delete_destination_folder and climada_time_period_dir.exists():
        import time
        import subprocess
        # Rename with unique timestamp
        suffix = f'_DELETE_{int(time.time() * 1000000)}'
        renamed_path = climada_time_period_dir.parent / (climada_time_period_dir.name + suffix)
        print(f"Renaming CLIMADA time_period folder for async deletion: {climada_time_period_dir.name}")
        climada_time_period_dir.rename(renamed_path)
        
        # Launch async deletion in background
        subprocess.Popen(
            ['ionice', '-c', '3', 'rm', '-rf', str(renamed_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    climada_dirs_created = []
    
    for basin in BASINS:
        # Define CLIMADA input path for this basin
        climada_input_dir = rfc.CLIMADA_INPUT_PATH / args.data_source / args.model / args.variant / args.scenario / args.time_period / basin
        
        # Create with 775 permissions (no deletion needed - already renamed entire time_period dir)
        climada_input_dir.mkdir(parents=True, exist_ok=True, mode=0o775)
        os.chmod(climada_input_dir, 0o775)
        
        climada_dirs_created.append(climada_input_dir)
        print(f"Verified CLIMADA directory and permissions (775): {climada_input_dir}")
    
    print(f"\n✅ Created {len(tc_dirs_to_create)} TC-risk directories and {len(climada_dirs_created)} CLIMADA directories")

if __name__ == "__main__":
    main()