"""
Create both input and output folders for a given model/variant/scenario/time_period.

This script is called by the orchestrator as Level 1 of the workflow.
It creates:
  - TC_RISK_INPUT_PATH / data_source / model / variant / scenario / time_period
  - TC_RISK_OUTPUT_PATH / data_source / model / variant / scenario / time_period

This script also appends to a CSV log of all created directories.
"""

import os
import argparse
import shutil
from pathlib import Path
import pandas as pd
import idd_climate_models.constants as rfc

def main():
    os.umask(0o002)

    parser = argparse.ArgumentParser(
        description='Create input and output folders for tc_risk pipeline'
    )
    parser.add_argument('--data_source', type=str, default='cmip6')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--variant', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--time_period', type=str, required=True)
    parser.add_argument('--delete_destination_folder', action='store_true',
                        help='Delete the output folders and all contents before creating them')
    args = parser.parse_args()

    # Define paths
    input_dir = rfc.TC_RISK_INPUT_PATH / args.data_source / args.model / args.variant / args.scenario / args.time_period
    output_dir = rfc.TC_RISK_OUTPUT_PATH / args.data_source / args.model / args.variant / args.scenario / args.time_period
    
    dirs_to_create = [input_dir, output_dir]

    for directory in dirs_to_create:
        # Delete existing folder if requested
        if args.delete_destination_folder and directory.exists():
            print(f"Deleting existing folder: {directory}")
            shutil.rmtree(directory)
        
        # mode=0o775 sets the permissions
        directory.mkdir(parents=True, exist_ok=True, mode=0o775)
        
        # Brute force: ensure the leaf directory is 775 
        # (Useful if the directory already existed with wrong permissions)
        os.chmod(directory, 0o775)
        
        print(f"Verified directory and permissions (775): {directory}")

    # Log this combination to a CSV file
    log_file = rfc.TC_RISK_INPUT_PATH / args.data_source / "folder_paths_registry.csv"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    new_rows = pd.DataFrame([
        {
            'model': args.model,
            'variant': args.variant,
            'scenario': args.scenario,
            'time_period': args.time_period,
            'input_path': str(input_dir),
            'output_path': str(output_dir),
        }
    ])
    
    # Append to existing CSV or create new one
    if log_file.exists():
        existing_df = pd.read_csv(log_file)
        combined_df = pd.concat([existing_df, new_rows], ignore_index=True)
    else:
        combined_df = new_rows
    
    # Remove duplicates (keep first occurrence)
    combined_df = combined_df.drop_duplicates(
        subset=['model', 'variant', 'scenario', 'time_period'], 
        keep='first'
    )
    
    combined_df.to_csv(log_file, index=False)
    print(f"Logged paths to: {log_file}")

if __name__ == "__main__":
    main()