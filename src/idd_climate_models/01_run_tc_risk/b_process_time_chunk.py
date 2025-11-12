import argparse
import sys
import re
import os
from pathlib import Path
import xarray as xr
import idd_climate_models.constants as rfc
from idd_climate_models.validation_functions import is_monthly, filter_files_by_time_bin


def recombine_variable_files(file_paths, output_path, variable, compression_level=7):
    """
    Combine multiple NetCDF files into a single file with high Zlib compression (complevel=7).
    """
    if not file_paths:
        return False
    
    try:
        # Open and combine datasets (logic omitted, assume 'combined' is defined)
        datasets = [xr.open_dataset(fpath) for fpath in file_paths]
        combined = xr.concat(datasets, dim='time', data_vars='all').sortby('time')
        
        # --- ADD COMPRESSION ENCODING ---
        
        # 1. Define the encoding dictionary
        encoding = {
                var: {
                    'zlib': True, 
                    'complevel': compression_level,
                    'shuffle': True,
                    'chunksizes': None
                } 
                for var in combined.data_vars
            }
        
        # 2. Write to output file using the encoding
        combined.to_netcdf(output_path, encoding=encoding, engine='netcdf4')
        return True
        
    except Exception as e:
        print(f"Error combining files: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Process a single variable/frequency combination')
    # All required arguments for path and lookup
    parser.add_argument('--data_source', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--time_bin', required=True)
    parser.add_argument('--variable', required=True)
    parser.add_argument('--grid', required=True)
    parser.add_argument('--frequency', required=True)

    args = parser.parse_args()

    processed_data_path = Path(rfc.PROCESSED_DATA_PATH)
    target_base_path = rfc.TC_RISK_INPUT_PATH / args.data_source

    source_dir = processed_data_path / args.data_source / args.model / args.variant / args.scenario / args.variable / args.grid / args.frequency
    target_dir = target_base_path / args.model / args.variant / args.scenario / args.time_bin
    print(f"Processing {args.variable} for {target_dir}")
        
    combined_files = 0
    errors = []
    
    # Setup bin variables for local use
    bin_name = args.time_bin
    bin_start_str, bin_end_str = args.time_bin.split('-')
    bin_start = int(bin_start_str)
    bin_end = int(bin_end_str)
    
    # Determine date range format based on frequency
    if is_monthly(args.frequency):
        date_start = f"{bin_start}01" 
        date_end = f"{bin_end}12" 
    else:
        date_start = f"{bin_start}0101" 
        date_end = f"{bin_end}1231" 
    
    # Target path (folders should already exist from Level 1 task)
    output_filename = f"{args.variable}_{args.frequency}_{args.model}_{args.scenario}_{args.variant}_{args.grid}_{date_start}-{date_end}.nc"
    output_path = target_dir / output_filename
    
    # Find yearly files in this time bin range
    files_in_range = filter_files_by_time_bin(
        str(source_dir), args.variable, bin_start, bin_end, args.frequency,
        args.model, args.scenario, args.variant, args.grid
    )
    
    # Recombine files for this variable if we have any
    if files_in_range:
        success = recombine_variable_files(
            sorted(set(files_in_range)), output_path, args.variable # <-- REMOVED args.dry_run
        )
        
        if success:
            combined_files += 1
            print(f"  ✓ Combined {len(files_in_range)} files → {output_filename}")
        else:
            # Uses locally defined bin_name (fix for NameError)
            error_msg = f"Failed to combine {args.variable} files for {bin_name}" 
            errors.append(error_msg)
            print(f"  ✗ {error_msg}")
    else:
        print(f"  - No files found for {args.variable} in {bin_start}-{bin_end}")
    
    print(f"✅ Completed {args.variable}/{args.frequency}")
    print(f"  Combined files: {combined_files}")
    
    if errors:
        print(f"  Errors: {len(errors)}")
        for error in errors:
            print(f"    - {error}")
        sys.exit(1)

if __name__ == "__main__":
    main()