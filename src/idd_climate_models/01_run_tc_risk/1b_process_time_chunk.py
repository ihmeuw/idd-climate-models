import argparse
import sys
import os
import subprocess
from pathlib import Path
import xarray as xr
import numpy as np
from xarray.backends.file_manager import FILE_CACHE
import idd_climate_models.constants as rfc
from idd_climate_models.climate_file_functions import is_curvilinear_grid, regrid_with_cdo, recombine_variable_files
from idd_climate_models.validation_functions import is_monthly, filter_files_by_time_bin

def main():
    parser = argparse.ArgumentParser(description='Process a single variable/frequency combination')
    parser.add_argument('--data_source', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--time_bin', required=True)
    parser.add_argument('--variable', required=True)
    parser.add_argument('--grid', required=True)
    parser.add_argument('--frequency', required=True)
    parser.add_argument('--needs_regridding_str', required=True, help='String that indicates if regridding is needed')

    args = parser.parse_args()
    args.needs_regridding = args.needs_regridding_str.lower() == 'true'

    # --- Directory setup ---
    processed_data_path = Path(rfc.PROCESSED_DATA_PATH)
    target_base_path = rfc.TC_RISK_INPUT_PATH / args.data_source

    source_dir = processed_data_path / args.data_source / args.model / args.variant / args.scenario / args.variable / args.grid / args.frequency
    target_dir = target_base_path / args.model / args.variant / args.scenario / args.time_bin
    
    print(f"Processing {args.variable} for {target_dir}")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # --- File finding and combining logic ---
    combined_files = 0
    errors = []
    
    bin_name = args.time_bin
    bin_start_str, bin_end_str = args.time_bin.split('-')
    bin_start = int(bin_start_str)
    bin_end = int(bin_end_str)
    
    if is_monthly(args.frequency):
        date_start = f"{bin_start}01" 
        date_end = f"{bin_end}12" 
    else:
        date_start = f"{bin_start}0101" 
        date_end = f"{bin_end}1231" 
    
    files_in_range = filter_files_by_time_bin(
        str(source_dir), args.variable, bin_start, bin_end, args.frequency,
        args.model, args.scenario, args.variant, args.grid
    )
    
    if files_in_range:
        # Check if files need regridding to determine output grid label
        needs_regridding = args.needs_regridding
        output_grid = 'gr' if needs_regridding else args.grid
        
        output_filename = f"{args.variable}_{args.frequency}_{args.model}_{args.scenario}_{args.variant}_{output_grid}_{date_start}-{date_end}.nc"
        output_path = target_dir / output_filename
        
        print(f"  → Combining {len(files_in_range)} files...")
        if needs_regridding:
            print(f"  → Output grid will be: {output_grid} (regridded from {args.grid})")
        
        success, was_regridded = recombine_variable_files(
            sorted(set(files_in_range)), output_path, args.variable
        )
        
        if success:
            combined_files += 1
            print(f"  ✅ Combined → {output_filename}")
        else:
            error_msg = f"Failed to combine {args.variable} files for {bin_name}" 
            errors.append(error_msg)
            print(f"  ❌ {error_msg}")
    else:
        print(f"  ⚠️  No files found for {args.variable} in {bin_start}-{bin_end}")
    
    print(f"\n✅ Completed {args.variable}/{args.frequency}")
    print(f"   Combined files: {combined_files}")
    
    if errors:
        print(f"   Errors: {len(errors)}")
        sys.exit(1)


if __name__ == "__main__":
    main()