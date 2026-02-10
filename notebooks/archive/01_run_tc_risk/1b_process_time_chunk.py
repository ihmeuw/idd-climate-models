import argparse
import sys
import os
import subprocess
from pathlib import Path
import xarray as xr
import numpy as np
from xarray.backends.file_manager import FILE_CACHE
import idd_climate_models.constants as rfc
from idd_climate_models.climate_file_functions import (
    should_regrid, 
    regrid_with_cdo, 
    recombine_variable_files
)
from idd_climate_models.validation_functions import is_monthly, filter_files_by_time_period

def main():
    parser = argparse.ArgumentParser(description='Process a single variable/frequency combination')
    parser.add_argument('--data_source', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--time_period', required=True)
    parser.add_argument('--variable', required=True)
    parser.add_argument('--grid', required=True)
    parser.add_argument('--frequency', required=True)

    args = parser.parse_args()

    # --- 1. DEFINE TARGET GRID ---
    TARGET_GRID = 'r360x180'  # 1°×1° global grid - CDO has built-in support for this notation

    # --- 2. DIRECTORY SETUP ---
    processed_data_path = Path(rfc.PROCESSED_DATA_PATH)
    target_base_path = rfc.TC_RISK_INPUT_PATH / args.data_source

    source_dir = processed_data_path / args.data_source / args.model / args.variant / args.scenario / args.variable / args.grid / args.frequency
    target_dir = target_base_path / args.model / args.variant / args.scenario / args.time_period
    
    print(f"\n{'='*80}")
    print(f"Processing {args.variable}/{args.frequency} for {args.model}")
    print(f"Time period: {args.time_period}")
    print(f"Source grid: {args.grid}")
    print(f"Target grid: {TARGET_GRID}")
    print(f"{'='*80}\n")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 3. FILE FINDING AND COMBINING LOGIC ---
    combined_files = 0
    errors = []
    
    bin_name = args.time_period
    bin_start_str, bin_end_str = args.time_period.split('-')
    bin_start = int(bin_start_str)
    bin_end = int(bin_end_str)
    
    # Construct date strings based on frequency
    if is_monthly(args.frequency):
        date_start = f"{bin_start}01" 
        date_end = f"{bin_end}12" 
    else:
        date_start = f"{bin_start}0101" 
        date_end = f"{bin_end}1231" 
    
    # Find files in the time range
    files_in_range = filter_files_by_time_period(
        str(source_dir), args.variable, bin_start, bin_end, args.frequency,
        args.model, args.scenario, args.variant, args.grid
    )
    
    if files_in_range:
        # Check if first file needs regridding to determine output grid label
        needs_regridding = should_regrid(files_in_range[0], target_grid=TARGET_GRID)
        
        # Set output grid label: 'gr' if regridded, keep original if not
        output_grid = 'gr' if needs_regridding else args.grid
        
        # Construct output filename
        output_filename = f"{args.variable}_{args.frequency}_{args.model}_{args.scenario}_{args.variant}_{output_grid}_{date_start}-{date_end}.nc"
        output_path = target_dir / output_filename
        
        print(f"  → Combining {len(files_in_range)} files...")
        if needs_regridding:
            print(f"  → Will regrid from {args.grid} to {output_grid} ({TARGET_GRID})")
        else:
            print(f"  → Files already on target grid, no regridding needed")
        
        # Combine files (with automatic regridding if needed)
        success, was_regridded = recombine_variable_files(
            sorted(set(files_in_range)), 
            output_path, 
            args.variable, 
            target_grid=TARGET_GRID
        )
        
        if success:
            combined_files += 1
            regrid_status = "regridded" if was_regridded else "combined"
            print(f"  ✅ Successfully {regrid_status} → {output_filename}")
        else:
            error_msg = f"Failed to combine {args.variable} files for {bin_name}" 
            errors.append(error_msg)
            print(f"  ❌ {error_msg}")
    else:
        print(f"  ⚠️  No files found for {args.variable} in {bin_start}-{bin_end}")
    
    # --- 4. FINAL SUMMARY ---
    print(f"\n{'='*80}")
    print(f"✅ Completed {args.variable}/{args.frequency}")
    print(f"   Combined files: {combined_files}")
    
    if errors:
        print(f"   ❌ Errors: {len(errors)}")
        for error in errors:
            print(f"      - {error}")
        sys.exit(1)
    else:
        print(f"   🎉 All operations successful!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()