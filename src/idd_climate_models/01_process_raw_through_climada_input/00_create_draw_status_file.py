"""
Generate a draw status file for a specific basin/time-period/scenario/variant/model.

This script validates existing output files and creates a CSV with NUM_DRAWS rows,
where each row indicates if that draw is complete (1) or incomplete (0).

Usage:
    python 00_create_draw_status_file.py \
        --data_source cmip6 \
        --model CMCC-ESM2 \
        --variant r1i1p1f1 \
        --scenario ssp126 \
        --time_period 2020-2024 \
        --basin NI
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import shutil

import idd_climate_models.constants as rfc
from idd_climate_models.zarr_functions import (
    validate_single_draw,
    create_draw_completion_marker,
    get_completed_draws_from_markers
)


def extract_draw_number(filename):
    """
    Extract draw number from filename using the correct mapping.
    tracks_BASIN_MODEL_SCENARIO_VARIANT_YYYYMM_YYYYMM.nc → Draw 0
    tracks_BASIN_MODEL_SCENARIO_VARIANT_YYYYMM_YYYYMM_e0.nc → Draw 1
    tracks_BASIN_MODEL_SCENARIO_VARIANT_YYYYMM_YYYYMM_e13.nc → Draw 14
    tracks_BASIN_MODEL_SCENARIO_VARIANT_YYYYMM_YYYYMM_e248.nc → Draw 249
    """
    try:
        if '_e' in filename.stem:
            # Get the number after '_e'
            draw_str = filename.stem.split('_e')[-1]
            # Add 1 because _e0 is draw 1, _e1 is draw 2, etc.
            return int(draw_str) + 1
        else:
            # No suffix means draw 0
            return 0
    except:
        return None


def create_draw_status_file(data_source, model, variant, scenario, time_period, basin, spot_check=False):
    """
    Create a draw status CSV file by validating existing outputs.
    Status file lives in CLIMADA input folder since draw is only complete when zarr exists.
    
    Returns:
        tuple: (status_file_path, num_complete, num_incomplete)
    """
    # Setup paths
    tc_risk_output = rfc.TC_RISK_OUTPUT_PATH / data_source / model / variant / scenario / time_period / basin
    climada_input = rfc.CLIMADA_INPUT_PATH / data_source / model / variant / scenario / time_period / basin
    
    # Status file lives in CLIMADA input folder
    climada_input.mkdir(parents=True, exist_ok=True)
    status_file = climada_input / "draw_status.csv"
    
    print(f"\n{'='*80}")
    print(f"Creating draw status file for:")
    print(f"  Model/Variant/Scenario: {model}/{variant}/{scenario}")
    print(f"  Time Period: {time_period}")
    print(f"  Basin: {basin}")
    print(f"{'='*80}")
    
    # Initialize all draws as incomplete: {draw_num: (netcdf_valid, zarr_valid)}
    draw_status = {draw: (0, 0) for draw in range(rfc.NUM_DRAWS)}
    
    # First, check for existing completion markers (fast check)
    existing_markers = get_completed_draws_from_markers(climada_input)
    if existing_markers:
        print(f"\nFound {len(existing_markers)} completion markers (skipping validation)")
        for draw_num in existing_markers:
            if draw_num < rfc.NUM_DRAWS:
                draw_status[draw_num] = (1, 1)
    
    # Check which draws have valid output files
    processed_zarr_files = set()  # Track which zarrs we've already checked
    if tc_risk_output.exists():
        print(f"\nValidating files in TC_RISK_OUTPUT and CLIMADA_INPUT...")
        nc_files = list(tc_risk_output.glob("tracks_*.nc"))
        print(f"  Found {len(nc_files)} NetCDF files")
        
        for nc_file in nc_files:
            draw_num = extract_draw_number(nc_file)
            if draw_num is None or draw_num >= rfc.NUM_DRAWS:
                continue
            
            # Skip if already marked complete (saves time)
            if draw_num in existing_markers:
                continue
            
            zarr_file = climada_input / nc_file.with_suffix('.zarr').name
            processed_zarr_files.add(zarr_file)  # Track this zarr
            
            # Validate using shared function
            success, error = validate_single_draw(
                nc_file, zarr_file, 
                spot_check=spot_check, 
                delete_on_failure=True
            )
            
            if error == "Zarr file missing":
                print(f"  ⚠️  Draw {draw_num:3d}: NetCDF exists, Zarr missing")
                draw_status[draw_num] = (1, 0)
            elif success:
                print(f"  ✓ Draw {draw_num:3d}: Both valid (verified)")
                draw_status[draw_num] = (1, 1)
                # Create completion marker
                create_draw_completion_marker(climada_input, draw_num)
            else:
                print(f"  ✗ Draw {draw_num:3d}: {error}")
                draw_status[draw_num] = (0, 0)
                if error not in ["NetCDF file missing", "Zarr file missing"]:
                    print(f"    Deleted both files")
    
    # Clean up orphaned Zarr files (Zarr exists but NC is missing)
    # Only check zarrs we haven't already processed
    if climada_input.exists():
        zarr_files = list(climada_input.glob("tracks_*.zarr"))
        for zarr_file in zarr_files:
            # Skip if we already processed this zarr in the first loop
            if zarr_file in processed_zarr_files:
                continue
                
            corresponding_nc = tc_risk_output / zarr_file.with_suffix('.nc').name
            if not corresponding_nc.exists():
                draw_num = extract_draw_number(zarr_file)
                print(f"  🗑️  Draw {draw_num:3d}: Orphaned Zarr file (no NetCDF)")
                try:
                    shutil.rmtree(zarr_file)
                    print(f"    Deleted orphaned Zarr")
                except Exception as e:
                    print(f"    Warning: Could not delete: {e}")
                    
    # Create DataFrame and save
    df = pd.DataFrame([
        {'draw': draw_num, 'netcdf_complete': nc_status, 'zarr_complete': zarr_status}
        for draw_num, (nc_status, zarr_status) in draw_status.items()
    ])
    
    df.to_csv(status_file, index=False)
    status_file.chmod(0o775)
    
    # Count fully complete draws (both NetCDF and Zarr valid)
    num_complete = sum(1 for nc, zarr in draw_status.values() if nc == 1 and zarr == 1)
    num_incomplete = rfc.NUM_DRAWS - num_complete
    num_netcdf_only = sum(1 for nc, zarr in draw_status.values() if nc == 1 and zarr == 0)
    
    print(f"\n{'='*80}")
    print(f"Draw status summary:")
    print(f"  Fully complete (both files):  {num_complete:3d}/{rfc.NUM_DRAWS}")
    print(f"  NetCDF only (Zarr missing):   {num_netcdf_only:3d}/{rfc.NUM_DRAWS}")
    print(f"  Incomplete (need both files): {num_incomplete:3d}/{rfc.NUM_DRAWS}")
    print(f"  Status file: {status_file}")
    print(f"{'='*80}\n")
    
    return status_file, num_complete, num_incomplete


def main():
    parser = argparse.ArgumentParser(description='Create draw status file for a basin')
    parser.add_argument('--data_source', required=True, help='Data source (e.g., cmip6)')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--variant', required=True, help='Variant name')
    parser.add_argument('--scenario', required=True, help='Scenario name')
    parser.add_argument('--time_period', required=True, help='Time period (e.g., 1970-1974)')
    parser.add_argument('--basin', required=True, help='Basin code (e.g., SP)')
    parser.add_argument('--spot_check', required=False, default=True, help='Whether to perform spot check validation (slower)')
    
    args = parser.parse_args()
    
    try:
        status_file, num_complete, num_incomplete = create_draw_status_file(
            args.data_source, args.model, args.variant, args.scenario,
            args.time_period, args.basin, args.spot_check
        )
        
        print(f"✅ Successfully created draw status file")
        print(f"   Complete: {num_complete}, Incomplete: {num_incomplete}")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to create draw status file")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
