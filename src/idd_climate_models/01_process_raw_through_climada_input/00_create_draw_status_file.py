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
    Create a draw status CSV file by checking for netCDF files and completion markers.
    Status file lives in CLIMADA input folder.
    
    Returns:
        tuple: (status_file_path, num_complete, num_incomplete)
    """
    # Setup paths
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
    
    # Initialize all draws as incomplete
    draw_status = {draw: 0 for draw in range(rfc.NUM_DRAWS)}
    
    # Check for existing completion markers (.nc_draw_*.complete)
    marker_files = list(climada_input.glob(".nc_draw_*.complete"))
    existing_markers = set()
    for marker_file in marker_files:
        try:
            draw_num = int(marker_file.stem.split('_')[-1])
            if draw_num < rfc.NUM_DRAWS:
                existing_markers.add(draw_num)
                draw_status[draw_num] = 1
        except (IndexError, ValueError):
            continue
    
    if existing_markers:
        print(f"\nFound {len(existing_markers)} completion markers")
        for draw_num in sorted(existing_markers):
            print(f"  ✓ Draw {draw_num:3d}: Complete (has marker)")
    
    # Check for netCDF files without markers (for backwards compatibility)
    if climada_input.exists():
        time_parts = time_period.split('-')
        time_start_str = f'{int(time_parts[0]):04d}01'
        time_end_str = f'{int(time_parts[1]):04d}12'
        base_pattern = f'tracks_{basin}_{model}_{scenario}_{variant}_{time_start_str}_{time_end_str}'
        
        nc_files = list(climada_input.glob(f"{base_pattern}*.nc"))
        
        if nc_files:
            print(f"\nFound {len(nc_files)} netCDF files")
            for nc_file in nc_files:
                draw_num = extract_draw_number(nc_file)
                if draw_num is None or draw_num >= rfc.NUM_DRAWS:
                    continue
                
                # If already marked complete, skip
                if draw_num in existing_markers:
                    continue
                
                # File exists but no marker - mark as complete and create marker
                draw_status[draw_num] = 1
                marker_path = climada_input / f".nc_draw_{draw_num:04d}.complete"
                marker_path.touch()
                try:
                    marker_path.chmod(0o775)
                except:
                    pass
                print(f"  ✓ Draw {draw_num:3d}: File exists, created marker")
                    
    # Create DataFrame and save
    df = pd.DataFrame([
        {'draw': draw_num, 'complete': status}
        for draw_num, status in draw_status.items()
    ])
    
    df.to_csv(status_file, index=False)
    status_file.chmod(0o775)
    
    # Count complete draws
    num_complete = sum(1 for status in draw_status.values() if status == 1)
    num_incomplete = rfc.NUM_DRAWS - num_complete
    
    print(f"\n{'='*80}")
    print(f"Draw status summary:")
    print(f"  Complete:   {num_complete:3d}/{rfc.NUM_DRAWS}")
    print(f"  Incomplete: {num_incomplete:3d}/{rfc.NUM_DRAWS}")
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
