"""
Run TC-risk downscaling for a specific basin with draw batching.
"""

import sys
import argparse
from pathlib import Path
import idd_climate_models.constants as rfc
from idd_climate_models.tc_risk_functions import (
    create_tc_risk_config_dict,
    execute_tc_risk_with_config
)
from idd_climate_models.zarr_functions import verify_zarr_integrity


def get_draw_suffix(draw_num):
    """
    Get the filename suffix for a given draw number.
    Draw 0 → '' (no suffix)
    Draw 1 → '_e0'
    Draw 2 → '_e1'
    ...
    Draw 249 → '_e248'
    """
    if draw_num == 0:
        return ''
    else:
        return f'_e{draw_num - 1}'


def get_existing_draw_numbers(data_source, model, variant, scenario, time_period, basin, 
                              draw_start, draw_end):
    """
    Check which draw numbers already exist within the specified range.
    Checks for .zarr files in the CLIMADA input path (final output location).
    Validates zarr integrity to exclude partial/corrupted files from job failures.
    
    Args:
        draw_start: Starting draw number (0-249, inclusive)
        draw_end: Ending draw number (0-249, inclusive)
    
    Returns:
        set: Set of draw numbers (within range) that have completed and validated files
    """
    # Check CLIMADA input path for .zarr files (the final output)
    output_path = rfc.CLIMADA_INPUT_PATH / data_source / model / variant / scenario / time_period / basin
    
    if not output_path.exists():
        return set()
    
    existing_draws = set()
    
    # Parse time_period
    time_period_parts = time_period.split('-')
    time_start_str = f'{int(time_period_parts[0]):04d}01'
    time_end_str = f'{int(time_period_parts[1]):04d}12'
    
    # Base pattern: tracks_{basin}_{model}_{scenario}_{variant}_{time_start}_{time_end}
    base_pattern = f'tracks_{basin}_{model}_{scenario}_{variant}_{time_start_str}_{time_end_str}'
    
    # Path to TC-risk output (original NC files)
    tc_risk_output_path = rfc.TC_RISK_OUTPUT_PATH / data_source / model / variant / scenario / time_period / basin
    
    # Only check draws within the requested range
    for draw_num in range(draw_start, draw_end + 1):
        suffix = get_draw_suffix(draw_num)
        zarr_file = output_path / f'{base_pattern}{suffix}.zarr'
        nc_file = tc_risk_output_path / f'{base_pattern}{suffix}.nc'
        
        if zarr_file.exists() and nc_file.exists():
            try:
                # Full verification: check storm count and spot-check 3 storms
                verify_zarr_integrity(str(nc_file), str(zarr_file), None)
                existing_draws.add(draw_num)
            except Exception as e:
                print(f"  ⚠️  Draw {draw_num} validation error ({e}), will be regenerated")
        elif zarr_file.exists() and not nc_file.exists():
            print(f"  ⚠️  Draw {draw_num} has zarr but missing NC file, will be regenerated")
    
    return existing_draws


def get_missing_draws_in_range(data_source, model, variant, scenario, time_period, basin, 
                                draw_start, draw_end):
    """
    Determine which draws in the specified range are missing.
    
    Args:
        draw_start: Starting draw number (0-249, inclusive)
        draw_end: Ending draw number (0-249, inclusive)
    
    Returns:
        list: List of missing draw numbers in sorted order
    """
    existing_draws = get_existing_draw_numbers(
        data_source, model, variant, scenario, time_period, basin,
        draw_start, draw_end
    )
    
    missing_draws = []
    for draw_num in range(draw_start, draw_end + 1):
        if draw_num not in existing_draws:
            missing_draws.append(draw_num)
    
    return sorted(missing_draws)

def validate_batch_output(data_source, model, variant, scenario, time_period, basin, 
                         expected_start, expected_end):
    """
    Validate that the batch produced the expected draws with correct storm counts.
    Checks each draw individually for:
    1. Both .nc and .zarr files exist
    2. Storm count matches between NC and zarr files
    
    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    import time
    import xarray as xr
    import zarr
    import numpy as np
    
    # Wait a moment for filesystem to sync
    time.sleep(2)
    
    output_path = rfc.CLIMADA_INPUT_PATH / data_source / model / variant / scenario / time_period / basin
    tc_risk_output_path = rfc.TC_RISK_OUTPUT_PATH / data_source / model / variant / scenario / time_period / basin
    
    time_parts = time_period.split('-')
    time_start_str = f'{int(time_parts[0]):04d}01'
    time_end_str = f'{int(time_parts[1]):04d}12'
    base_pattern = f'tracks_{basin}_{model}_{scenario}_{variant}_{time_start_str}_{time_end_str}'
    
    validation_errors = []
    
    # Check each draw individually
    for draw_num in range(expected_start, expected_end + 1):
        suffix = get_draw_suffix(draw_num)
        zarr_file = output_path / f'{base_pattern}{suffix}.zarr'
        nc_file = tc_risk_output_path / f'{base_pattern}{suffix}.nc'
        
        # Check both files exist
        if not zarr_file.exists():
            validation_errors.append(f"Draw {draw_num}: missing zarr file")
            continue
        if not nc_file.exists():
            validation_errors.append(f"Draw {draw_num}: missing NC file")
            continue
        
        try:
            # Count valid storms in NC file (same logic as verify_zarr_integrity)
            ds_nc = xr.open_dataset(nc_file)
            n_trk_nc = ds_nc.sizes.get("n_trk", 0)
            
            valid_count = 0
            for i in range(n_trk_nc):
                if not (np.isnan(ds_nc["tc_years"][i].item()) or 
                       np.isnan(ds_nc["tc_month"][i].item())):
                    lon = ds_nc["lon_trks"][i].values
                    lat = ds_nc["lat_trks"][i].values
                    valid_idx = np.isfinite(lon) & np.isfinite(lat)
                    if valid_idx.sum() > 0:
                        valid_count += 1
            ds_nc.close()
            
            # Count storms in zarr file
            zarr_store = zarr.open(str(zarr_file), mode='r')
            zarr_storm_count = len(list(zarr_store.group_keys()))
            
            # Verify storm counts match
            if valid_count != zarr_storm_count:
                validation_errors.append(
                    f"Draw {draw_num}: storm count mismatch (NC={valid_count}, zarr={zarr_storm_count})"
                )
            
        except Exception as e:
            validation_errors.append(f"Draw {draw_num}: validation error - {e}")
    
    if validation_errors:
        error_msg = "; ".join(validation_errors)
        return False, error_msg
    
    return True, None

def main():
    import time
    script_start = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=str, default='cmip6')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--variant', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--time_period', type=str, required=True)
    parser.add_argument('--basin', type=str, required=True)
    parser.add_argument('--draw_start', type=int, required=True,
                        help='Starting draw number (0-249, inclusive)')
    parser.add_argument('--draw_end', type=int, required=True,
                        help='Ending draw number (0-249, inclusive)')
    parser.add_argument('--total_memory', type=str, required=True,
                        help='Total memory allocation (e.g., "20G", "30G")')
    
    args = parser.parse_args()

    print("=" * 80)
    print(f"Basin TC-risk batch:")
    print(f"  {args.model}/{args.variant}/{args.scenario}/{args.time_period}/{args.basin}")
    print(f"  Draws: {args.draw_start}-{args.draw_end}")
    print("=" * 80)

    # PART 1: Check which draws in this batch are missing
    check_start = time.time()
    print(f"\n[1/3] Checking for missing draws...")
    missing_draws = get_missing_draws_in_range(
        args.data_source, args.model, args.variant,
        args.scenario, args.time_period, args.basin,
        args.draw_start, args.draw_end
    )
    check_time = time.time() - check_start
    print(f"      ⏱️  Missing draw check took {check_time:.1f}s")
    
    if not missing_draws:
        total_time = time.time() - script_start
        print(f"✅ All draws in batch {args.draw_start}-{args.draw_end} already complete. Skipping.")
        print(f"\n⏱️  TOTAL SCRIPT TIME: {total_time:.1f}s")
        return
    
    print(f"\nMissing draws in this batch: {len(missing_draws)}/{args.draw_end - args.draw_start + 1}")
    if len(missing_draws) <= 20:
        print(f"  {missing_draws}")
    else:
        print(f"  First 10: {missing_draws[:10]}")
        print(f"  Last 10: {missing_draws[-10:]}")
    
    # Pass only the missing draws to TC-Risk
    args.draw_start_batch = missing_draws[0] if missing_draws else args.draw_start
    args.draw_end_batch = missing_draws[-1] if missing_draws else args.draw_end
    args.draws_to_run = missing_draws  # Pass the actual list of draws to run
    
    # Create config dict (NO namelist file!)
    config_dict = create_tc_risk_config_dict(args)
    
    # PART 2: Execute TC-risk with batching
    exec_start = time.time()
    print(f"\n[2/3] Running TC-risk for {len(missing_draws)} draws...")
    success = execute_tc_risk_with_config(
        config_dict, 
        script_name='run_downscaling', 
        args=args,
        total_memory=args.total_memory
    )
    exec_time = time.time() - exec_start
    print(f"\n      ⏱️  TC-risk execution took {exec_time:.1f}s ({exec_time/60:.1f} min)")
    
    if not success:
        print(f"\n❌ TC-risk execution FAILED")
        sys.exit(1)

    # PART 3: VALIDATE BATCH OUTPUT
    validation_start = time.time()
    print(f"\n[3/3] Validating batch output...")
    print(f"{'=' * 80}")
    
    success, error = validate_batch_output(
        args.data_source, args.model, args.variant,
        args.scenario, args.time_period, args.basin,
        args.draw_start, args.draw_end
    )
    validation_time = time.time() - validation_start
    print(f"\n      ⏱️  Validation took {validation_time:.1f}s")
    
    if not success:
        print(f"\n❌ VALIDATION FAILED: {error}")
        sys.exit(1)  # Exit with error code so Jobmon marks task as failed
    
    print(f"\n✅ Validation passed: all draws {args.draw_start}-{args.draw_end} present")
    print(f"{'=' * 80}")
    
    # TIMING SUMMARY
    total_time = time.time() - script_start
    print(f"\n⏱️  TIMING SUMMARY:")
    print(f"     Check missing draws: {check_time:.1f}s ({check_time/total_time*100:.1f}%)")
    print(f"     TC-risk execution:   {exec_time:.1f}s ({exec_time/60:.1f} min, {exec_time/total_time*100:.1f}%)")
    print(f"     Validation:          {validation_time:.1f}s ({validation_time/total_time*100:.1f}%)")
    print(f"     TOTAL:               {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()