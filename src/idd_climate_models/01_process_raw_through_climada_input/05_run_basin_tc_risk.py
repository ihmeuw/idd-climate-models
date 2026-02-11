"""
Run TC-risk downscaling for a specific basin with draw batching.

This script can operate in two modes:
1. Task-based mode: --task_id (reads task_assignments.csv)
2. Direct mode: --model --variant --scenario --time_period --basin --draw_start --draw_end
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from collections import defaultdict
import idd_climate_models.constants as rfc
from idd_climate_models.tc_risk_functions import (
    create_tc_risk_config_dict,
    execute_tc_risk_with_config
)
from idd_climate_models.zarr_functions import validate_single_draw


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
    Reads completion marker files (.draw_####.complete) created after successful validation.
    
    Args:
        draw_start: Starting draw number (0-249, inclusive)
        draw_end: Ending draw number (0-249, inclusive)
    
    Returns:
        set: Set of draw numbers (within range) that have completed and validated files
    """
    from idd_climate_models.zarr_functions import get_completed_draws_from_markers
    
    # Path to CLIMADA input (where markers live)
    climada_input = rfc.CLIMADA_INPUT_PATH / data_source / model / variant / scenario / time_period / basin
    
    # Read completion markers
    all_completed = get_completed_draws_from_markers(climada_input)
    
    # Filter to requested range
    existing_draws = set(d for d in all_completed if draw_start <= d <= draw_end)
    
    return existing_draws


def validate_batch_output(data_source, model, variant, scenario, time_period, basin, 
                         expected_start, expected_end, draws_list=None):
    """
    Validate that the batch produced the expected draws with correct storm counts.
    Uses validate_single_draw to perform 3-layer validation on each draw.
    
    Args:
        draws_list: Optional list of specific draws to validate. If None, validates all draws in [expected_start, expected_end]
    
    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    import time
    
    # Wait a moment for filesystem to sync
    time.sleep(2)
    
    output_path = rfc.CLIMADA_INPUT_PATH / data_source / model / variant / scenario / time_period / basin
    tc_risk_output_path = rfc.TC_RISK_OUTPUT_PATH / data_source / model / variant / scenario / time_period / basin
    
    time_parts = time_period.split('-')
    time_start_str = f'{int(time_parts[0]):04d}01'
    time_end_str = f'{int(time_parts[1]):04d}12'
    base_pattern = f'tracks_{basin}_{model}_{scenario}_{variant}_{time_start_str}_{time_end_str}'
    
    validation_errors = []
    
    # Determine which draws to validate
    if draws_list is not None:
        draws_to_validate = draws_list
    else:
        draws_to_validate = range(expected_start, expected_end + 1)
    
    # Check each draw individually
    for draw_num in draws_to_validate:
        suffix = get_draw_suffix(draw_num)
        zarr_file = output_path / f'{base_pattern}{suffix}.zarr'
        nc_file = tc_risk_output_path / f'{base_pattern}{suffix}.nc'
        
        # Use FULL validation (with spot-check) for fresh output
        # Do NOT delete on failure - this is validation only
        success, error = validate_single_draw(
            nc_file, zarr_file,
            spot_check=True,
            delete_on_failure=False
        )
        
        if not success:
            validation_errors.append(f"Draw {draw_num}: {error}")
    
    if validation_errors:
        error_msg = "; ".join(validation_errors)
        return False, error_msg
    
    return True, None


def process_single_combination(data_source, model, variant, scenario, time_period, basin, 
                               draw_start, draw_end, total_memory, script_start, draws_list=None):
    """
    Process a single model/variant/scenario/time_period/basin combination.
    
    Args:
        draws_list: Optional list of specific draws to process. If None, processes all draws in [draw_start, draw_end]
    
    Returns True if successful, False if failed.
    """
    if draws_list is not None:
        print("\n" + "=" * 80)
        print(f"Processing combination:")
        print(f"  {model}/{variant}/{scenario}/{time_period}/{basin}")
        print(f"  Draws: {draws_list} ({len(draws_list)} draws)")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print(f"Processing combination:")
        print(f"  {model}/{variant}/{scenario}/{time_period}/{basin}")
        print(f"  Draws: {draw_start}-{draw_end}")
        print("=" * 80)
    
    # Create args namespace for compatibility with existing functions
    from argparse import Namespace
    args = Namespace(
        data_source=data_source,
        model=model,
        variant=variant,
        scenario=scenario,
        time_period=time_period,
        basin=basin,
        draw_start=draw_start,
        draw_end=draw_end,
        total_memory=total_memory
    )

    # PART 1: Check which draws in this batch are missing
    import time
    check_start = time.time()
    print(f"\n[1/3] Checking for missing draws...")
    existing_draws = get_existing_draw_numbers(
        data_source, model, variant,
        scenario, time_period, basin,
        draw_start, draw_end
    )
    
    # If draws_list provided, only check those draws; otherwise check entire range
    if draws_list is not None:
        missing_draws = sorted([d for d in draws_list if d not in existing_draws])
    else:
        missing_draws = sorted([d for d in range(draw_start, draw_end + 1) 
                                if d not in existing_draws])
    check_time = time.time() - check_start
    print(f"      ⏱️  Missing draw check took {check_time:.1f}s")
    
    if not missing_draws:
        if draws_list is not None:
            print(f"✅ All specified draws already complete. Skipping.")
        else:
            print(f"✅ All draws in batch {draw_start}-{draw_end} already complete. Skipping.")
        return True
    
    total_requested = len(draws_list) if draws_list is not None else (draw_end - draw_start + 1)
    print(f"\nMissing draws in this batch: {len(missing_draws)}/{total_requested}")
    if len(missing_draws) <= 20:
        print(f"  {missing_draws}")
    else:
        print(f"  First 10: {missing_draws[:10]}")
        print(f"  Last 10: {missing_draws[-10:]}")
    
    # Pass only the missing draws to TC-Risk
    args.draw_start_batch = missing_draws[0]
    args.draw_end_batch = missing_draws[-1]
    args.draws_to_run = missing_draws
    
    # Create config dict
    config_dict = create_tc_risk_config_dict(args)
    
    # PART 2: Execute TC-risk with batching
    exec_start = time.time()
    print(f"\n[2/2] Running TC-risk for {len(missing_draws)} draws (with validation)...")
    success = execute_tc_risk_with_config(
        config_dict, 
        script_name='run_downscaling', 
        args=args,
        total_memory=total_memory
    )
    exec_time = time.time() - exec_start
    print(f"\n      ⏱️  TC-risk execution took {exec_time:.1f}s ({exec_time/60:.1f} min)")
    
    if not success:
        print(f"\n❌ TC-risk execution FAILED")
        return False
    
    print(f"\n✅ TC-risk execution completed successfully (validation done during processing)")
    
    # TIMING SUMMARY
    elapsed = time.time() - script_start
    print(f"\n⏱️  TIMING SUMMARY (this combination):")
    print(f"     Check missing draws: {check_time:.1f}s ({check_time/elapsed*100:.1f}%)")
    print(f"     TC-risk execution:   {exec_time:.1f}s ({exec_time/60:.1f} min, {exec_time/elapsed*100:.1f}%)")
    print(f"     TOTAL:               {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 80}")
    
    return True


def find_model_variant_scenario_timeperiods_for_basins(data_source, basins):
    """
    Find all model/variant/scenario/time_period combinations that have the specified basins.
    
    Args:
        data_source: Data source (e.g., 'cmip6')
        basins: List of basin codes (e.g., ['NA', 'EP'])
    
    Returns:
        dict: {basin: [(model, variant, scenario, time_period), ...]}
    """
    climada_input = rfc.CLIMADA_INPUT_PATH / data_source
    basin_combinations = defaultdict(list)
    
    print(f"\nScanning for model/variant/scenario/time_period combinations with basins: {basins}")
    
    for basin in basins:
        # Find all paths matching: data_source/model/variant/scenario/time_period/basin
        basin_dirs = list(climada_input.glob(f"*/*/*/*/{basin}"))
        
        for basin_dir in basin_dirs:
            parts = basin_dir.parts
            data_source_idx = parts.index(data_source)
            model = parts[data_source_idx + 1]
            variant = parts[data_source_idx + 2]
            scenario = parts[data_source_idx + 3]
            time_period = parts[data_source_idx + 4]
            
            basin_combinations[basin].append((model, variant, scenario, time_period))
    
    for basin in basins:
        print(f"  {basin}: {len(basin_combinations[basin])} combinations")
    
    return basin_combinations


def main():
    import time
    script_start = time.time()
    
    parser = argparse.ArgumentParser(
        description='Run TC-risk basin downscaling in task-based or direct mode'
    )
    parser.add_argument('--data_source', type=str, default='cmip6')
    parser.add_argument('--total_memory', type=str, required=True,
                        help='Total memory allocation (e.g., "20G", "30G")')
    
    # Task-based mode
    parser.add_argument('--task_id', type=int, required=False,
                        help='Task ID to process (reads from task_assignments.csv)')
    
    # Direct mode (all required if not using task_id)
    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--variant', type=str, required=False)
    parser.add_argument('--scenario', type=str, required=False)
    parser.add_argument('--time_period', type=str, required=False)
    parser.add_argument('--basin', type=str, required=False)
    parser.add_argument('--draw_start', type=int, required=False,
                        help='Starting draw number (0-249, inclusive)')
    parser.add_argument('--draw_end', type=int, required=False,
                        help='Ending draw number (0-249, inclusive)')
    
    args = parser.parse_args()

    # Determine which mode we're in
    if args.task_id is not None:
        # TASK-BASED MODE
        print("=" * 80)
        print(f"TASK-BASED MODE: Processing task_id {args.task_id}")
        print("=" * 80)
        
        # Read task assignments
        task_assignments_path = rfc.CLIMADA_INPUT_PATH / args.data_source / "task_assignments.csv"
        
        if not task_assignments_path.exists():
            print(f"\n❌ ERROR: Task assignments file not found: {task_assignments_path}")
            print("Run Level 0 of the orchestrator to create task assignments.")
            sys.exit(1)
        
        print(f"\nReading task assignments from: {task_assignments_path}")
        df_assignments = pd.read_csv(task_assignments_path, keep_default_na=False)
        
        # Get ALL rows for this task (one row per draw)
        task_df = df_assignments[df_assignments['task_id'] == args.task_id]
        
        if task_df.empty:
            print(f"\n❌ ERROR: No assignments found for task_id {args.task_id}")
            sys.exit(1)
        
        print(f"\nTask {args.task_id} has {len(task_df)} draw assignments")
        
        # Group by combination (model/variant/scenario/time_period/basin)
        # Each task should only have ONE combination, but let's be robust
        grouped = task_df.groupby(['model', 'variant', 'scenario', 'time_period', 'basin'])
        
        if len(grouped) != 1:
            print(f"\n❌ ERROR: Task {args.task_id} has {len(grouped)} different combinations (expected 1)")
            sys.exit(1)
        
        # Get the combination and its draws
        (model, variant, scenario, time_period, basin), group = next(iter(grouped))
        draws = sorted(group['draw'].tolist())
        
        print(f"\nTask {args.task_id} assignment:")
        print(f"  Model/Variant/Scenario: {model}/{variant}/{scenario}")
        print(f"  Time Period: {time_period}")
        print(f"  Basin: {basin}")
        print(f"  Draws: {draws} ({len(draws)} draws)")
        
        # Process this single combination with explicit draw list
        success = process_single_combination(
            args.data_source, model, variant, scenario, time_period, basin,
            draws[0], draws[-1], args.total_memory, script_start,
            draws_list=draws  # Pass explicit draw list to avoid processing gaps
        )
        
        # Final summary
        total_time = time.time() - script_start
        print(f"\n{'='*80}")
        print(f"TASK {args.task_id} COMPLETE")
        print(f"{'='*80}")
        print(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"Total task time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"{'='*80}")
        
        if not success:
            sys.exit(1)
        
    else:
        # DIRECT MODE
        # Validate that all required arguments are provided
        required_args = ['model', 'variant', 'scenario', 'time_period', 'basin', 'draw_start', 'draw_end']
        missing_args = [arg for arg in required_args if getattr(args, arg) is None]
        
        if missing_args:
            print(f"\n❌ ERROR: Direct mode requires either --task_id OR all of the following:")
            print(f"  --model, --variant, --scenario, --time_period, --basin, --draw_start, --draw_end")
            print(f"\nMissing: {', '.join(missing_args)}")
            sys.exit(1)
        
        print("=" * 80)
        print(f"DIRECT MODE: Processing single combination")
        print("=" * 80)
        
        success = process_single_combination(
            args.data_source, args.model, args.variant, args.scenario, 
            args.time_period, args.basin, args.draw_start, args.draw_end,
            args.total_memory, script_start
        )
        
        total_time = time.time() - script_start
        print(f"\n⏱️  TOTAL SCRIPT TIME: {total_time:.1f}s ({total_time/60:.1f} min)")
        
        if not success:
            sys.exit(1)


if __name__ == '__main__':
    main()
