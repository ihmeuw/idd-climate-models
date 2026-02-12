"""
Create task assignments file from draw status files.

This script reads all draw_status.csv files under the data source directory
and creates a level_4_task_assignments.csv file that distributes incomplete draws evenly across tasks.

Can operate in two modes:
1. Smart mode (default): Read status files and only assign incomplete draws
2. Full run mode (--full_run): Create uniform assignments for ALL draws (ignores status files)

Usage:
    # Smart mode (only run incomplete draws)
    python 00_create_level_4_task_assignments.py \\
        --data_source cmip6 \\
        --draws_per_batch 25
    
    # Full run mode (run all draws uniformly)
    python 00_create_level_4_task_assignments.py \\
        --data_source cmip6 \\
        --draws_per_batch 25 \\
        --full_run
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from collections import defaultdict

import idd_climate_models.constants as rfc
import idd_climate_models.orchestrator_utils as utils
from idd_climate_models.time_period_functions import get_time_bins_path


def create_level_4_task_assignments_from_status_files(data_source, draws_per_batch, max_period_duration=None):
    """
    Create task assignments per (model, variant, scenario, time_period, basin) combination.
    
    For each combination:
    - Read draw_status.csv
    - Find incomplete draws
    - Split into batches of draws_per_batch
    - Create one task per batch
    
    Only processes combinations that exist in the time_bins file (same validation as orchestrator).
    
    Args:
        data_source: Data source (e.g., 'cmip6')
        draws_per_batch: Target number of draws per task
        max_period_duration: Maximum time period duration in years (None = use original bins)
    
    Returns:
        tuple: (assignments_file_path, total_tasks_needed)
    """
    climada_base = rfc.CLIMADA_INPUT_PATH / data_source
    
    print(f"\n{'='*80}")
    print(f"Creating task assignments from status files")
    print(f"  Data source: {data_source}")
    print(f"  Base path: {climada_base}")
    print(f"  Target draws per batch: {draws_per_batch}")
    print(f"{'='*80}")
    
    # Load time bins to determine valid combinations (same logic as orchestrator)
    print(f"\nLoading time bins to validate combinations...")
    print(f"  MAX_PERIOD_DURATION: {max_period_duration}")
    time_bins_path = get_time_bins_path(max_period_duration)
    time_bins_df = pd.read_csv(time_bins_path)
    
    # Create set of valid (model, variant, scenario, time_period) combinations from time bins
    time_bins_df['time_period'] = time_bins_df['start_year'].astype(str) + '-' + time_bins_df['end_year'].astype(str)
    valid_combinations = set()
    for _, row in time_bins_df.iterrows():
        valid_combinations.add((row['model'], row['variant'], row['scenario'], row['time_period']))
    
    print(f"Found {len(valid_combinations)} valid combinations in time bins")
    
    # Find all draw_status.csv files, filtering out invalid combinations
    all_status_files = list(climada_base.glob("*/*/*/*/*/draw_status.csv"))
    
    if not all_status_files:
        print(f"\n⚠️  No status files found - cannot create smart assignments")
        print(f"Run with --full_run to create uniform 'run all' assignments")
        sys.exit(1)
    
    print(f"Found {len(all_status_files)} total status files, filtering to valid combinations...")
    
    # Filter to valid combinations only
    status_files = []
    for status_file in all_status_files:
        # Skip paths with _DELETE_
        if '_DELETE_' in str(status_file):
            continue
        
        # Extract combination from path
        parts = status_file.parts
        try:
            data_source_idx = parts.index(data_source)
            model = parts[data_source_idx + 1]
            variant = parts[data_source_idx + 2]
            scenario = parts[data_source_idx + 3]
            time_period = parts[data_source_idx + 4]
            
            # Check if combination is valid
            if (model, variant, scenario, time_period) in valid_combinations:
                status_files.append(status_file)
        except (ValueError, IndexError):
            # Path doesn't match expected structure, skip
            continue
    
    if not status_files:
        print(f"\n⚠️  No valid status files found after filtering")
        print(f"     (All {len(all_status_files)} files were either deleted or not in time bins)")
        sys.exit(1)
    
    print(f"Filtered to {len(status_files)} valid status files")
    print(f"Processing each combination...\n")
    
    # Collect task information per combination
    assignments = []
    task_id = 1
    total_complete = 0
    total_incomplete = 0
    combinations_with_work = 0
    combinations_complete = 0
    
    for status_file in sorted(status_files):
        # Extract model/variant/scenario/time_period/basin from path
        # Path: .../data_source/model/variant/scenario/time_period/basin/draw_status.csv
        parts = status_file.parts
        data_source_idx = parts.index(data_source)
        model = parts[data_source_idx + 1]
        variant = parts[data_source_idx + 2]
        scenario = parts[data_source_idx + 3]
        time_period = parts[data_source_idx + 4]
        basin = parts[data_source_idx + 5]
        
        # Get path to this combination's directory
        combination_path = status_file.parent
        
        # Read status file to get list of all draws
        df = pd.read_csv(status_file, keep_default_na=False)
        all_draws = set(df['draw'].tolist())
        
        # Get completed draws from .nc_draw_####.complete marker files
        marker_files = list(combination_path.glob(".nc_draw_*.complete"))
        completed_draws = set()
        for marker_file in marker_files:
            try:
                draw_num = int(marker_file.stem.split('_')[-1])
                completed_draws.add(draw_num)
            except (IndexError, ValueError):
                continue
        
        # Find incomplete draws (those without completion markers)
        incomplete_draws = sorted(all_draws - completed_draws)
        num_complete = len(completed_draws)
        num_incomplete = len(incomplete_draws)
        
        total_complete += num_complete
        total_incomplete += num_incomplete
        
        if num_incomplete == 0:
            # This combination is complete - skip
            combinations_complete += 1
            continue
        
        combinations_with_work += 1
        
        # Split incomplete draws into batches
        num_batches = (num_incomplete + draws_per_batch - 1) // draws_per_batch
        
        print(f"  {model}/{variant}/{scenario}/{time_period}/{basin}:")
        print(f"    Incomplete: {num_incomplete}, Complete: {num_complete}")
        print(f"    Creating {num_batches} task(s):")
        
        for batch_idx in range(num_batches):
            batch_start_idx = batch_idx * draws_per_batch
            batch_end_idx = min(batch_start_idx + draws_per_batch, num_incomplete)
            batch_draws = incomplete_draws[batch_start_idx:batch_end_idx]
            
            # Create ONE ROW PER DRAW
            for draw in batch_draws:
                assignments.append({
                    'task_id': task_id,
                    'model': model,
                    'variant': variant,
                    'scenario': scenario,
                    'time_period': time_period,
                    'basin': basin,
                    'draw': draw
                })
            
            print(f"      Task {task_id}: draws {batch_draws} ({len(batch_draws)} draws)")
            task_id += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Total status files processed: {len(status_files)}")
    print(f"  Combinations complete (skipped): {combinations_complete}")
    print(f"  Combinations with work: {combinations_with_work}")
    print(f"  Total complete draws: {total_complete}")
    print(f"  Total incomplete draws: {total_incomplete}")
    print(f"  Total tasks created: {task_id - 1}")
    print(f"  Total CSV rows (one per draw): {len(assignments)}")
    print(f"{'='*80}")
    
    if not assignments:
        print(f"\n✅ All draws are complete - no task assignments needed!")
        sys.exit(0)
    
    # Create DataFrame and save
    df = pd.DataFrame(assignments)
    output_file = climada_base / "level_4_task_assignments.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\nTask assignments saved to:")
    print(f"  {output_file}")
    print(f"\nFormat: One row per draw (task_id groups draws into batches)")
    print(f"Columns: task_id, model, variant, scenario, time_period, basin, draw")
    print(f"\nSample rows:")
    print(df.head(30).to_string(index=False))
    if len(df) > 35:
        print(f"...")
        print(df.tail(5).to_string(index=False))
    print(f"\n{'='*80}\n")
    
    # Return path and number of unique tasks (not total rows)
    num_tasks = task_id - 1
    return output_file, num_tasks


def create_full_run_assignments(data_source, draws_per_batch):
    """
    Create uniform task assignments for ALL draws (ignores status files).
    
    Creates simple batches: Task 1 runs draws 0-24, Task 2 runs 25-49, etc.
    for all basins defined in rfc.BASINS.
    
    Args:
        data_source: Data source (e.g., 'cmip6')
        draws_per_batch: Number of draws per task
    
    Returns:
        tuple: (assignments_file_path, total_tasks_needed)
    """
    climada_base = rfc.CLIMADA_INPUT_PATH / data_source
    
    print(f"\n{'='*80}")
    print(f"Creating FULL RUN task assignments (all draws)")
    print(f"  Data source: {data_source}")
    print(f"  Basins: {', '.join(rfc.BASINS)}")
    print(f"  Total draws: {rfc.NUM_DRAWS}")
    print(f"  Draws per batch: {draws_per_batch}")
    print(f"{'='*80}")
    
    # Calculate number of batches needed
    num_batches = (rfc.NUM_DRAWS + draws_per_batch - 1) // draws_per_batch
    print(f"\nCreating {num_batches} batches for {len(rfc.BASINS)} basins")
    
    # Create assignments
    assignments = []
    task_id = 1
    
    for basin in rfc.BASINS:
        for batch_idx in range(num_batches):
            batch_start = batch_idx * draws_per_batch
            batch_end = min(batch_start + draws_per_batch - 1, rfc.NUM_DRAWS - 1)
            
            # Add all draws in this batch
            for draw in range(batch_start, batch_end + 1):
                assignments.append({
                    'task_id': task_id,
                    'basin': basin,
                    'draw': draw
                })
            
            print(f"  Task {task_id}: {basin} draws {batch_start}-{batch_end} ({batch_end - batch_start + 1} draws)")
            task_id += 1
    
    # Create DataFrame and save
    df = pd.DataFrame(assignments)
    climada_base.mkdir(parents=True, exist_ok=True)
    output_file = climada_base / "level_4_task_assignments.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"Task assignments saved to:")
    print(f"  {output_file}")
    print(f"Total tasks: {df['task_id'].nunique()}")
    print(f"Total draws assigned: {len(df)}")
    print(f"{'='*80}\n")
    
    return output_file, df['task_id'].nunique()


def main():
    parser = argparse.ArgumentParser(
        description='Create task assignments from draw status files or for full run',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smart mode (only incomplete draws)
  python 00_create_level_4_task_assignments.py --data_source cmip6 --draws_per_batch 25
  
  # Full run mode (all draws uniformly)
  python 00_create_level_4_task_assignments.py --data_source cmip6 --draws_per_batch 25 --full_run
        """
    )
    parser.add_argument('--data_source', required=True, help='Data source (e.g., cmip6)')
    parser.add_argument('--draws_per_batch', type=int, default=25, help='Target draws per task (default: 25)')
    parser.add_argument('--max_period_duration', type=str, default=None, 
                       help='Maximum time period duration in years (None = use original bins, must match orchestrator)')
    parser.add_argument('--full_run', action='store_true', 
                       help='Create uniform assignments for ALL draws (ignore status files)')
    
    args = parser.parse_args()
    
    # Convert max_period_duration: "None" string -> None, "5" -> 5
    if args.max_period_duration is None or args.max_period_duration == "None":
        max_period_duration = None
    else:
        max_period_duration = int(args.max_period_duration)
    
    try:
        if args.full_run:
            assignments_file, num_tasks = create_full_run_assignments(
                args.data_source, args.draws_per_batch
            )
        else:
            assignments_file, num_tasks = create_level_4_task_assignments_from_status_files(
                args.data_source, args.draws_per_batch, max_period_duration
            )
        
        print(f"✅ Successfully created task assignments")
        print(f"   File: {assignments_file}")
        print(f"   Tasks: {num_tasks}")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to create task assignments")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

