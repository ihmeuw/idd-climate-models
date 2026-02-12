"""
Cleanup script to remove old time-period folders that don't match current time_bins.csv.

This script:
1. Loads current time bins (respecting MAX_PERIOD_DURATION setting)
2. Scans filesystem for existing time-period folders
3. Identifies folders that don't match any valid time period
4. Deletes them (with confirmation or dry-run mode)
"""

import argparse
import pandas as pd
import shutil
import time
import subprocess
from pathlib import Path
from collections import defaultdict
import idd_climate_models.constants as rfc
from idd_climate_models.time_period_functions import get_time_bins_path


def get_valid_time_periods(max_period_duration=None):
    """
    Load valid time periods from time_bins.csv.
    
    Returns:
        dict: {(model, variant, scenario): set of valid time_period strings}
    """
    time_bins_path = get_time_bins_path(max_period_duration)
    time_bins_df = pd.read_csv(time_bins_path)
    
    # Filter to BayesPoisson method only (if needed)
    if max_period_duration is None:
        time_bins_df = time_bins_df[time_bins_df['method'] == 'BayesPoisson']
    
    # Create time_period strings
    time_bins_df['time_period'] = time_bins_df['start_year'].astype(str) + '-' + time_bins_df['end_year'].astype(str)
    
    # Build lookup: (model, variant, scenario) -> set of valid time periods
    valid_periods = defaultdict(set)
    for _, row in time_bins_df.iterrows():
        key = (row['model'], row['variant'], row['scenario'])
        valid_periods[key].add(row['time_period'])
    
    return dict(valid_periods)


def find_old_folders(data_source, valid_time_periods, verbose=False):
    """
    Scan filesystem for time-period folders that don't match valid time periods.
    
    Returns:
        dict: Categorized folders to delete
            {
                'tc_risk_input': [list of paths],
                'tc_risk_output': [list of paths],
                'climada_input': [list of paths]
            }
    """
    folders_to_delete = {
        'tc_risk_input': [],
        'tc_risk_output': [],
        'climada_input': []
    }
    
    base_paths = {
        'tc_risk_input': rfc.TC_RISK_INPUT_PATH / data_source,
        'tc_risk_output': rfc.TC_RISK_OUTPUT_PATH / data_source,
        'climada_input': rfc.CLIMADA_INPUT_PATH / data_source
    }
    
    for folder_type, base_path in base_paths.items():
        if not base_path.exists():
            print(f"  Skipping {folder_type}: {base_path} does not exist")
            continue
        
        # Walk through model/variant/scenario structure
        for model_path in base_path.iterdir():
            if not model_path.is_dir():
                continue
            model = model_path.name
            
            for variant_path in model_path.iterdir():
                if not variant_path.is_dir():
                    continue
                variant = variant_path.name
                
                for scenario_path in variant_path.iterdir():
                    if not scenario_path.is_dir():
                        continue
                    scenario = scenario_path.name
                    
                    # Get valid time periods for this combination
                    key = (model, variant, scenario)
                    valid_periods_set = valid_time_periods.get(key, set())
                    
                    # Check what time-period folders exist
                    for time_period_path in scenario_path.iterdir():
                        if not time_period_path.is_dir():
                            continue
                        
                        time_period = time_period_path.name
                        
                        # Skip folders already queued for deletion
                        if '_DELETE_' in time_period:
                            # if verbose:
                            #     print(f"  Skipping (already queued for deletion): {model}/{variant}/{scenario}/{time_period}")
                            continue
                        
                        # Check if this time period is valid
                        if time_period not in valid_periods_set:
                            folders_to_delete[folder_type].append(time_period_path)
                            if verbose:
                                print(f"  Found invalid: {model}/{variant}/{scenario}/{time_period}")
    
    return folders_to_delete


def delete_folders(folders_to_delete, dry_run=False):
    """
    Delete identified folders (or just print if dry_run=True).
    Uses async deletion for CLIMADA folders (many files) and regular deletion for TC-risk folders.
    """
    total_deleted = 0
    total_failed = 0
    
    for folder_type, paths in folders_to_delete.items():
        if not paths:
            continue
        
        print(f"\n{folder_type.upper().replace('_', ' ')}:")
        print(f"  Found {len(paths)} folders to delete")
        
        # Use async deletion for CLIMADA (many files), regular for TC-risk
        use_async = (folder_type == 'climada_input')
        
        for path in paths:
            # Extract relevant path info for display
            parts = path.parts
            # Find index of data_source and show from there
            try:
                ds_idx = parts.index('cmip6')  # or data_source
                display_path = '/'.join(parts[ds_idx:])
            except ValueError:
                display_path = str(path)
            
            if dry_run:
                print(f"    [DRY RUN] Would delete: {display_path}")
            else:
                try:
                    if use_async:
                        # Async deletion: rename + background ionice rm
                        suffix = f'_DELETE_{int(time.time() * 1000000)}'
                        renamed_path = path.parent / (path.name + suffix)
                        path.rename(renamed_path)
                        
                        # Launch async deletion in background
                        subprocess.Popen(
                            ['ionice', '-c', '3', 'rm', '-rf', str(renamed_path)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        print(f"    ✓ Queued for async deletion: {display_path}")
                    else:
                        # Regular synchronous deletion for TC-risk folders (typically smaller)
                        shutil.rmtree(path)
                        print(f"    ✓ Deleted: {display_path}")
                    
                    total_deleted += 1
                except Exception as e:
                    print(f"    ✗ Failed to delete {display_path}: {e}")
                    total_failed += 1
    
    return total_deleted, total_failed


def main():
    parser = argparse.ArgumentParser(
        description='Cleanup old time-period folders that don\'t match current time_bins.csv'
    )
    parser.add_argument('--data_source', type=str, default='cmip6',
                        help='Data source (default: cmip6)')
    parser.add_argument('--max_period_duration', type=int, default=5,
                        help='Maximum time period duration in years (default: 5). Use 0 for original bins.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be deleted without actually deleting')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')
    
    args = parser.parse_args()
    
    # Convert max_period_duration
    max_period = None if args.max_period_duration == 0 else args.max_period_duration
    
    print("=" * 80)
    print("CLEANUP: Old Time-Period Folders")
    print("=" * 80)
    print(f"Data source: {args.data_source}")
    print(f"Max period duration: {max_period if max_period else 'original bins'}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'DELETE'}")
    print()
    
    # Step 1: Load valid time periods
    print("Step 1: Loading valid time periods from time_bins.csv...")
    valid_time_periods = get_valid_time_periods(max_period)
    print(f"  Found {len(valid_time_periods)} (model, variant, scenario) combinations")
    total_periods = sum(len(periods) for periods in valid_time_periods.values())
    print(f"  Total valid time periods: {total_periods}")
    
    # Step 2: Scan filesystem
    print("\nStep 2: Scanning filesystem for invalid time-period folders...")
    folders_to_delete = find_old_folders(args.data_source, valid_time_periods, args.verbose)
    
    total_folders = sum(len(paths) for paths in folders_to_delete.values())
    
    if total_folders == 0:
        print("\n✓ No invalid time-period folders found!")
        return
    
    print(f"\nFound {total_folders} invalid time-period folders:")
    for folder_type, paths in folders_to_delete.items():
        if paths:
            print(f"  {folder_type}: {len(paths)}")
    
    # Step 3: Delete (or dry run)
    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN: The following folders would be deleted:")
        print("=" * 80)
        total_deleted, total_failed = delete_folders(folders_to_delete, dry_run=True)
        print("\n" + "=" * 80)
        print("To actually delete these folders, re-run without --dry_run")
    else:
        print("\n" + "=" * 80)
        print("WARNING: About to delete folders! Press Ctrl+C to cancel...")
        print("=" * 80)
        input("\nPress Enter to continue...")
        
        total_deleted, total_failed = delete_folders(folders_to_delete, dry_run=False)
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Successfully deleted: {total_deleted}")
        if total_failed > 0:
            print(f"Failed to delete: {total_failed}")
        
        # Note about async deletion
        climada_count = len(folders_to_delete.get('climada_input', []))
        if climada_count > 0:
            print(f"\nNote: {climada_count} CLIMADA folders were renamed and queued for")
            print("async background deletion (using ionice to avoid I/O impact).")
            print("Deletion will continue after this script exits.")
        print()


if __name__ == "__main__":
    main()
