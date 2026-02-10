"""
Validation script to monitor TC-risk pipeline progress and detect race conditions.

Run this periodically during your pipeline execution:
  python validate_pipeline_progress.py --model MRI-ESM2-0 --variant r1i1p1f1 --scenario historical --time_period 1970-1999

Or check all combinations:
  python validate_pipeline_progress.py --check_all
"""

import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
import idd_climate_models.constants as rfc

def get_draw_files_for_basin(data_source, model, variant, scenario, time_period, basin):
    """
    Parse all draw files in a basin folder and return detailed info.
    
    Returns:
        dict: {draw_num: file_path}
    """
    output_path = rfc.TC_RISK_OUTPUT_PATH / data_source / model / variant / scenario / time_period / basin
    
    if not output_path.exists():
        return {}
    
    draw_files = {}
    time_parts = time_period.split('-')
    time_start_str = f'{int(time_parts[0]):04d}01'
    time_end_str = f'{int(time_parts[1]):04d}12'
    base_pattern = f'tracks_{basin}_{model}_{scenario}_{variant}_{time_start_str}_{time_end_str}'
    
    # Check draw 0 (no suffix)
    draw_0_file = output_path / f'{base_pattern}.nc'
    if draw_0_file.exists():
        draw_files[0] = draw_0_file
    
    # Check draws 1-249
    for nc_file in output_path.glob(f"{base_pattern}_e*.nc"):
        try:
            filename = nc_file.stem
            draw_suffix = filename.split('_e')[-1]
            draw_idx_in_suffix = int(draw_suffix)
            draw_num = draw_idx_in_suffix + 1
            draw_files[draw_num] = nc_file
        except (ValueError, IndexError):
            continue
    
    return draw_files


def get_zarr_files_for_basin(data_source, model, variant, scenario, time_period, basin):
    """
    Get all zarr directories (post-processed outputs) for a basin.
    
    Returns:
        dict: {draw_num: zarr_path}
    """
    climada_path = rfc.CLIMADA_INPUT_PATH / data_source / model / variant / scenario / time_period / basin
    
    if not climada_path.exists():
        return {}
    
    zarr_files = {}
    
    # Pattern: storm_list_GL_0.zarr, storm_list_GL_1.zarr, etc.
    for zarr_dir in climada_path.glob(f"storm_list_{basin}_*.zarr"):
        try:
            # Extract draw number from filename
            draw_str = zarr_dir.stem.split('_')[-1]
            draw_num = int(draw_str)
            zarr_files[draw_num] = zarr_dir
        except (ValueError, IndexError):
            continue
    
    return zarr_files


def check_for_duplicates(draw_files):
    """Check if any draw numbers appear twice (race condition indicator)."""
    draw_counts = defaultdict(int)
    for draw_num in draw_files.keys():
        draw_counts[draw_num] += 1
    
    duplicates = {draw: count for draw, count in draw_counts.items() if count > 1}
    return duplicates


def check_basin_status(data_source, model, variant, scenario, time_period, basin, 
                       expected_draws=250, draws_per_batch=10):
    """
    Comprehensive check of basin processing status.
    
    Returns:
        dict: Status information
    """
    draw_files = get_draw_files_for_basin(data_source, model, variant, scenario, time_period, basin)
    zarr_files = get_zarr_files_for_basin(data_source, model, variant, scenario, time_period, basin)
    
    # Check for gaps in draw sequence
    present_draws = sorted(draw_files.keys())
    expected_draws_set = set(range(expected_draws))
    missing_draws = sorted(expected_draws_set - set(present_draws))
    
    # Check for duplicates (RACE CONDITION)
    duplicates = check_for_duplicates(draw_files)
    
    # Check zarr post-processing
    zarr_draws = sorted(zarr_files.keys())
    missing_zarr = sorted(set(present_draws) - set(zarr_draws))
    
    # Identify which batches are complete/incomplete
    num_batches = (expected_draws + draws_per_batch - 1) // draws_per_batch
    batch_status = {}
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * draws_per_batch
        batch_end = min(batch_start + draws_per_batch - 1, expected_draws - 1)
        batch_draws = set(range(batch_start, batch_end + 1))
        
        present_in_batch = batch_draws & set(present_draws)
        complete = len(present_in_batch) == len(batch_draws)
        
        batch_status[batch_idx] = {
            'range': f"{batch_start}-{batch_end}",
            'complete': complete,
            'present': len(present_in_batch),
            'expected': len(batch_draws),
            'missing': sorted(batch_draws - present_in_batch)
        }
    
    return {
        'basin': basin,
        'total_draws': len(present_draws),
        'expected_draws': expected_draws,
        'missing_draws': missing_draws,
        'duplicates': duplicates,
        'zarr_count': len(zarr_draws),
        'missing_zarr': missing_zarr,
        'batch_status': batch_status,
        'complete': len(present_draws) == expected_draws and len(missing_zarr) == 0
    }


def print_basin_status(status):
    """Pretty print basin status."""
    print(f"\n{'='*80}")
    print(f"Basin: {status['basin']}")
    print(f"{'='*80}")
    
    # Overall status
    progress_pct = (status['total_draws'] / status['expected_draws']) * 100
    print(f"Progress: {status['total_draws']}/{status['expected_draws']} draws ({progress_pct:.1f}%)")
    
    # RACE CONDITION CHECK
    if status['duplicates']:
        print(f"\n⚠️  WARNING: DUPLICATE DRAWS DETECTED (RACE CONDITION):")
        for draw, count in status['duplicates'].items():
            print(f"  Draw {draw}: appears {count} times")
    else:
        print(f"✅ No duplicate draws detected")
    
    # Missing draws
    if status['missing_draws']:
        print(f"\nMissing draws: {len(status['missing_draws'])}")
        if len(status['missing_draws']) <= 20:
            print(f"  {status['missing_draws']}")
        else:
            print(f"  First 10: {status['missing_draws'][:10]}")
            print(f"  Last 10: {status['missing_draws'][-10:]}")
    
    # Zarr post-processing status
    zarr_pct = (status['zarr_count'] / status['total_draws'] * 100) if status['total_draws'] > 0 else 0
    print(f"\nPost-processing (Zarr): {status['zarr_count']}/{status['total_draws']} ({zarr_pct:.1f}%)")
    if status['missing_zarr']:
        print(f"  Missing zarr for {len(status['missing_zarr'])} draws")
        if len(status['missing_zarr']) <= 10:
            print(f"  {status['missing_zarr']}")
    
    # Batch status summary
    complete_batches = sum(1 for b in status['batch_status'].values() if b['complete'])
    total_batches = len(status['batch_status'])
    print(f"\nBatch completion: {complete_batches}/{total_batches} batches")
    
    # Show incomplete batches
    incomplete = {k: v for k, v in status['batch_status'].items() if not v['complete']}
    if incomplete:
        print(f"\nIncomplete batches:")
        for batch_idx, batch_info in sorted(incomplete.items()):
            print(f"  Batch {batch_idx} ({batch_info['range']}): "
                  f"{batch_info['present']}/{batch_info['expected']} draws")
            if batch_info['missing'] and len(batch_info['missing']) <= 5:
                print(f"    Missing: {batch_info['missing']}")
    
    if status['complete']:
        print(f"\n✅ Basin {status['basin']} is COMPLETE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', default='cmip6')
    parser.add_argument('--model', help='Model name')
    parser.add_argument('--variant', help='Variant name')
    parser.add_argument('--scenario', help='Scenario name')
    parser.add_argument('--time_period', help='Time period (e.g., 1970-1999)')
    parser.add_argument('--basin', help='Specific basin (or leave blank for all)')
    parser.add_argument('--check_all', action='store_true', 
                       help='Check all combinations in time_bins.csv')
    parser.add_argument('--expected_draws', type=int, default=250)
    parser.add_argument('--draws_per_batch', type=int, default=10)
    parser.add_argument('--summary_only', action='store_true',
                       help='Show only summary, not per-basin details')
    
    args = parser.parse_args()
    
    BASINS = ['EP', 'NA', 'NI', 'SI', 'AU', 'SP', 'WP']
    
    if args.check_all:
        # Load time_bins and check all combinations
        time_bins_df = pd.read_csv(rfc.TIME_BINS_DF_PATH)
        time_bins_df = time_bins_df[time_bins_df['method'] == 'BayesPoisson']
        time_bins_df['time_period'] = (time_bins_df['start_year'].astype(str) + '-' + 
                                        time_bins_df['end_year'].astype(str))
        
        combinations = time_bins_df[['model', 'variant', 'scenario', 'time_period']].drop_duplicates()
        
        print(f"Checking {len(combinations)} time-period combinations...")
        
        all_statuses = []
        race_conditions = []
        
        for _, row in combinations.iterrows():
            for basin in BASINS:
                status = check_basin_status(
                    args.data_source, row['model'], row['variant'], 
                    row['scenario'], row['time_period'], basin,
                    args.expected_draws, args.draws_per_batch
                )
                all_statuses.append(status)
                
                if status['duplicates']:
                    race_conditions.append(status)
                
                if not args.summary_only:
                    print_basin_status(status)
        
        # Summary
        print(f"\n{'='*80}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*80}")
        print(f"Total basin combinations checked: {len(all_statuses)}")
        complete = sum(1 for s in all_statuses if s['complete'])
        print(f"Complete: {complete}/{len(all_statuses)}")
        
        if race_conditions:
            print(f"\n⚠️  RACE CONDITIONS DETECTED: {len(race_conditions)} basins")
            for status in race_conditions:
                print(f"  {status['basin']}: {status['duplicates']}")
        else:
            print(f"\n✅ No race conditions detected")
        
    else:
        # Check single combination
        if not all([args.model, args.variant, args.scenario, args.time_period]):
            print("Error: Must specify --model, --variant, --scenario, and --time_period")
            print("Or use --check_all to check everything")
            return
        
        basins_to_check = [args.basin] if args.basin else BASINS
        
        for basin in basins_to_check:
            status = check_basin_status(
                args.data_source, args.model, args.variant,
                args.scenario, args.time_period, basin,
                args.expected_draws, args.draws_per_batch
            )
            print_basin_status(status)


if __name__ == '__main__':
    main()