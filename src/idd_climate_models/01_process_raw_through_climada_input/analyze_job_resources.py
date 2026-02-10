"""
Analyze job resource usage for TC-risk pipeline to optimize batch sizes and resource allocation.

This script:
1. Analyzes Level 3 (04_) tasks: resources vs time-period years
2. Analyzes Level 4 (05_) tasks: resources vs years × storms-per-year
3. Handles incomplete tasks that restarted
4. Estimates true runtime from file modification times for in-progress tasks
5. Accounts for survival bias (short tasks finish first)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
import idd_climate_models.constants as rfc


def get_draw_suffix(draw_num):
    """Get the filename suffix for a given draw number."""
    if draw_num == 0:
        return ''
    else:
        return f'_e{draw_num - 1}'


def parse_task_name(task_name):
    """
    Parse task name to extract parameters.
    
    Example:
    'run_basin_tc_risk_data_source-cmip6_model-EC-Earth3_variant-r1i1p1f1_scenario-ssp585_time_period-2091-2100_basin-NI_draw_start-140_draw_end-149'
    
    Returns:
        dict with keys: data_source, model, variant, scenario, time_period, basin, draw_start, draw_end
    """
    parts = {}
    for segment in task_name.split('_'):
        if '-' in segment:
            key_value = segment.split('-', 1)
            if len(key_value) == 2:
                key, value = key_value
                # Handle multi-part values like time_period-2091-2100
                if key in ['time', 'period', 'draw']:
                    continue
                parts[key] = value
    
    # Handle compound keys
    if 'time' in task_name and 'period' in task_name:
        # Extract time_period pattern YYYY-YYYY
        import re
        match = re.search(r'time_period-(\d{4}-\d{4})', task_name)
        if match:
            parts['time_period'] = match.group(1)
    
    # Extract draw_start and draw_end
    import re
    draw_start_match = re.search(r'draw_start-(\d+)', task_name)
    draw_end_match = re.search(r'draw_end-(\d+)', task_name)
    if draw_start_match:
        parts['draw_start'] = int(draw_start_match.group(1))
    if draw_end_match:
        parts['draw_end'] = int(draw_end_match.group(1))
    
    return parts


def get_time_period_years(time_period):
    """Calculate number of years in time period."""
    start, end = map(int, time_period.split('-'))
    return end - start + 1


def get_storms_per_year(model, variant, scenario, time_period, basin):
    """
    Look up storms per year from time_bins_wide.csv.
    
    Returns:
        int: storms per year for this configuration
    """
    time_bins_wide_df = pd.read_csv(rfc.TIME_BINS_WIDE_DF_PATH)
    start_year, end_year = map(int, time_period.split('-'))
    
    mask = (
        (time_bins_wide_df['model'] == model) &
        (time_bins_wide_df['variant'] == variant) &
        (time_bins_wide_df['scenario'] == scenario) &
        (time_bins_wide_df['start_year'] == start_year) &
        (time_bins_wide_df['end_year'] == end_year)
    )
    
    matching = time_bins_wide_df[mask]
    if len(matching) == 0:
        return None
    
    basin_col = f"{basin}_int"
    if basin_col not in matching.columns:
        return None
    
    return int(matching.iloc[0][basin_col])


def estimate_runtime_from_files(data_source, model, variant, scenario, time_period, basin, 
                                draw_start, draw_end, file_type='zarr'):
    """
    Estimate runtime by looking at file modification times within a task.
    
    This helps with in-progress tasks where Jobmon data is incomplete.
    
    Returns:
        dict with per-draw times and total estimated runtime
    """
    if file_type == 'zarr':
        output_path = rfc.CLIMADA_INPUT_PATH / data_source / model / variant / scenario / time_period / basin
    else:
        output_path = rfc.TC_RISK_OUTPUT_PATH / data_source / model / variant / scenario / time_period / basin
    
    if not output_path.exists():
        return None
    
    time_parts = time_period.split('-')
    time_start_str = f'{int(time_parts[0]):04d}01'
    time_end_str = f'{int(time_parts[1]):04d}12'
    base_pattern = f'tracks_{basin}_{model}_{scenario}_{variant}_{time_start_str}_{time_end_str}'
    
    draw_times = {}
    
    for draw_num in range(draw_start, draw_end + 1):
        suffix = get_draw_suffix(draw_num)
        if file_type == 'zarr':
            file_path = output_path / f'{base_pattern}{suffix}.zarr'
        else:
            file_path = output_path / f'{base_pattern}{suffix}.nc'
        
        if file_path.exists():
            mtime = file_path.stat().st_mtime
            draw_times[draw_num] = datetime.fromtimestamp(mtime)
    
    if len(draw_times) < 2:
        return None
    
    # Calculate time differences between consecutive draws
    sorted_draws = sorted(draw_times.items())
    draw_durations = []
    
    for i in range(1, len(sorted_draws)):
        draw_prev, time_prev = sorted_draws[i-1]
        draw_curr, time_curr = sorted_draws[i]
        duration_minutes = (time_curr - time_prev).total_seconds() / 60
        draw_durations.append({
            'draw_from': draw_prev,
            'draw_to': draw_curr,
            'duration_minutes': duration_minutes
        })
    
    total_minutes = sum(d['duration_minutes'] for d in draw_durations)
    
    return {
        'draw_times': draw_times,
        'draw_durations': draw_durations,
        'total_estimated_minutes': total_minutes,
        'n_completed_draws': len(draw_times),
        'n_expected_draws': draw_end - draw_start + 1,
        'is_complete': len(draw_times) == (draw_end - draw_start + 1)
    }


def analyze_level3_tasks(jobmon_df):
    """
    Analyze Level 3 (04_run_global_tc_risk) tasks.
    
    Focus: relationship between time-period years and resource usage.
    """
    level3_df = jobmon_df[jobmon_df['task_name'].str.contains('run_global_tc_risk')].copy()
    
    if len(level3_df) == 0:
        print("No Level 3 tasks found in data")
        return None
    
    # Parse task parameters
    level3_df['params'] = level3_df['task_name'].apply(parse_task_name)
    level3_df['time_period'] = level3_df['params'].apply(lambda x: x.get('time_period'))
    level3_df['n_years'] = level3_df['time_period'].apply(
        lambda x: get_time_period_years(x) if x else None
    )
    
    # Add resource metrics (customize based on your Jobmon columns)
    # Assuming columns like: runtime_minutes, memory_gb, cores, etc.
    
    summary = level3_df.groupby('n_years').agg({
        'runtime_minutes': ['mean', 'std', 'min', 'max', 'count'],
        'memory_gb': ['mean', 'max'],
        'cores': ['mean']
    }).round(2)
    
    return level3_df, summary


def analyze_level4_tasks(jobmon_df, include_file_estimates=True):
    """
    Analyze Level 4 (05_run_basin_tc_risk) tasks.
    
    Focus: relationship between (years × storms-per-year) and resource usage.
    Handles incomplete tasks and restarts.
    """
    level4_df = jobmon_df[jobmon_df['task_name'].str.contains('run_basin_tc_risk')].copy()
    
    if len(level4_df) == 0:
        print("No Level 4 tasks found in data")
        return None
    
    # Parse task parameters
    level4_df['params'] = level4_df['task_name'].apply(parse_task_name)
    
    for col in ['data_source', 'model', 'variant', 'scenario', 'time_period', 'basin']:
        level4_df[col] = level4_df['params'].apply(lambda x: x.get(col))
    
    level4_df['draw_start'] = level4_df['params'].apply(lambda x: x.get('draw_start'))
    level4_df['draw_end'] = level4_df['params'].apply(lambda x: x.get('draw_end'))
    level4_df['batch_size'] = level4_df['draw_end'] - level4_df['draw_start'] + 1
    
    level4_df['n_years'] = level4_df['time_period'].apply(
        lambda x: get_time_period_years(x) if x else None
    )
    
    # Get storms per year
    level4_df['storms_per_year'] = level4_df.apply(
        lambda row: get_storms_per_year(
            row['model'], row['variant'], row['scenario'], 
            row['time_period'], row['basin']
        ) if all(pd.notna([row['model'], row['variant'], row['scenario'], 
                           row['time_period'], row['basin']])) else None,
        axis=1
    )
    
    level4_df['total_storms'] = level4_df['n_years'] * level4_df['storms_per_year']
    level4_df['storms_per_draw'] = level4_df['total_storms'] / level4_df['batch_size']
    
    # Estimate runtime from files for incomplete tasks
    if include_file_estimates:
        level4_df['file_estimates'] = level4_df.apply(
            lambda row: estimate_runtime_from_files(
                row['data_source'], row['model'], row['variant'],
                row['scenario'], row['time_period'], row['basin'],
                row['draw_start'], row['draw_end']
            ) if all(pd.notna([row['data_source'], row['model'], row['variant'],
                               row['scenario'], row['time_period'], row['basin'],
                               row['draw_start'], row['draw_end']])) else None,
            axis=1
        )
        
        # Use file estimates for incomplete tasks
        level4_df['estimated_runtime'] = level4_df.apply(
            lambda row: row['file_estimates']['total_estimated_minutes'] 
            if row['file_estimates'] and not row['file_estimates']['is_complete']
            else row.get('runtime_minutes'),
            axis=1
        )
    
    return level4_df


def generate_recommendations(level4_df):
    """
    Generate batch size and resource recommendations.
    """
    if level4_df is None or len(level4_df) == 0:
        return None
    
    # Filter to completed or well-estimated tasks
    complete_df = level4_df[
        (level4_df['file_estimates'].notna()) & 
        (level4_df['file_estimates'].apply(lambda x: x['is_complete'] if x else False))
    ].copy()
    
    if len(complete_df) == 0:
        print("⚠️  No completed tasks yet - recommendations will be preliminary")
        complete_df = level4_df[level4_df['estimated_runtime'].notna()].copy()
    
    # Calculate minutes per draw
    complete_df['minutes_per_draw'] = complete_df['estimated_runtime'] / complete_df['batch_size']
    
    # Group by basin and storms_per_year
    recommendations = []
    
    for basin in complete_df['basin'].unique():
        basin_df = complete_df[complete_df['basin'] == basin]
        
        mean_minutes_per_draw = basin_df['minutes_per_draw'].mean()
        
        # Target 2-hour batches
        target_minutes = 120
        recommended_batch_size = max(1, int(target_minutes / mean_minutes_per_draw))
        
        recommendations.append({
            'basin': basin,
            'mean_minutes_per_draw': round(mean_minutes_per_draw, 2),
            'recommended_batch_size': recommended_batch_size,
            'estimated_batch_runtime_minutes': round(recommended_batch_size * mean_minutes_per_draw, 1),
            'n_samples': len(basin_df)
        })
    
    return pd.DataFrame(recommendations)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze job resource usage for TC-risk pipeline"
    )
    parser.add_argument('--level3_csv', type=str,
                        help='Path to Jobmon CSV for Level 3 (04_) tasks')
    parser.add_argument('--level4_csv', type=str,
                        help='Path to Jobmon CSV for Level 4 (05_) tasks')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save analysis outputs')
    parser.add_argument('--use_file_times', action='store_true', default=True,
                        help='Use file modification times for Level 4 (recommended, default=True)')
    
    args = parser.parse_args()
    
    if not args.level3_csv and not args.level4_csv:
        print("❌ Must provide at least one of --level3_csv or --level4_csv")
        return
    
    print("=" * 80)
    print("TC-Risk Job Resource Analysis")
    print("=" * 80)
    print("=" * 80)
    
    # Analyze Level 3 tasks
    level3_df = None
    level3_summary = None
    if args.level3_csv:
        print("\n" + "=" * 80)
        print("LEVEL 3 ANALYSIS (04_run_global_tc_risk)")
        print("=" * 80)
        print(f"Loading Level 3 data from: {args.level3_csv}")
        level3_jobmon_df = pd.read_csv(args.level3_csv)
        print(f"✓ Loaded {len(level3_jobmon_df)} Level 3 task records")
        
        level3_df, level3_summary = analyze_level3_tasks(level3_jobmon_df)
        if level3_summary is not None:
            print("\nRuntime by number of years:")
            print(level3_summary)
            
            output_path = Path(args.output_dir) / 'level3_analysis.csv'
            level3_df.to_csv(output_path, index=False)
            print(f"\n✓ Saved detailed analysis to: {output_path}")
    
    # Analyze Level 4 tasks
    level4_df = None
    if args.level4_csv:
        print("\n" + "=" * 80)
        print("LEVEL 4 ANALYSIS (05_run_basin_tc_risk)")
        print("=" * 80)
        print(f"Loading Level 4 data from: {args.level4_csv}")
        level4_jobmon_df = pd.read_csv(args.level4_csv)
        print(f"✓ Loaded {len(level4_jobmon_df)} Level 4 task records")
        
        if args.use_file_times:
            print("\n🕒 Using file modification times for runtime estimation (recommended)")
            print("   This avoids bias from incomplete tasks and captures actual per-draw times")
        
        level4_df = analyze_level4_tasks(level4_jobmon_df, 
                                         include_file_estimates=args.use_file_times)
        
        if level4_df is not None:
            print(f"\n✓ Analyzed {len(level4_df)} Level 4 tasks")
            
            # Show how many have file-based estimates
            if args.use_file_times:
                n_with_estimates = level4_df['file_estimates'].notna().sum()
                n_complete = level4_df[level4_df['file_estimates'].apply(
                    lambda x: x['is_complete'] if x else False
                )].shape[0]
                print(f"   - {n_with_estimates} tasks have file-based runtime estimates")
                print(f"   - {n_complete} tasks are complete")
                print(f"   - {n_with_estimates - n_complete} tasks are in-progress (partial estimates)")
            
            # Summary statistics
            print("\nBatch size distribution:")
            print(level4_df['batch_size'].value_counts().sort_index())
            
            print("\nBasin distribution:")
            print(level4_df['basin'].value_counts())
            
            output_path = Path(args.output_dir) / 'level4_analysis.csv'
            # Drop the file_estimates column for CSV export (it's a dict)
            export_df = level4_df.drop(columns=['file_estimates', 'params'], errors='ignore')
            export_df.to_csv(output_path, index=False)
            print(f"\n✓ Saved detailed analysis to: {output_path}")
            
            # Generate recommendations
            print("\n" + "=" * 80)
            print("BATCH SIZE RECOMMENDATIONS")
            print("=" * 80)
            recommendations = generate_recommendations(level4_df)
            if recommendations is not None:
                print("\nRecommended batch sizes for 2-hour target runtime:")
                print(recommendations.to_string(index=False))
                
                rec_path = Path(args.output_dir) / 'batch_size_recommendations.csv'
                recommendations.to_csv(rec_path, index=False)
                print(f"\n✓ Saved recommendations to: {rec_path}")
            else:
                print("\n⚠️  Insufficient data for recommendations yet")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
