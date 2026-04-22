"""
Run the storm-admin0 analysis for a specific model/variant/scenario combination.

This script is executed as a Level 2 task after all Level 1 draw-processing tasks complete.
Instead of using papermill, it directly executes the analysis logic.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import idd_climate_models.constants as rfc

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def main():
    parser = argparse.ArgumentParser(description="Run storm-admin0 analysis")
    parser.add_argument("--model", required=True, help="Climate model name")
    parser.add_argument("--variant", required=True, help="Model variant")
    parser.add_argument("--scenario", required=True, help="Emissions scenario")
    parser.add_argument("--data_source", default="cmip6", help="Data source")
    
    args = parser.parse_args()
    
    print(f"Running analysis for {args.model}/{args.variant}/{args.scenario}")
    
    # ========================================================================
    # 1. Construct File Paths and Filter Data
    # ========================================================================
    
    base_path = rfc.CLIMADA_INPUT_PATH / args.data_source
    task_assignments_file = base_path / "level_4_task_assignments.csv"
    df_assignments = pd.read_csv(task_assignments_file, keep_default_na=False)
    
    df_filtered = df_assignments[
        (df_assignments['model'] == args.model) &
        (df_assignments['variant'] == args.variant) &
        (df_assignments['scenario'] == args.scenario)
    ]
    
    task_locations = df_filtered.groupby('task_id').agg({
        'model': 'first',
        'variant': 'first', 
        'scenario': 'first',
        'time_period': 'first',
        'basin': 'first'
    }).reset_index()
    
    csv_files = []
    for _, row in task_locations.iterrows():
        task_id = row['task_id']
        file_path = (base_path / row['model'] / row['variant'] / row['scenario'] / 
                     row['time_period'] / row['basin'] / 
                     f"storm_admin_impacts_task_{task_id:04d}.csv")
        if file_path.exists():
            csv_files.append(file_path)
    
    print(f"  Found {len(csv_files)} CSV files to process")
    
    if not csv_files:
        print(f"❌ No data files found for {args.model}/{args.variant}/{args.scenario}")
        sys.exit(1)
    
    # ========================================================================
    # 2. Load and Combine Data
    # ========================================================================
    
    dfs = [pd.read_csv(f, keep_default_na=False, na_values=['']) for f in csv_files]
    df_combined = pd.concat(dfs, ignore_index=True)
    
    df_combined['storm_id'] = (
        df_combined['model'].astype(str) + '_' +
        df_combined['variant'].astype(str) + '_' +
        df_combined['scenario'].astype(str) + '_' +
        df_combined['time_period'].astype(str) + '_' +
        df_combined['basin'].astype(str) + '_' +
        df_combined['draw'].astype(str) + '_' +
        df_combined['storm_track'].astype(str)
    )
    
    storm_max_category = df_combined.groupby('storm_id').agg({
        'model': 'first',
        'variant': 'first',
        'scenario': 'first',
        'time_period': 'first',
        'basin': 'first',
        'draw': 'first',
        'storm_track': 'first',
        'year': 'first',
        'month': 'first',
        'ADM0_CODE': 'first',
        'ADM0_NAME': 'first',
        'loc_id': 'first',
        'max_wind_speed': 'max',
    }).reset_index()
    
    storm_max_category['storm_category'] = storm_max_category['max_wind_speed'].apply(rfc.classify_storm)
    
    print(f"  Processed {len(storm_max_category)} unique storms")
    
    # ========================================================================
    # 3. Country-Level Storm Statistics by Year
    # ========================================================================
    
    country_annual_by_draw = storm_max_category.groupby(
        ['ADM0_CODE', 'ADM0_NAME', 'loc_id', 'year', 'draw']
    ).size().reset_index(name='total_storms')
    
    category_by_draw = storm_max_category.groupby(
        ['ADM0_CODE', 'ADM0_NAME', 'loc_id', 'year', 'draw', 'storm_category']
    ).size().reset_index(name='storm_count')
    
    category_pivot = category_by_draw.pivot_table(
        index=['ADM0_CODE', 'ADM0_NAME', 'loc_id', 'year', 'draw'],
        columns='storm_category',
        values='storm_count',
        fill_value=0
    ).reset_index()
    
    country_annual_by_draw = country_annual_by_draw.merge(
        category_pivot, 
        on=['ADM0_CODE', 'ADM0_NAME', 'loc_id', 'year', 'draw'], 
        how='left'
    )
    
    for cat in ['tropical_storm', 'category_1', 'category_2', 'category_3', 'category_4', 'category_5']:
        if cat not in country_annual_by_draw.columns:
            country_annual_by_draw[cat] = 0
    
    country_annual_by_draw['at_least_tropical_storm'] = country_annual_by_draw['total_storms']
    country_annual_by_draw['at_least_hurricane'] = (
        country_annual_by_draw['category_1'] + country_annual_by_draw['category_2'] + 
        country_annual_by_draw['category_3'] + country_annual_by_draw['category_4'] + 
        country_annual_by_draw['category_5']
    )
    country_annual_by_draw['hurricane_1_to_3'] = (
        country_annual_by_draw['category_1'] + country_annual_by_draw['category_2'] + 
        country_annual_by_draw['category_3']
    )
    country_annual_by_draw['hurricane_4_plus'] = (
        country_annual_by_draw['category_4'] + country_annual_by_draw['category_5']
    )
    
    def calculate_stats(df, group_cols, value_cols):
        stats = []
        for name, group in df.groupby(group_cols):
            row = dict(zip(group_cols, name if isinstance(name, tuple) else [name]))
            for col in value_cols:
                row[f'{col}_mean'] = group[col].mean()
                row[f'{col}_lower'] = group[col].quantile(0.025)
                row[f'{col}_upper'] = group[col].quantile(0.975)
            stats.append(row)
        return pd.DataFrame(stats)
    
    metrics = [
        'total_storms', 'tropical_storm', 'category_1', 'category_2', 
        'category_3', 'category_4', 'category_5', 'at_least_tropical_storm',
        'at_least_hurricane', 'hurricane_1_to_3', 'hurricane_4_plus'
    ]
    
    country_annual_stats = calculate_stats(
        country_annual_by_draw, 
        ['ADM0_CODE', 'ADM0_NAME', 'loc_id', 'year'], 
        metrics
    )
    
    print(f"  Calculated country-level statistics")
    
    # ========================================================================
    # 4. Annual Statistics Across All Countries
    # ========================================================================
    
    annual_by_draw = storm_max_category.groupby(['year', 'draw']).size().reset_index(name='total_storms')
    
    category_by_draw_all = storm_max_category.groupby(
        ['year', 'draw', 'storm_category']
    ).size().reset_index(name='storm_count')
    
    category_pivot_all = category_by_draw_all.pivot_table(
        index=['year', 'draw'],
        columns='storm_category',
        values='storm_count',
        fill_value=0
    ).reset_index()
    
    annual_by_draw = annual_by_draw.merge(category_pivot_all, on=['year', 'draw'], how='left')
    
    for cat in ['tropical_storm', 'category_1', 'category_2', 'category_3', 'category_4', 'category_5']:
        if cat not in annual_by_draw.columns:
            annual_by_draw[cat] = 0
    
    annual_by_draw['at_least_tropical_storm'] = annual_by_draw['total_storms']
    annual_by_draw['at_least_hurricane'] = (
        annual_by_draw['category_1'] + annual_by_draw['category_2'] + 
        annual_by_draw['category_3'] + annual_by_draw['category_4'] + 
        annual_by_draw['category_5']
    )
    annual_by_draw['hurricane_1_to_3'] = (
        annual_by_draw['category_1'] + annual_by_draw['category_2'] + 
        annual_by_draw['category_3']
    )
    annual_by_draw['hurricane_4_plus'] = (
        annual_by_draw['category_4'] + annual_by_draw['category_5']
    )
    
    annual_stats = calculate_stats(annual_by_draw, ['year'], metrics)
    
    print(f"  Calculated annual statistics across all countries")
    
    # ========================================================================
    # 4b. Basin-Level Annual Statistics
    # ========================================================================
    
    basin_annual_by_draw = storm_max_category.groupby(['basin', 'year', 'draw']).size().reset_index(name='total_storms')
    
    category_by_draw_basin = storm_max_category.groupby(
        ['basin', 'year', 'draw', 'storm_category']
    ).size().reset_index(name='storm_count')
    
    category_pivot_basin = category_by_draw_basin.pivot_table(
        index=['basin', 'year', 'draw'],
        columns='storm_category',
        values='storm_count',
        fill_value=0
    ).reset_index()
    
    basin_annual_by_draw = basin_annual_by_draw.merge(category_pivot_basin, on=['basin', 'year', 'draw'], how='left')
    
    for cat in ['tropical_storm', 'category_1', 'category_2', 'category_3', 'category_4', 'category_5']:
        if cat not in basin_annual_by_draw.columns:
            basin_annual_by_draw[cat] = 0
    
    basin_annual_by_draw['at_least_tropical_storm'] = basin_annual_by_draw['total_storms']
    basin_annual_by_draw['at_least_hurricane'] = (
        basin_annual_by_draw['category_1'] + basin_annual_by_draw['category_2'] + 
        basin_annual_by_draw['category_3'] + basin_annual_by_draw['category_4'] + 
        basin_annual_by_draw['category_5']
    )
    basin_annual_by_draw['hurricane_1_to_3'] = (
        basin_annual_by_draw['category_1'] + basin_annual_by_draw['category_2'] + 
        basin_annual_by_draw['category_3']
    )
    basin_annual_by_draw['hurricane_4_plus'] = (
        basin_annual_by_draw['category_4'] + basin_annual_by_draw['category_5']
    )
    
    basin_annual_stats = calculate_stats(basin_annual_by_draw, ['basin', 'year'], metrics)
    
    print(f"  Calculated basin-level statistics")
    
    # ========================================================================
    # 5. Save Summary Tables
    # ========================================================================
    
    output_dir = Path("/ihme/homes/bcreiner/repos/idd-climate-models/outputs/storm_admin_summaries")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_scenario_label = f"{args.model}_{args.variant}_{args.scenario}"
    
    country_annual_stats.to_csv(
        output_dir / f"country_annual_stats_{model_scenario_label}.csv", 
        index=False
    )
    
    annual_stats.to_csv(
        output_dir / f"annual_stats_all_countries_{model_scenario_label}.csv", 
        index=False
    )
    
    basin_annual_stats.to_csv(
        output_dir / f"basin_annual_stats_{model_scenario_label}.csv", 
        index=False
    )
    
    storm_max_category.to_csv(
        output_dir / f"storm_level_data_{model_scenario_label}.csv", 
        index=False
    )
    
    print(f"✓ Saved 4 summary CSV files to {output_dir}")
    print(f"  - country_annual_stats_{model_scenario_label}.csv")
    print(f"  - annual_stats_all_countries_{model_scenario_label}.csv")
    print(f"  - basin_annual_stats_{model_scenario_label}.csv")
    print(f"  - storm_level_data_{model_scenario_label}.csv")


if __name__ == "__main__":
    main()
