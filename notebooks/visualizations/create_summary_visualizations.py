"""
Summary visualizations for storm-admin0 impact analysis.

Compares storms across:
- Years, scenarios, model/variants
- Storm intensity categories
- Geographic locations (countries/basins)
- Relative to historical baseline

Usage:
    python create_summary_visualizations.py --data_dir /path/to/summaries
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Dict

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_all_data(data_dir: Path, file_type: str = "country_annual") -> pd.DataFrame:
    """
    Load all CSV files of a given type from the data directory.
    
    Args:
        data_dir: Directory containing summary CSV files
        file_type: One of 'country_annual', 'annual_stats_all', 'storm_level'
    
    Returns:
        Combined DataFrame with model/variant/scenario columns
    """
    pattern = f"{file_type}_*.csv"
    files = list(data_dir.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {data_dir}")
    
    dfs = []
    for file in files:
        # Parse model/variant/scenario from filename
        # Format: {file_type}_{model}_{variant}_{scenario}.csv
        parts = file.stem.replace(f"{file_type}_", "").split("_")
        if len(parts) >= 3:
            # Handle model names with underscores (e.g., CMCC-ESM2 becomes CMCC-ESM2)
            scenario = parts[-1]
            variant = parts[-2]
            model = "_".join(parts[:-2])
            
            df = pd.read_csv(file)
            df['model'] = model
            df['variant'] = variant
            df['scenario'] = scenario
            dfs.append(df)
    
    if not dfs:
        raise ValueError(f"No valid data files found for {file_type}")
    
    return pd.concat(dfs, ignore_index=True)


def calculate_historical_baseline(df: pd.DataFrame, 
                                   group_cols: List[str], 
                                   metrics: List[str]) -> pd.DataFrame:
    """
    Calculate historical baseline (mean across years) for normalization.
    
    Args:
        df: DataFrame with historical data
        group_cols: Columns to group by (e.g., ['model', 'variant', 'basin'])
        metrics: Metric columns to calculate baseline for
    
    Returns:
        DataFrame with baseline values
    """
    hist_data = df[df['scenario'] == 'historical'].copy()
    
    baseline = hist_data.groupby(group_cols)[metrics].mean().reset_index()
    baseline.columns = group_cols + [f'{m}_baseline' for m in metrics]
    
    return baseline


def compute_relative_change(df: pd.DataFrame, 
                            baseline: pd.DataFrame,
                            group_cols: List[str],
                            metrics: List[str]) -> pd.DataFrame:
    """
    Compute relative change from historical baseline.
    
    Args:
        df: DataFrame with data to normalize
        baseline: DataFrame with baseline values
        group_cols: Columns to merge on
        metrics: Metrics to compute relative change for
    
    Returns:
        DataFrame with relative change columns
    """
    df_merged = df.merge(baseline, on=group_cols, how='left')
    
    for metric in metrics:
        baseline_col = f'{metric}_baseline'
        if baseline_col in df_merged.columns:
            df_merged[f'{metric}_rel_change'] = (
                (df_merged[metric] - df_merged[baseline_col]) / 
                (df_merged[baseline_col] + 1e-10)  # Avoid division by zero
            ) * 100  # Convert to percentage
    
    return df_merged


def plot_time_series_by_scenario(df: pd.DataFrame,
                                 metric: str,
                                 scenario_filter: Optional[List[str]] = None,
                                 title: str = None,
                                 ylabel: str = None,
                                 output_file: Optional[Path] = None):
    """
    Plot time series of a metric across scenarios.
    
    Args:
        df: DataFrame with year, scenario, and metric columns (with _mean, _lower, _upper)
        metric: Metric name (without _mean suffix)
        scenario_filter: List of scenarios to include (None = all)
        title: Plot title
        ylabel: Y-axis label
        output_file: Path to save figure
    """
    if scenario_filter:
        df = df[df['scenario'].isin(scenario_filter)]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    scenarios = df['scenario'].unique()
    colors = sns.color_palette("husl", len(scenarios))
    
    for scenario, color in zip(scenarios, colors):
        df_scenario = df[df['scenario'] == scenario]
        
        ax.plot(df_scenario['year'], df_scenario[f'{metric}_mean'], 
                'o-', color=color, linewidth=2, markersize=4, label=scenario)
        
        if f'{metric}_lower' in df_scenario.columns and f'{metric}_upper' in df_scenario.columns:
            ax.fill_between(df_scenario['year'], 
                          df_scenario[f'{metric}_lower'], 
                          df_scenario[f'{metric}_upper'],
                          alpha=0.2, color=color)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(ylabel or metric, fontsize=12)
    ax.set_title(title or f'{metric} by Scenario', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_time_series_by_model(df: pd.DataFrame,
                              metric: str,
                              model_filter: Optional[List[str]] = None,
                              title: str = None,
                              ylabel: str = None,
                              output_file: Optional[Path] = None):
    """
    Plot time series of a metric across models/variants.
    
    Similar to plot_time_series_by_scenario but colored by model/variant.
    """
    if model_filter:
        df = df[df['model'].isin(model_filter)]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    df['model_variant'] = df['model'] + '/' + df['variant']
    model_variants = df['model_variant'].unique()
    colors = sns.color_palette("husl", len(model_variants))
    
    for mv, color in zip(model_variants, colors):
        df_mv = df[df['model_variant'] == mv]
        
        ax.plot(df_mv['year'], df_mv[f'{metric}_mean'], 
                'o-', color=color, linewidth=2, markersize=4, label=mv)
        
        if f'{metric}_lower' in df_mv.columns and f'{metric}_upper' in df_mv.columns:
            ax.fill_between(df_mv['year'], 
                          df_mv[f'{metric}_lower'], 
                          df_mv[f'{metric}_upper'],
                          alpha=0.2, color=color)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(ylabel or metric, fontsize=12)
    ax.set_title(title or f'{metric} by Model/Variant', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_relative_change_vs_historical(df: pd.DataFrame,
                                       metric: str,
                                       group_by: str = 'scenario',
                                       title: str = None,
                                       ylabel: str = None,
                                       output_file: Optional[Path] = None):
    """
    Plot relative change from historical baseline over time.
    
    Args:
        df: DataFrame with relative change columns
        metric: Metric name
        group_by: 'scenario' or 'model' to color lines by
        title: Plot title
        ylabel: Y-axis label
        output_file: Path to save figure
    """
    # Exclude historical from the plot
    df_plot = df[df['scenario'] != 'historical'].copy()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    if group_by == 'scenario':
        groups = df_plot['scenario'].unique()
        df_plot['group_label'] = df_plot['scenario']
    else:  # model
        df_plot['group_label'] = df_plot['model'] + '/' + df_plot['variant']
        groups = df_plot['group_label'].unique()
    
    colors = sns.color_palette("husl", len(groups))
    
    for group, color in zip(groups, colors):
        df_group = df_plot[df_plot['group_label'] == group]
        
        rel_col = f'{metric}_rel_change'
        if rel_col in df_group.columns:
            ax.plot(df_group['year'], df_group[rel_col], 
                   'o-', color=color, linewidth=2, markersize=4, label=group)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Historical baseline')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(ylabel or f'{metric} % Change from Historical', fontsize=12)
    ax.set_title(title or f'{metric} Relative Change from Historical', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_scenario_model_heatmap(df: pd.DataFrame,
                                metric: str,
                                year: int,
                                title: str = None,
                                output_file: Optional[Path] = None):
    """
    Create heatmap comparing metric values across scenarios and models for a specific year.
    
    Args:
        df: DataFrame with scenario, model, variant, year, and metric columns
        metric: Metric to plot
        year: Year to show
        title: Plot title
        output_file: Path to save figure
    """
    df_year = df[df['year'] == year].copy()
    df_year['model_variant'] = df_year['model'] + '/' + df_year['variant']
    
    metric_col = f'{metric}_mean' if f'{metric}_mean' in df_year.columns else metric
    
    pivot = df_year.pivot_table(
        index='model_variant',
        columns='scenario',
        values=metric_col,
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, cbar_kws={'label': metric})
    
    ax.set_title(title or f'{metric} by Model/Variant and Scenario ({year})', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Model/Variant', fontsize=12)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_top_countries_comparison(df: pd.DataFrame,
                                  metric: str,
                                  year: int,
                                  top_n: int = 15,
                                  scenario: str = None,
                                  title: str = None,
                                  output_file: Optional[Path] = None):
    """
    Plot top N countries by metric value for a specific year.
    
    Args:
        df: Country-level DataFrame
        metric: Metric to rank by
        year: Year to show
        top_n: Number of top countries to show
        scenario: Specific scenario to plot (None = all scenarios)
        title: Plot title
        output_file: Path to save figure
    """
    df_year = df[df['year'] == year].copy()
    
    if scenario:
        df_year = df_year[df_year['scenario'] == scenario]
    
    metric_col = f'{metric}_mean' if f'{metric}_mean' in df_year.columns else metric
    
    # Get top N countries by metric value
    top_countries = df_year.groupby('ADM0_NAME')[metric_col].sum().nlargest(top_n).index
    df_plot = df_year[df_year['ADM0_NAME'].isin(top_countries)]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if scenario:
        # Single scenario - simple bar plot
        country_values = df_plot.groupby('ADM0_NAME')[metric_col].sum().sort_values(ascending=True)
        country_values.plot(kind='barh', ax=ax, color='steelblue')
    else:
        # Multiple scenarios - grouped bar plot
        pivot = df_plot.pivot_table(
            index='ADM0_NAME',
            columns='scenario',
            values=metric_col,
            aggfunc='sum'
        )
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]
        pivot.plot(kind='barh', ax=ax)
    
    ax.set_xlabel(metric, fontsize=12)
    ax.set_ylabel('Country', fontsize=12)
    ax.set_title(title or f'Top {top_n} Countries by {metric} ({year})', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    else:
        plt.show()
    
    plt.close()


def create_all_visualizations(data_dir: Path, output_dir: Path):
    """
    Create comprehensive set of visualizations.
    
    Args:
        data_dir: Directory with summary CSV files
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Creating Summary Visualizations")
    print("=" * 80)
    
    # ========================================================================
    # Load Data
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80)
    
    try:
        annual_all = load_all_data(data_dir, "annual_stats_all_countries")
        print(f"✓ Loaded annual statistics across all countries")
        print(f"  Shape: {annual_all.shape}")
        print(f"  Models: {annual_all['model'].unique()}")
        print(f"  Scenarios: {annual_all['scenario'].unique()}")
    except Exception as e:
        print(f"❌ Could not load annual statistics: {e}")
        annual_all = None
    
    try:
        country_annual = load_all_data(data_dir, "country_annual_stats")
        print(f"✓ Loaded country-level annual statistics")
        print(f"  Shape: {country_annual.shape}")
        print(f"  Countries: {country_annual['ADM0_NAME'].nunique()}")
    except Exception as e:
        print(f"❌ Could not load country statistics: {e}")
        country_annual = None
    
    # ========================================================================
    # Define Metrics
    # ========================================================================
    
    metrics = [
        'total_storms',
        'at_least_tropical_storm',
        'at_least_hurricane',
        'hurricane_1_to_3',
        'hurricane_4_plus',
    ]
    
    metric_labels = {
        'total_storms': 'Total Storms',
        'at_least_tropical_storm': 'At Least Tropical Storm',
        'at_least_hurricane': 'At Least Hurricane (Cat 1+)',
        'hurricane_1_to_3': 'Hurricane Cat 1-3',
        'hurricane_4_plus': 'Major Hurricane (Cat 4-5)',
    }
    
    # ========================================================================
    # Annual Statistics Plots (All Countries)
    # ========================================================================
    
    if annual_all is not None:
        print("\n" + "=" * 80)
        print("Creating Annual Statistics Plots (All Countries)")
        print("=" * 80)
        
        # Time series by scenario
        for metric in metrics:
            if f'{metric}_mean' in annual_all.columns:
                plot_time_series_by_scenario(
                    annual_all,
                    metric=metric,
                    title=f'{metric_labels.get(metric, metric)} Over Time by Scenario',
                    ylabel='Storm Count',
                    output_file=output_dir / f'timeseries_scenario_{metric}.png'
                )
        
        # Time series by model/variant
        for metric in metrics:
            if f'{metric}_mean' in annual_all.columns:
                plot_time_series_by_model(
                    annual_all,
                    metric=metric,
                    title=f'{metric_labels.get(metric, metric)} Over Time by Model/Variant',
                    ylabel='Storm Count',
                    output_file=output_dir / f'timeseries_model_{metric}.png'
                )
        
        # Heatmaps for specific years
        years_to_plot = [2030, 2050, 2070, 2090]
        for metric in ['at_least_hurricane', 'hurricane_4_plus']:
            if f'{metric}_mean' in annual_all.columns:
                for year in years_to_plot:
                    if year in annual_all['year'].values:
                        plot_scenario_model_heatmap(
                            annual_all,
                            metric=metric,
                            year=year,
                            title=f'{metric_labels.get(metric, metric)} - {year}',
                            output_file=output_dir / f'heatmap_{metric}_{year}.png'
                        )
    
    # ========================================================================
    # Country-Level Plots
    # ========================================================================
    
    if country_annual is not None:
        print("\n" + "=" * 80)
        print("Creating Country-Level Plots")
        print("=" * 80)
        
        # Top countries by scenario
        years_to_plot = [2030, 2050, 2070, 2090]
        for year in years_to_plot:
            if year in country_annual['year'].values:
                for scenario in country_annual['scenario'].unique():
                    plot_top_countries_comparison(
                        country_annual,
                        metric='at_least_hurricane',
                        year=year,
                        top_n=15,
                        scenario=scenario,
                        title=f'Top 15 Countries by Hurricane Count - {scenario} ({year})',
                        output_file=output_dir / f'top_countries_{scenario}_{year}.png'
                    )
    
    # ========================================================================
    # Relative Change from Historical
    # ========================================================================
    
    if annual_all is not None:
        print("\n" + "=" * 80)
        print("Calculating Relative Change from Historical Baseline")
        print("=" * 80)
        
        # Calculate baseline
        baseline_cols = ['model', 'variant']
        baseline = calculate_historical_baseline(annual_all, baseline_cols, metrics)
        
        # Compute relative change
        annual_rel = compute_relative_change(annual_all, baseline, baseline_cols, metrics)
        
        # Plot relative changes by scenario
        for metric in metrics:
            if f'{metric}_rel_change' in annual_rel.columns:
                plot_relative_change_vs_historical(
                    annual_rel,
                    metric=metric,
                    group_by='scenario',
                    title=f'{metric_labels.get(metric, metric)} - % Change from Historical by Scenario',
                    ylabel='% Change from Historical Baseline',
                    output_file=output_dir / f'relchange_scenario_{metric}.png'
                )
        
        # Plot relative changes by model
        for metric in metrics:
            if f'{metric}_rel_change' in annual_rel.columns:
                plot_relative_change_vs_historical(
                    annual_rel,
                    metric=metric,
                    group_by='model',
                    title=f'{metric_labels.get(metric, metric)} - % Change from Historical by Model/Variant',
                    ylabel='% Change from Historical Baseline',
                    output_file=output_dir / f'relchange_model_{metric}.png'
                )
    
    print("\n" + "=" * 80)
    print("Visualization Summary")
    print("=" * 80)
    print(f"All plots saved to: {output_dir}")
    print(f"Total files created: {len(list(output_dir.glob('*.png')))}")


def main():
    parser = argparse.ArgumentParser(
        description="Create summary visualizations for storm-admin0 impact analysis"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/ihme/homes/bcreiner/repos/idd-climate-models/outputs/storm_admin_summaries"),
        help="Directory containing summary CSV files"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/mnt/team/idd/pub/idd_climate_models/outputs/storm_admin_visualizations"),
        help="Directory to save plots"
    )
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        print("\nRun the analysis workflow first to generate summary data.")
        return 1
    
    create_all_visualizations(args.data_dir, args.output_dir)
    
    return 0


if __name__ == "__main__":
    exit(main())
