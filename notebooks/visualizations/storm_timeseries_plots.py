"""
Storm Impact Time Series Visualization Module

Flexible time-series plotting for storm impact data across scenarios, models, basins, and metrics.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Union
from statsmodels.nonparametric.smoothers_lowess import lowess

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
import idd_climate_models.constants as rfc

# ============================================================================
# COLOR PALETTES
# ============================================================================

def get_scenario_colors():
    """Get scenario colors from constants."""
    return {k: v['color'] for k, v in rfc.ssp_scenario_map.items()}

def get_basin_colors():
    """Get basin color palette (placeholder - will be moved to constants)."""
    return {
        'GL': '#2E2E2E',  # Dark gray for global
        'NA': '#1f77b4',  # Blue
        'EP': '#ff7f0e',  # Orange
        'WP': '#2ca02c',  # Green
        'NI': '#d62728',  # Red
        'SI': '#9467bd',  # Purple
        'AU': '#8c564b',  # Brown
        'SP': '#e377c2',  # Pink
    }

def get_model_colors():
    """Get model color palette (placeholder - will be moved to constants)."""
    # Will use seaborn palette for now until we define specific colors
    return None  # Will trigger automatic color assignment

def get_metric_colors():
    """Get metric color palette (placeholder - will be moved to constants)."""
    return {
        'total_storms': '#636363',
        'tropical_storm': '#4ECDC4',
        'category_1': '#FFE66D',
        'category_2': '#FF9F1C',
        'category_3': '#FF6B35',
        'category_4': '#E53935',
        'category_5': '#4A0000',
        'at_least_tropical_storm': '#2E8B57',
        'at_least_hurricane': '#DC143C',
        'hurricane_1_to_3': '#FF8C00',
        'hurricane_4_plus': '#8B0000',
    }

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_combined_data(annual_all: pd.DataFrame, basin_annual: pd.DataFrame) -> pd.DataFrame:
    """
    Combine global (annual_all) and basin data into a single DataFrame.
    
    Args:
        annual_all: Global annual statistics (no basin column)
        basin_annual: Basin-level annual statistics
        
    Returns:
        Combined DataFrame with 'basin' column (GL for global data)
    """
    dfs_to_combine = []
    
    # Add 'basin' column to global data
    if not annual_all.empty:
        annual_gl = annual_all.copy()
        annual_gl['basin'] = 'GL'
        # Create model_variant column
        if 'model_variant' not in annual_gl.columns and 'model' in annual_gl.columns and 'variant' in annual_gl.columns:
            annual_gl['model_variant'] = annual_gl['model'] + '/' + annual_gl['variant']
        dfs_to_combine.append(annual_gl)
    
    # Process basin data
    if not basin_annual.empty:
        basin_data = basin_annual.copy()
        # Create model_variant column
        if 'model_variant' not in basin_data.columns and 'model' in basin_data.columns and 'variant' in basin_data.columns:
            basin_data['model_variant'] = basin_data['model'] + '/' + basin_data['variant']
        dfs_to_combine.append(basin_data)
    
    if not dfs_to_combine:
        print("WARNING: No data to combine!")
        return pd.DataFrame()
    
    # Combine
    combined = pd.concat(dfs_to_combine, ignore_index=True)
    
    print(f"Combined data info:")
    print(f"  Total rows: {len(combined)}")
    print(f"  Unique model/variants: {combined['model_variant'].nunique()}")
    print(f"  Unique basins: {combined['basin'].nunique()}")
    print(f"  Unique scenarios: {combined['scenario'].nunique()}")
    print(f"  Year range: {combined['year'].min()} - {combined['year'].max()}")
    
    return combined

def filter_years(df: pd.DataFrame, year_range: Tuple[int, int]) -> pd.DataFrame:
    """Filter data to year range."""
    return df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])].copy()

def apply_smoothing(df: pd.DataFrame, value_col: str, frac: float = 0.2) -> pd.DataFrame:
    """
    Apply LOWESS smoothing to a value column.
    
    Args:
        df: DataFrame with 'year' and value column (must be sorted by year)
        value_col: Name of column to smooth
        frac: LOWESS smoothing fraction (0-1)
        
    Returns:
        DataFrame with smoothed values
    """
    df = df.copy()
    if len(df) < 3:
        return df  # Not enough points to smooth
    
    smoothed = lowess(df[value_col].values, df['year'].values, frac=frac, return_sorted=False)
    df[value_col] = smoothed
    return df

def calculate_relative_to_historical(df: pd.DataFrame, 
                                     group_cols: List[str], 
                                     metric: str) -> pd.DataFrame:
    """
    Calculate values relative to historical baseline (divide by historical mean).
    
    Args:
        df: DataFrame with data including historical
        group_cols: Columns to group by (e.g., ['model', 'variant', 'basin'])
        metric: Metric column name (e.g., 'total_storms_mean')
        
    Returns:
        DataFrame with relative values
    """
    df = df.copy()
    
    # Calculate historical baseline for each group
    hist_data = df[df['scenario'] == 'historical'].copy()
    baseline = hist_data.groupby(group_cols)[metric].mean().reset_index()
    baseline = baseline.rename(columns={metric: f'{metric}_baseline'})
    
    # Merge baseline and calculate relative
    df = df.merge(baseline, on=group_cols, how='left')
    df[metric] = df[metric] / (df[f'{metric}_baseline'] + 1e-10)
    
    return df

# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def calculate_grid_layout(n_panels: int, nrows: Optional[int] = None, 
                         ncols: Optional[int] = None) -> Tuple[int, int]:
    """
    Calculate grid layout for multipanel plots.
    
    Args:
        n_panels: Number of panels needed
        nrows: Fixed number of rows (optional)
        ncols: Fixed number of columns (optional)
        
    Returns:
        (nrows, ncols) tuple
    """
    if nrows is not None and ncols is not None:
        return nrows, ncols
    elif nrows is not None:
        return nrows, int(np.ceil(n_panels / nrows))
    elif ncols is not None:
        return int(np.ceil(n_panels / ncols)), ncols
    else:
        # Auto-calculate to be roughly square
        ncols = int(np.ceil(np.sqrt(n_panels)))
        nrows = int(np.ceil(n_panels / ncols))
        return nrows, ncols

def get_color_for_value(value: str, variable: str, color_palette: dict = None) -> str:
    """
    Get color for a specific value of a variable.
    
    Args:
        value: The specific value (e.g., 'ssp126', 'NA', 'CMCC-ESM2/r1i1p1f1')
        variable: The variable type ('scenario', 'basin', 'model_variant', 'metric')
        color_palette: Optional override color palette
        
    Returns:
        Hex color string
    """
    if color_palette and value in color_palette:
        return color_palette[value]
    
    if variable == 'scenario':
        return get_scenario_colors().get(value, '#000000')
    elif variable == 'basin':
        return get_basin_colors().get(value, '#666666')
    elif variable == 'metric':
        return get_metric_colors().get(value, '#666666')
    else:
        # For models or other, return None to use automatic colors
        return None

def plot_single_line(ax, df: pd.DataFrame, metric: str, color: str, label: str,
                     uncertainty: bool = False, alpha: float = 0.2):
    """Plot a single time series line with optional uncertainty band."""
    df = df.sort_values('year')
    
    mean_col = f'{metric}_mean'
    lower_col = f'{metric}_lower'
    upper_col = f'{metric}_upper'
    
    ax.plot(df['year'], df[mean_col], 'o-', color=color, linewidth=2, 
            markersize=4, label=label, alpha=0.9)
    
    if uncertainty and lower_col in df.columns and upper_col in df.columns:
        ax.fill_between(df['year'], df[lower_col], df[upper_col], 
                       color=color, alpha=alpha)

# ============================================================================
# TYPE-SPECIFIC PLOTTING FUNCTIONS
# ============================================================================

def plot_type_a(df: pd.DataFrame, metric: str, config: dict, ax=None):
    """
    Type A: Single plot, all variables fixed, one line.
    
    Config should contain:
        - model_variant: str
        - scenario: str (or list with one scenario)
        - basin: str
        - uncertainty: bool
        - relative: bool
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.get('figsize', (12, 6)))
    
    # Debug: Show available values
    print(f"Debug - Filtering for model_variant='{config['model_variant']}', basin='{config['basin']}'")
    print(f"  Available model_variants: {sorted(df['model_variant'].unique())[:5]}...")
    print(f"  Available basins: {sorted(df['basin'].unique())}")
    
    # Filter to specified values
    df_plot = df[
        (df['model_variant'] == config['model_variant']) &
        (df['basin'] == config['basin'])
    ].copy()
    
    if df_plot.empty:
        print(f"ERROR: No data found for model_variant='{config['model_variant']}', basin='{config['basin']}'")
        print(f"Please check your spelling and available options above.")
        return ax
    
    # Handle scenarios (plot historical + specified future scenario)
    scenarios = ['historical']
    if isinstance(config['scenario'], list):
        scenarios.extend([s for s in config['scenario'] if s != 'historical'])
    else:
        if config['scenario'] != 'historical':
            scenarios.append(config['scenario'])
    
    lines_plotted = 0
    for scenario in scenarios:
        df_scenario = df_plot[df_plot['scenario'] == scenario].copy()
        if df_scenario.empty:
            print(f"  Warning: No data for scenario '{scenario}'")
            continue
        
        color = get_color_for_value(scenario, 'scenario')
        label = rfc.ssp_scenario_map.get(scenario, {}).get('name', scenario)
        
        plot_single_line(ax, df_scenario, metric, color, label, 
                        uncertainty=config.get('uncertainty', False))
        lines_plotted += 1
    
    if lines_plotted == 0:
        print(f"ERROR: No data plotted. Check that scenarios {scenarios} exist in your data.")
        print(f"  Available scenarios: {sorted(df_plot['scenario'].unique())}")
    
    ax.set_xlabel('Year', fontsize=12)
    ylabel = metric.replace('_', ' ').title()
    if config.get('relative', False):
        ylabel += ' (Relative to Historical)'
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Only add legend if show_legend is True (default True for standalone plots)
    if lines_plotted > 0 and config.get('show_legend', True):
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_type_b(df: pd.DataFrame, metric: str, config: dict, ax=None):
    """
    Type B: Single plot, one variable unfixed (multiple lines).
    
    Config should contain:
        - unfixed_var: str (variable name that's unfixed)
        - unfixed_values: list (values to plot, or ['all'])
        - Fixed variables (the ones not unfixed)
        - uncertainty: bool
        - relative: bool
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.get('figsize', (12, 6)))
    
    unfixed_var = config['unfixed_var']
    unfixed_values = config['unfixed_values']
    
    # Filter to fixed values
    df_plot = df.copy()
    for var in ['model_variant', 'scenario', 'basin']:
        if var != unfixed_var and var in config:
            if var == 'scenario':
                # For scenario, we need to handle historical separately
                scenarios = config[var] if isinstance(config[var], list) else [config[var]]
                df_plot = df_plot[df_plot[var].isin(scenarios + ['historical'])]
            else:
                val = config[var]
                df_plot = df_plot[df_plot[var] == val]
    
    # Get values to plot for unfixed variable
    if unfixed_values == ['all']:
        if unfixed_var == 'scenario':
            # For scenario, exclude historical from line iteration
            plot_values = [s for s in df_plot[unfixed_var].unique() if s != 'historical']
        else:
            plot_values = sorted(df_plot[unfixed_var].unique())
    else:
        plot_values = unfixed_values
    
    # Plot historical once if scenario is unfixed
    if unfixed_var == 'scenario':
        df_hist = df_plot[df_plot['scenario'] == 'historical'].copy()
        if not df_hist.empty:
            color = get_color_for_value('historical', 'scenario')
            label = 'Historical'
            plot_single_line(ax, df_hist, metric, color, label, 
                           uncertainty=config.get('uncertainty', False))
    
    # Determine colors
    colors = {}
    if config.get('color_palette'):
        colors = config['color_palette']
    else:
        palette_colors = None
        if unfixed_var == 'model_variant':
            # Use seaborn palette for models
            palette_colors = sns.color_palette('husl', len(plot_values))
            colors = {val: plt.matplotlib.colors.rgb2hex(c) for val, c in zip(plot_values, palette_colors)}
        else:
            for val in plot_values:
                c = get_color_for_value(val, unfixed_var)
                if c:
                    colors[val] = c
        
        # If we still don't have colors, use seaborn
        if not colors:
            palette_colors = sns.color_palette('husl', len(plot_values))
            colors = {val: plt.matplotlib.colors.rgb2hex(c) for val, c in zip(plot_values, palette_colors)}
    
    # Plot each value
    for val in plot_values:
        df_val = df_plot[df_plot[unfixed_var] == val].copy()
        if df_val.empty:
            continue
        
        color = colors.get(val, '#666666')
        label = val
        if unfixed_var == 'scenario':
            label = rfc.ssp_scenario_map.get(val, {}).get('name', val)
        elif unfixed_var == 'basin':
            label = rfc.basin_dict.get(val, {}).get('name', val)
        
        plot_single_line(ax, df_val, metric, color, label, 
                        uncertainty=config.get('uncertainty', False))
    
    ax.set_xlabel('Year', fontsize=12)
    ylabel = metric.replace('_', ' ').title()
    if config.get('relative', False):
        ylabel += ' (Relative to Historical)'
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Only add legend if show_legend is True (default True for standalone plots)
    if config.get('show_legend', True):
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_type_c(df: pd.DataFrame, metric_or_var: Union[str, List[str]], config: dict):
    """
    Type C: Multiple panels, one line per panel.
    
    Config should contain:
        - unfixed_var: str (variable that varies across panels)
        - unfixed_values: list (panel values, or ['all'])
        - is_metric_varied: bool (True if metric is varied, False if another var)
        - Fixed variables
        - uncertainty: bool
        - relative: bool
        - nrows, ncols: optional panel layout
        - shared_y: bool
    """
    unfixed_var = config['unfixed_var']
    unfixed_values = config['unfixed_values']
    is_metric_varied = config.get('is_metric_varied', False)
    
    # Determine what to plot
    if is_metric_varied:
        if unfixed_values == ['all']:
            # Get all available metrics
            metrics = [col.replace('_mean', '') for col in df.columns if col.endswith('_mean')]
        else:
            metrics = unfixed_values
        panel_items = metrics
    else:
        if unfixed_values == ['all']:
            panel_items = sorted(df[unfixed_var].unique())
        else:
            panel_items = unfixed_values
        metrics = [metric_or_var] * len(panel_items)  # Same metric for all panels
    
    # Calculate layout
    n_panels = len(panel_items)
    nrows, ncols = calculate_grid_layout(n_panels, config.get('nrows'), config.get('ncols'))
    
    # Create figure
    figsize = config.get('figsize', (14, 4 * nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                             sharex=True, sharey=config.get('shared_y', False))
    axes = np.atleast_1d(axes).flatten()
    
    # Plot each panel
    for idx, (panel_item, metric) in enumerate(zip(panel_items, metrics)):
        ax = axes[idx]
        
        # Filter data for this panel
        if is_metric_varied:
            df_panel = df.copy()
        else:
            df_panel = df[df[unfixed_var] == panel_item].copy()
        
        # Create sub-config for type_a style plot
        panel_config = config.copy()
        panel_config['show_legend'] = False  # Don't show legend on individual panels
        if not is_metric_varied:
            panel_config[unfixed_var] = panel_item
        
        # Plot
        plot_type_a(df_panel, metric, panel_config, ax=ax)
        
        # Set title
        if is_metric_varied:
            title = metric.replace('_', ' ').title()
        else:
            if unfixed_var == 'basin':
                title = rfc.basin_dict.get(panel_item, {}).get('name', panel_item)
            elif unfixed_var == 'scenario':
                title = rfc.ssp_scenario_map.get(panel_item, {}).get('name', panel_item)
            else:
                title = panel_item
        ax.set_title(title, fontsize=11, fontweight='bold')
    
    # Hide extra axes
    for idx in range(len(panel_items), len(axes)):
        axes[idx].set_visible(False)
    
    # Create a single shared legend at the bottom
    if len(panel_items) > 0:
        # Get handles and labels from the first panel with data
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                      ncol=min(len(handles), 5), fontsize=10, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for legend
    return fig

def plot_type_d(df: pd.DataFrame, metric: str, config: dict):
    """
    Type D: Multiple panels (first unfixed var), multiple lines per panel (second unfixed var).
    
    Config should contain:
        - unfixed_var_1: str (panels)
        - unfixed_values_1: list (or ['all'])
        - unfixed_var_2: str (lines within panels)
        - unfixed_values_2: list (or ['all'])
        - Fixed variables
        - uncertainty: bool
        - relative: bool
        - nrows, ncols: optional
        - shared_y: bool
    """
    unfixed_var_1 = config['unfixed_var_1']
    unfixed_values_1 = config['unfixed_values_1']
    unfixed_var_2 = config['unfixed_var_2']
    unfixed_values_2 = config['unfixed_values_2']
    
    # Get panel values
    if unfixed_values_1 == ['all']:
        panel_values = sorted(df[unfixed_var_1].unique())
    else:
        panel_values = unfixed_values_1
    
    # Calculate layout
    n_panels = len(panel_values)
    nrows, ncols = calculate_grid_layout(n_panels, config.get('nrows'), config.get('ncols'))
    
    # Create figure
    figsize = config.get('figsize', (14, 4 * nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=True, sharey=config.get('shared_y', False))
    axes = np.atleast_1d(axes).flatten()
    
    # Plot each panel
    for idx, panel_val in enumerate(panel_values):
        ax = axes[idx]
        
        # Filter to this panel
        df_panel = df[df[unfixed_var_1] == panel_val].copy()
        
        # Create config for type_b style plot
        panel_config = config.copy()
        panel_config['show_legend'] = False  # Don't show legend on individual panels
        panel_config[unfixed_var_1] = panel_val
        panel_config['unfixed_var'] = unfixed_var_2
        panel_config['unfixed_values'] = unfixed_values_2
        
        # Plot
        plot_type_b(df_panel, metric, panel_config, ax=ax)
        
        # Set title
        if unfixed_var_1 == 'basin':
            title = rfc.basin_dict.get(panel_val, {}).get('name', panel_val)
        elif unfixed_var_1 == 'scenario':
            title = rfc.ssp_scenario_map.get(panel_val, {}).get('name', panel_val)
        else:
            title = panel_val
        ax.set_title(title, fontsize=11, fontweight='bold')
    
    # Hide extra axes
    for idx in range(len(panel_values), len(axes)):
        axes[idx].set_visible(False)
    
    # Create a single shared legend at the bottom
    if len(panel_values) > 0:
        # Get handles and labels from the first panel with data
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
                      ncol=min(len(handles), 5), fontsize=10, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for legend
    return fig

# ============================================================================
# MAIN PLOTTING WRAPPER
# ============================================================================

def plot_timeseries(
    annual_all: pd.DataFrame,
    basin_annual: pd.DataFrame,
    metric: Union[str, List[str]],
    model_variant: Optional[Union[str, List[str]]] = None,
    scenario: Optional[Union[str, List[str]]] = None,
    basin: Optional[Union[str, List[str]]] = None,
    uncertainty: bool = True,
    smooth: bool = False,
    smooth_frac: float = 0.2,
    relative: bool = False,
    year_range: Tuple[int, int] = (1980, 2100),
    shared_y: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    save_path: Optional[str] = None,
    color_palette: Optional[dict] = None,
    main_title: Optional[str] = None,
):
    """
    Main wrapper function for time series plotting.
    
    Args:
        annual_all: Global annual statistics DataFrame
        basin_annual: Basin annual statistics DataFrame
        metric: Metric name(s) to plot. Can be:
            - str: Single metric for most plot types
            - List[str]: Multiple metrics for Type C plots where metric varies
        model_variant: Model/variant spec. Options:
            - str: Fixed to specific model/variant
            - List[str]: Specific subset to plot
            - ['all']: All available
            - None: Must be unfixed variable
        scenario: Scenario spec (same options as model_variant)
        basin: Basin spec (same options as model_variant)
        uncertainty: Whether to plot uncertainty bands
        smooth: Whether to apply LOWESS smoothing
        smooth_frac: LOWESS smoothing fraction (0-1)
        relative: Whether to plot relative to historical baseline (divides by historical mean)
        year_range: (start_year, end_year) tuple
        shared_y: Whether to share y-axis across panels (multipanel plots)
        figsize: Figure size (width, height)
        nrows: Number of rows for multipanel plots
        ncols: Number of columns for multipanel plots
        save_path: Path to save figure (None = don't save)
        color_palette: Override color palette dict
        
    Returns:
        matplotlib figure or axis object
    """
    # Validate relative and uncertainty
    if relative and uncertainty:
        raise ValueError("Cannot use relative=True with uncertainty=True")
    
    # Prepare data
    df = prepare_combined_data(annual_all, basin_annual)
    
    if df.empty:
        raise ValueError("No data available after combining annual_all and basin_annual")
    
    df = filter_years(df, year_range)
    
    # Validate metric exists
    if not isinstance(metric, list):
        metric_col = f'{metric}_mean'
        if metric_col not in df.columns:
            available_metrics = [col.replace('_mean', '') for col in df.columns if col.endswith('_mean')]
            raise ValueError(f"Metric '{metric}' not found in data. Available metrics: {available_metrics}")
    
    # Determine plot type based on which variables are lists/None
    vars_config = {
        'model_variant': model_variant,
        'scenario': scenario,
        'basin': basin,
        'metric': metric
    }
    
    # Count unfixed variables (None or list)
    unfixed = []
    for var_name, var_val in vars_config.items():
        if var_val is None or isinstance(var_val, list):
            unfixed.append(var_name)
    
    # Determine metric handling
    if isinstance(metric, list):
        if len(unfixed) > 1:
            raise ValueError("Cannot vary metric and multiple other variables simultaneously")
        plot_type = 'c'
        is_metric_varied = True
        unfixed_var = 'metric'
        unfixed_values = metric
    else:
        # Standard variable handling
        n_unfixed = len(unfixed)
        
        if n_unfixed == 0:
            plot_type = 'a'
        elif n_unfixed == 1:
            # Could be type b or c depending on scenario
            unfixed_var = unfixed[0]
            unfixed_values = vars_config[unfixed_var]
            
            # If it's a single-item list, it's type b style (one line)
            # If it's multi-item or 'all', need to check if we want multiple panels
            if unfixed_values is None:
                unfixed_values = ['all']
            
            # Simple heuristic: if unfixed var is scenario and it's a multi-value, type b
            # Otherwise ask user or default to b
            plot_type = 'b'  # Default to multiple lines in one plot
            
        elif n_unfixed == 2:
            plot_type = 'd'
            unfixed_var_1 = unfixed[0]
            unfixed_var_2 = unfixed[1]
            unfixed_values_1 = vars_config[unfixed_var_1] if vars_config[unfixed_var_1] is not None else ['all']
            unfixed_values_2 = vars_config[unfixed_var_2] if vars_config[unfixed_var_2] is not None else ['all']
        else:
            raise ValueError(f"Too many unfixed variables ({n_unfixed}). Maximum is 2.")
    
    # Build config dict
    config = {
        'uncertainty': uncertainty,
        'relative': relative,
        'figsize': figsize,
        'shared_y': shared_y,
        'nrows': nrows,
        'ncols': ncols,
        'color_palette': color_palette,
    }
    
    # Add fixed values to config
    for var_name, var_val in vars_config.items():
        if var_val is not None and not isinstance(var_val, list):
            config[var_name] = var_val
        elif isinstance(var_val, list) and len(var_val) == 1 and var_val[0] != 'all':
            config[var_name] = var_val[0]
    
    # Prepare metric column name
    if not isinstance(metric, list):
        metric_col = f'{metric}_mean'
    
    # Apply relative transformation if requested
    if relative:
        if isinstance(metric, list):
            raise ValueError("Cannot use relative=True when varying metrics")
        
        # Group by all non-scenario variables
        group_cols = []
        for var in ['model_variant', 'basin']:
            if var in config:
                group_cols.append(var.replace('_variant', ''))
                if var == 'model_variant':
                    group_cols.append('variant')
        
        df = calculate_relative_to_historical(df, group_cols, metric_col)
    
    # Apply smoothing if requested
    if smooth:
        # Group by all variable combinations and smooth
        group_vars = ['model', 'variant', 'scenario', 'basin']
        
        dfs_smoothed = []
        for name, group in df.groupby(group_vars):
            group = group.sort_values('year')
            
            # Smooth mean
            if isinstance(metric, list):
                # Smooth each metric in the list
                for m in metric:
                    m_col = f'{m}_mean'
                    if m_col in group.columns:
                        group = apply_smoothing(group, m_col, smooth_frac)
                    
                    # Smooth uncertainty bounds if present
                    if uncertainty:
                        lower_col = f'{m}_lower'
                        upper_col = f'{m}_upper'
                        if lower_col in group.columns:
                            group = apply_smoothing(group, lower_col, smooth_frac)
                        if upper_col in group.columns:
                            group = apply_smoothing(group, upper_col, smooth_frac)
            else:
                # Single metric
                group = apply_smoothing(group, metric_col, smooth_frac)
                
                # Smooth uncertainty bounds if present and requested
                if uncertainty:
                    lower_col = f'{metric}_lower'
                    upper_col = f'{metric}_upper'
                    if lower_col in group.columns:
                        group = apply_smoothing(group, lower_col, smooth_frac)
                    if upper_col in group.columns:
                        group = apply_smoothing(group, upper_col, smooth_frac)
            
            dfs_smoothed.append(group)
        
        df = pd.concat(dfs_smoothed, ignore_index=True)

    # Route to appropriate plotting function
    if plot_type == 'a':
        result = plot_type_a(df, metric, config)
        fig = result.figure
        
    elif plot_type == 'b':
        config['unfixed_var'] = unfixed_var
        config['unfixed_values'] = unfixed_values
        result = plot_type_b(df, metric, config)
        fig = result.figure
        
    elif plot_type == 'c':
        if is_metric_varied:
            config['unfixed_var'] = 'metric'
            config['unfixed_values'] = metric
            config['is_metric_varied'] = True
        else:
            # User would need to explicitly request type c somehow
            # For now, default to type b behavior
            config['unfixed_var'] = unfixed_var
            config['unfixed_values'] = unfixed_values
            config['is_metric_varied'] = False
        
        fig = plot_type_c(df, metric, config)
        
    elif plot_type == 'd':
        config['unfixed_var_1'] = unfixed_var_1
        config['unfixed_values_1'] = unfixed_values_1
        config['unfixed_var_2'] = unfixed_var_2
        config['unfixed_values_2'] = unfixed_values_2
        
        fig = plot_type_d(df, metric, config)
    
    # Add main title if requested
    if main_title:
        fig.suptitle(main_title, fontsize=16, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig
