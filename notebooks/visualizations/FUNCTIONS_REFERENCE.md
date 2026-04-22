# Visualizations Functions Reference

This document catalogs all reusable functions in the `notebooks/visualizations/` folder for creating new analysis scripts.

## Table of Contents
1. [Data Loading & Preparation](#data-loading--preparation)
2. [Spatial Analysis](#spatial-analysis)
3. [Storm Counting](#storm-counting)
4. [Time Series Plotting](#time-series-plotting)
5. [Summary Visualizations](#summary-visualizations)
6. [Helper Utilities](#helper-utilities)

---

## Data Loading & Preparation

### From `create_summary_visualizations.py`

#### `load_all_data(data_dir: Path, file_type: str = "country_annual") -> pd.DataFrame`
Load all CSV files of a given type from the data directory.

**Parameters:**
- `data_dir`: Directory containing summary CSV files
- `file_type`: One of 'country_annual', 'annual_stats_all', 'storm_level'

**Returns:** Combined DataFrame with model/variant/scenario columns

**Example:**
```python
from pathlib import Path
data_dir = Path("/path/to/summaries")
df = load_all_data(data_dir, file_type="country_annual")
```

---

#### `calculate_historical_baseline(df, group_cols, metrics) -> pd.DataFrame`
Calculate historical baseline (mean across years) for normalization.

**Parameters:**
- `df`: DataFrame with historical data
- `group_cols`: Columns to group by (e.g., ['model', 'variant', 'basin'])
- `metrics`: Metric columns to calculate baseline for

**Returns:** DataFrame with baseline values

**Example:**
```python
baseline = calculate_historical_baseline(
    df, 
    group_cols=['model', 'variant', 'basin'],
    metrics=['total_storms', 'category_5']
)
```

---

#### `compute_relative_change(df, baseline, group_cols, metrics) -> pd.DataFrame`
Compute relative change from historical baseline.

**Parameters:**
- `df`: DataFrame with data to normalize
- `baseline`: DataFrame with baseline values
- `group_cols`: Columns to merge on
- `metrics`: Metrics to compute relative change for

**Returns:** DataFrame with relative change columns

---

### From `storm_timeseries_plots.py`

#### `prepare_combined_data(annual_all, basin_annual) -> pd.DataFrame`
Combine global (annual_all) and basin data into a single DataFrame.

**Parameters:**
- `annual_all`: Global annual statistics (no basin column)
- `basin_annual`: Basin-level annual statistics

**Returns:** Combined DataFrame with 'basin' column (GL for global data)

**Example:**
```python
combined = prepare_combined_data(annual_all_df, basin_annual_df)
```

---

#### `filter_years(df, year_range: Tuple[int, int]) -> pd.DataFrame`
Filter data to year range.

**Parameters:**
- `df`: DataFrame with 'year' column
- `year_range`: (start_year, end_year) tuple

**Example:**
```python
df_filtered = filter_years(df, (1980, 2050))
```

---

#### `apply_smoothing(df, value_col: str, frac: float = 0.2) -> pd.DataFrame`
Apply LOWESS smoothing to a value column.

**Parameters:**
- `df`: DataFrame with 'year' and value column (must be sorted by year)
- `value_col`: Name of column to smooth
- `frac`: LOWESS smoothing fraction (0-1), default 0.2

**Returns:** DataFrame with smoothed values

**Example:**
```python
df_smooth = apply_smoothing(df, 'total_storms_mean', frac=0.3)
```

---

#### `calculate_relative_to_historical(df, group_cols, metric) -> pd.DataFrame`
Calculate values relative to historical baseline (divide by historical mean).

**Parameters:**
- `df`: DataFrame with data including historical
- `group_cols`: Columns to group by (e.g., ['model', 'variant', 'basin'])
- `metric`: Metric column name (e.g., 'total_storms_mean')

**Returns:** DataFrame with relative values

---

## Spatial Analysis

### From `clip_admin_shapefile.py`

#### `parse_basin_coordinate(coord_str: str) -> float`
Convert coordinate string like '260E' or '45S' to numeric value.

**Parameters:**
- `coord_str`: Coordinate string with direction (E/W/N/S) at the end

**Returns:** Numeric coordinate value (positive East/North, negative West/South)

**Example:**
```python
lon = parse_basin_coordinate('260E')  # Returns -100.0
lat = parse_basin_coordinate('45S')   # Returns -45.0
```

**Note:** Automatically converts longitude from 0-360 to -180-180 if needed.

---

### From `run_storm_admin_analysis.py`

#### `create_storm_admin_impact_dataframe(ds, admin0_gdf, basin_bounds_polygon=None)`
Create a dataframe of storm impacts on admin 0 regions using optimized vectorized operations.

**Parameters:**
- `ds`: xarray Dataset with storm track data (lon, lat, max_sustained_wind, tc_years, tc_month)
- `admin0_gdf`: GeoDataFrame with admin boundaries
- `basin_bounds_polygon`: Optional shapely polygon to clip admin boundaries

**Returns:** DataFrame with columns:
- `storm_track`, `year`, `month`
- `ADM0_CODE`, `ADM0_NAME`, `loc_id`
- `max_wind_speed`, `storm_category`

**Example:**
```python
import xarray as xr
import geopandas as gpd

ds = xr.open_dataset('tracks.nc')
admin_gdf = gpd.read_file('admin_boundaries.shp')
impact_df = create_storm_admin_impact_dataframe(ds, admin_gdf)
```

**Performance:** Uses vectorized operations and spatial indexes for efficiency.

---

## Storm Counting

### From `count_storms.py`

Functions for counting storms by different groupings from storm-level data files.

**Key Operations:**
- Count storms per draw across all models/variants/scenarios
- Count storms by model/variant/scenario/basin/time_period/draw
- Calculate average storms per draw by scenario/basin/time_period
- Generate detailed and summary count files

**Example Usage:**
```python
# Load storm-level data
storm_files = list(DATA_DIR.glob("storm_level_data_*.csv"))
dfs = []
for file in storm_files:
    parts = file.stem.replace("storm_level_data_", "").split("_")
    scenario = parts[-1]
    variant = parts[-2]
    model = "_".join(parts[:-2])
    df = pd.read_csv(file)
    df['model'] = model
    df['variant'] = variant
    df['scenario'] = scenario
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# Count by different groupings
storms_per_draw = df_all.groupby('draw').size()
storms_detailed = df_all.groupby(
    ['model', 'variant', 'scenario', 'basin', 'time_period', 'draw']
).size()
```

---

### From `count_all_storms.py`

Functions for counting ALL storms (landfall + non-landfall) directly from TC risk NetCDF files.

**Key Features:**
- Opens track files directly from TC risk output
- Counts storms using `ds.dims.get('n_trk', 0)`
- Parses file paths to extract model/variant/scenario metadata

**Example:**
```python
import xarray as xr
from pathlib import Path

# Count storms in a track file
with xr.open_dataset(track_file) as ds:
    n_storms = ds.dims.get('n_trk', 0)
```

---

### From `count_all_storms_from_timebins.py`

Fast storm counting using pre-computed TempestExtremes time bins file.

**Key Advantage:** Much faster than opening individual NetCDF files!

**Example:**
```python
import idd_climate_models.constants as rfc

# Load time bins (contains storm counts per basin/model/scenario)
df = pd.read_csv(rfc.TIME_BINS_WIDE_DF_PATH)

# Basin columns: AU_int, EP_int, GL_int, NA_int, NI_int, SI_int, SP_int, WP_int
basin_cols = ['AU_int', 'EP_int', 'GL_int', 'NA_int', 'NI_int', 'SI_int', 'SP_int', 'WP_int']

# Total storms by basin
for basin in ['AU', 'EP', 'GL', 'NA', 'NI', 'SI', 'SP', 'WP']:
    total = df[f'{basin}_int'].sum()
    print(f"{basin}: {total}")

# By scenario
scenario_totals = df.groupby('scenario')[basin_cols].sum().sum(axis=1)
```

---

## Time Series Plotting

### From `storm_timeseries_plots.py`

This module provides a flexible, comprehensive time-series plotting system.

### Color Palettes

#### `get_scenario_colors() -> dict`
Get scenario colors from constants.

#### `get_basin_colors() -> dict`
Get basin color palette.

#### `get_model_colors() -> dict`
Get model color palette (uses seaborn automatic colors).

#### `get_metric_colors() -> dict`
Get metric color palette for different storm categories.

---

### Main Plotting Function

#### `plot_timeseries(...)`
Main wrapper function for flexible time series plotting.

**Parameters:**
- `annual_all`: Global annual statistics DataFrame
- `basin_annual`: Basin annual statistics DataFrame
- `metric`: Metric name(s) to plot (str or List[str])
- `model_variant`: str, List[str], ['all'], or None
- `scenario`: str, List[str], ['all'], or None
- `basin`: str, List[str], ['all'], or None
- `uncertainty`: Whether to plot uncertainty bands (default True)
- `smooth`: Whether to apply LOWESS smoothing (default False)
- `smooth_frac`: LOWESS smoothing fraction 0-1 (default 0.2)
- `relative`: Plot relative to historical baseline (default False)
- `year_range`: (start_year, end_year) tuple (default (1980, 2100))
- `shared_y`: Share y-axis across panels (default False)
- `figsize`: Figure size (width, height)
- `nrows`, `ncols`: Panel layout for multipanel plots
- `save_path`: Path to save figure (None = don't save)
- `color_palette`: Override color palette dict

**Plot Types (automatic detection):**

**Type A:** Single plot, one line
```python
# Fixed model, scenario, basin
fig = plot_timeseries(
    annual_all, basin_annual,
    metric='total_storms',
    model_variant='CMCC-ESM2/r1i1p1f1',
    scenario='ssp126',
    basin='NA'
)
```

**Type B:** Single plot, multiple lines
```python
# Compare scenarios for fixed model/basin
fig = plot_timeseries(
    annual_all, basin_annual,
    metric='total_storms',
    model_variant='CMCC-ESM2/r1i1p1f1',
    scenario=['historical', 'ssp126', 'ssp245', 'ssp585'],  # Multiple scenarios
    basin='NA'
)
```

**Type C:** Multiple panels, one line per panel
```python
# Show different metrics in separate panels
fig = plot_timeseries(
    annual_all, basin_annual,
    metric=['total_storms', 'category_4', 'category_5'],  # Multiple metrics
    model_variant='CMCC-ESM2/r1i1p1f1',
    scenario='ssp126',
    basin='GL',
    nrows=1, ncols=3
)
```

**Type D:** Multiple panels, multiple lines per panel
```python
# Compare scenarios across multiple basins
fig = plot_timeseries(
    annual_all, basin_annual,
    metric='total_storms',
    model_variant='CMCC-ESM2/r1i1p1f1',
    scenario=['historical', 'ssp126', 'ssp585'],  # Lines
    basin=['NA', 'EP', 'WP', 'NI'],  # Panels
    nrows=2, ncols=2
)
```

**Advanced Options:**
```python
# With smoothing and relative to historical
fig = plot_timeseries(
    annual_all, basin_annual,
    metric='total_storms',
    model_variant='CMCC-ESM2/r1i1p1f1',
    scenario=['ssp126', 'ssp585'],
    basin='GL',
    smooth=True,
    smooth_frac=0.3,
    relative=True,
    year_range=(1990, 2100),
    save_path='output/my_plot.png'
)
```

---

### Low-Level Plotting Functions

#### `plot_type_a(df, metric, config, ax=None)`
Type A: Single plot, all variables fixed, one line.

#### `plot_type_b(df, metric, config, ax=None)`
Type B: Single plot, one variable unfixed (multiple lines).

#### `plot_type_c(df, metric_or_var, config)`
Type C: Multiple panels, one line per panel.

#### `plot_type_d(df, metric, config)`
Type D: Multiple panels, multiple lines per panel.

#### `plot_single_line(ax, df, metric, color, label, uncertainty=False, alpha=0.2)`
Plot a single time series line with optional uncertainty band.

---

### Helper Functions

#### `calculate_grid_layout(n_panels, nrows=None, ncols=None) -> Tuple[int, int]`
Calculate grid layout for multipanel plots.

#### `get_color_for_value(value: str, variable: str, color_palette=None) -> str`
Get color for a specific value of a variable.

---

## Summary Visualizations

### From `create_summary_visualizations.py`

#### `plot_time_series_by_scenario(df, metric, scenario_filter=None, title=None, ylabel=None, output_file=None)`
Plot time series of a metric across scenarios with uncertainty bands.

**Example:**
```python
plot_time_series_by_scenario(
    df,
    metric='total_storms',
    scenario_filter=['historical', 'ssp126', 'ssp585'],
    ylabel='Number of Storms',
    output_file='storms_by_scenario.png'
)
```

---

#### `plot_time_series_by_model(df, metric, model_filter=None, title=None, ylabel=None, output_file=None)`
Plot time series of a metric across models/variants.

---

#### `plot_relative_change_vs_historical(df, metric, group_by='scenario', title=None, ylabel=None, output_file=None)`
Plot relative change from historical baseline over time.

**Parameters:**
- `group_by`: 'scenario' or 'model' to color lines by

---

#### `plot_scenario_model_heatmap(df, metric, year, title=None, output_file=None)`
Create heatmap comparing metric values across scenarios and models for a specific year.

---

## Helper Utilities

### Available Metrics

Common storm metrics available in the data:
- `total_storms` - All storms
- `tropical_storm` - Tropical storm category
- `category_1` through `category_5` - Hurricane categories
- `at_least_tropical_storm` - Tropical storm or stronger
- `at_least_hurricane` - Hurricane or stronger (Cat 1+)
- `hurricane_1_to_3` - Major hurricanes Cat 1-3
- `hurricane_4_plus` - Extreme hurricanes Cat 4-5

### Standard Grouping Columns

Common grouping columns used across functions:
- `model` - Climate model name (e.g., 'CMCC-ESM2', 'EC-Earth3')
- `variant` - Model variant (e.g., 'r1i1p1f1')
- `scenario` - Climate scenario (e.g., 'historical', 'ssp126', 'ssp245', 'ssp585')
- `basin` - Ocean basin (e.g., 'GL', 'NA', 'EP', 'WP', 'NI', 'SI', 'AU', 'SP')
- `time_period` - Time period (e.g., '1965-1969', '2010-2014')
- `draw` - Ensemble draw number
- `year` - Year

---

## Complete Example Script

Here's a complete example showing how to use these functions together:

```python
#!/usr/bin/env python3
"""
Example: Create storm impact visualizations
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
import idd_climate_models.constants as rfc

# Import visualization functions
from create_summary_visualizations import load_all_data, calculate_historical_baseline
from storm_timeseries_plots import plot_timeseries, prepare_combined_data

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("/ihme/homes/bcreiner/repos/idd-climate-models/outputs/storm_admin_summaries")
OUTPUT_DIR = Path("./outputs/my_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")
annual_all = load_all_data(DATA_DIR, file_type="annual_stats_all_countries")
basin_annual = pd.DataFrame()  # Or load basin-specific data if available

# ============================================================================
# EXAMPLE 1: Compare scenarios for specific model
# ============================================================================

print("\nCreating scenario comparison plot...")
fig = plot_timeseries(
    annual_all, basin_annual,
    metric='total_storms',
    model_variant='CMCC-ESM2/r1i1p1f1',
    scenario=['historical', 'ssp126', 'ssp245', 'ssp585'],
    basin='GL',
    year_range=(1980, 2100),
    save_path=OUTPUT_DIR / 'scenario_comparison.png'
)
print("✓ Saved scenario_comparison.png")

# ============================================================================
# EXAMPLE 2: Compare multiple metrics
# ============================================================================

print("\nCreating multi-metric plot...")
fig = plot_timeseries(
    annual_all, basin_annual,
    metric=['total_storms', 'category_4', 'category_5'],
    model_variant='CMCC-ESM2/r1i1p1f1',
    scenario='ssp585',
    basin='GL',
    nrows=1, ncols=3,
    save_path=OUTPUT_DIR / 'multi_metric.png'
)
print("✓ Saved multi_metric.png")

# ============================================================================
# EXAMPLE 3: Relative change with smoothing
# ============================================================================

print("\nCreating relative change plot...")
fig = plot_timeseries(
    annual_all, basin_annual,
    metric='total_storms',
    model_variant='CMCC-ESM2/r1i1p1f1',
    scenario=['ssp126', 'ssp585'],
    basin='GL',
    relative=True,
    smooth=True,
    smooth_frac=0.3,
    save_path=OUTPUT_DIR / 'relative_change_smooth.png'
)
print("✓ Saved relative_change_smooth.png")

print("\nDone! All visualizations saved to:", OUTPUT_DIR)
```

---

## File Processing Patterns

### Pattern 1: Load and combine storm-level data
```python
from pathlib import Path
import pandas as pd

DATA_DIR = Path("outputs/storm_admin_summaries")
storm_files = list(DATA_DIR.glob("storm_level_data_*.csv"))

dfs = []
for file in storm_files:
    # Parse filename: storm_level_data_{model}_{variant}_{scenario}.csv
    parts = file.stem.replace("storm_level_data_", "").split("_")
    scenario = parts[-1]
    variant = parts[-2]
    model = "_".join(parts[:-2])
    
    df = pd.read_csv(file, keep_default_na=False, na_values=[''])
    df['model'] = model
    df['variant'] = variant
    df['scenario'] = scenario
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
```

### Pattern 2: Load track files from TC risk output
```python
from pathlib import Path
import xarray as xr
from idd_climate_models.climate_file_functions import get_track_path
from argparse import Namespace

args = Namespace(
    data_source='cmip6',
    model='CMCC-ESM2',
    variant='r1i1p1f1',
    scenario='ssp126',
    time_period='2010-2014',
    basin='NA',
    draw=0,
    input_data_type='tc_risk',
    input_io_data_type='output',
    output_data_type='climada',
    output_io_data_type='input'
)

track_file = get_track_path(args)
ds = xr.open_dataset(track_file)
```

### Pattern 3: Calculate statistics with uncertainty
```python
# Group by year/scenario and calculate mean, lower, upper quantiles
stats = df.groupby(['year', 'scenario', 'basin'])['total_storms'].agg([
    ('total_storms_mean', 'mean'),
    ('total_storms_lower', lambda x: x.quantile(0.025)),
    ('total_storms_upper', lambda x: x.quantile(0.975))
]).reset_index()
```

---

## Tips for Creating New Scripts

1. **Start with existing patterns:** Use the complete example script as a template

2. **Use appropriate data loading:**
   - For pre-computed summaries: `load_all_data()`
   - For raw track files: `xr.open_dataset()` with `get_track_path()`
   - For time bins: Load `rfc.TIME_BINS_WIDE_DF_PATH`

3. **Think about dimensions:**
   - 0 unfixed variables → Type A plot (single line)
   - 1 unfixed variable → Type B plot (multiple lines)
   - 2+ unfixed variables → Type C or D plots (panels)

4. **Performance considerations:**
   - Use `count_all_storms_from_timebins.py` for fast storm counts
   - Use vectorized operations when processing spatial data
   - Use `create_storm_admin_impact_dataframe()` for efficient spatial joins

5. **Standard workflows:**
   - Load data → Filter/group → Calculate stats → Plot
   - Always include model/variant/scenario/basin columns
   - Use consistent column naming (e.g., `_mean`, `_lower`, `_upper` suffixes)

---

## Related Documentation

- **Pipeline Overview:** See [README_storm_admin_analysis.md](README_storm_admin_analysis.md)
- **Constants:** Check `src/idd_climate_models/constants.py` for basin definitions, scenarios, etc.
- **File Functions:** See `src/idd_climate_models/climate_file_functions.py` for path utilities
