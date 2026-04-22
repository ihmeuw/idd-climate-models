"""
CLIMADA summary aggregation script.

Takes Level 3 draw summaries and creates:
- Level 4: Basin (all draws) - Year × Month × Country × Severity × Draw
- Level 5: Basin summary - Mean/CI across draws
- Level 6: Global summary - Combine basins, handle multi-basin countries
- Level 7: Scenario summary - Basin × Year × Month × Severity for plotting

Run after all Level 1 tasks (run_climada_storm_analysis.py) complete.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import idd_climate_models.constants as rfc


def sanitize_attrs_for_netcdf(attrs: dict) -> dict:
    """Convert boolean attrs to int for netCDF compatibility."""
    return {k: (int(v) if isinstance(v, bool) else v) for k, v in attrs.items()}

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description="CLIMADA summary aggregation")
parser.add_argument("--model", required=True, help="Climate model name")
parser.add_argument("--variant", required=True, help="Model variant")
parser.add_argument("--scenario", required=True, help="Emissions scenario")
parser.add_argument("--data_source", default="cmip6", help="Data source")
parser.add_argument("--admin_level", type=int, default=0, help="Admin level")
parser.add_argument("--output_dir", default=None, help="Output directory (default: auto)")
parser.add_argument("--level", type=int, default=None, 
                    help="Run only this level (4-7). Default: run all")

args = parser.parse_args()

MODEL = args.model
VARIANT = args.variant
SCENARIO = args.scenario
DATA_SOURCE = args.data_source
ADMIN_LEVEL = args.admin_level
RUN_LEVEL = args.level  # None = run all, or specific level 4-7

def should_run(level: int) -> bool:
    """Check if a specific level should run."""
    return RUN_LEVEL is None or RUN_LEVEL == level

# Output directory
if args.output_dir:
    OUTPUT_DIR = Path(args.output_dir)
else:
    OUTPUT_DIR = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada_admin_summaries")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print(f"CLIMADA Summary Aggregation")
print(f"  Model: {MODEL}")
print(f"  Variant: {VARIANT}")
print(f"  Scenario: {SCENARIO}")
print(f"  Admin level: {ADMIN_LEVEL}")
print(f"  Level(s): {RUN_LEVEL if RUN_LEVEL else 'all (4-7)'}")
print(f"  Admin level: {ADMIN_LEVEL}")
print("=" * 80)

# ============================================================================
# FIND AND LOAD LEVEL 3 FILES
# ============================================================================

print("\nSearching for Level 3 draw summary files...")

# Level 3 files are stored in OUTPUT_DIR, organized by model/variant/scenario/time_period/basin
base_path = OUTPUT_DIR

# Find all time periods and basins for this model/variant/scenario
model_path = base_path / MODEL / VARIANT / SCENARIO

if not model_path.exists():
    print(f"ERROR: Path not found: {model_path}")
    sys.exit(1)

# Discover all zarr files
zarr_files = list(model_path.glob(f"*/*/draw_*_admin{ADMIN_LEVEL}_summary.zarr"))
print(f"Found {len(zarr_files)} draw summary files")

if not zarr_files:
    print("ERROR: No Level 3 files found")
    sys.exit(1)

# Organize by time_period/basin
files_by_location = {}
for f in zarr_files:
    # Path: base/model/variant/scenario/time_period/basin/draw_XXXX_adminX_summary.zarr
    basin = f.parent.name
    time_period = f.parent.parent.name
    key = (time_period, basin)
    
    if key not in files_by_location:
        files_by_location[key] = []
    files_by_location[key].append(f)

print(f"Found data for {len(files_by_location)} time_period/basin combinations:")
for (tp, basin), files in files_by_location.items():
    print(f"  {tp} / {basin}: {len(files)} draws")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_combined_categories(ds: xr.Dataset) -> xr.Dataset:
    """Add combined category variables to dataset."""
    # Get the storm counts
    counts = ds['storm_count']
    
    # Combined categories
    ds['at_least_tropical_storm'] = counts.sum(dim='severity')
    
    ds['at_least_hurricane'] = counts.sel(
        severity=['category_1', 'category_2', 'category_3', 'category_4', 'category_5']
    ).sum(dim='severity')
    
    ds['hurricane_1_to_3'] = counts.sel(
        severity=['category_1', 'category_2', 'category_3']
    ).sum(dim='severity')
    
    ds['hurricane_4_plus'] = counts.sel(
        severity=['category_4', 'category_5']
    ).sum(dim='severity')
    
    return ds


def calculate_stats(data: xr.DataArray, dim: str = 'draw') -> Dict[str, xr.DataArray]:
    """Calculate mean, lower (2.5%), upper (97.5%) across a dimension."""
    lower = data.quantile(0.025, dim=dim)
    upper = data.quantile(0.975, dim=dim)
    
    # Drop 'quantile' coordinate to avoid conflicts when combining
    if 'quantile' in lower.coords:
        lower = lower.drop_vars('quantile')
    if 'quantile' in upper.coords:
        upper = upper.drop_vars('quantile')
    
    return {
        'mean': data.mean(dim=dim),
        'lower': lower,
        'upper': upper,
    }


def combine_draws_for_basin(
    zarr_files: List[Path],
    time_period: str,
    basin: str
) -> xr.Dataset:
    """
    Combine all draw files for a basin into Level 4 dataset.
    
    Returns dataset with dimensions: year × month × country × severity × draw
    """
    datasets = []
    draws = []
    
    for f in sorted(zarr_files):
        ds = xr.open_zarr(f)
        draw = ds.attrs.get('draw', int(f.stem.split('_')[1]))
        draws.append(draw)
        datasets.append(ds)
    
    # Combine along new draw dimension
    combined = xr.concat(datasets, dim='draw')
    combined = combined.assign_coords(draw=draws)
    
    # Add metadata
    combined.attrs['time_period'] = time_period
    combined.attrs['basin'] = basin
    combined.attrs['n_draws'] = len(draws)
    
    return combined


# ============================================================================
# LEVEL 4: COMBINE ALL DRAWS PER BASIN
# ============================================================================

basin_datasets = {}

if should_run(4):
    print("\n" + "=" * 80)
    print("Creating Level 4: Basin (all draws)")
    print("=" * 80)

for (time_period, basin), files in files_by_location.items():
    if should_run(4):
        print(f"\nProcessing {time_period} / {basin}...")
    
    # Combine draws
    basin_ds = combine_draws_for_basin(files, time_period, basin)
    if should_run(4):
        print(f"  Combined {len(files)} draws")
        print(f"  Shape: {dict(basin_ds.dims)}")
    
        # Save Level 4
        output_path = (OUTPUT_DIR / f"level4_basin_all_draws_{MODEL}_{VARIANT}_{SCENARIO}_"
                       f"{time_period}_{basin}_admin{ADMIN_LEVEL}.zarr")
        basin_ds.to_zarr(output_path, mode='w')
        print(f"  ✓ Saved: {output_path.name}")
    
    basin_datasets[(time_period, basin)] = basin_ds

# ============================================================================
# LEVEL 5: BASIN SUMMARY (STATS ACROSS DRAWS)
# ============================================================================

basin_summaries = {}

# Only process Level 5+ if we need it (Level 5, 6, 7, or all)
if RUN_LEVEL is None or RUN_LEVEL >= 5:
    if should_run(5):
        print("\n" + "=" * 80)
        print("Creating Level 5: Basin summary (mean/CI)")
        print("=" * 80)

    for (time_period, basin), ds in basin_datasets.items():
        if should_run(5):
            print(f"\nProcessing {time_period} / {basin}...")
        
        # Add combined categories first
        ds = calculate_combined_categories(ds)
        
        # Variables to summarize
        variables = ['storm_count', 'at_least_tropical_storm', 'at_least_hurricane',
                     'hurricane_1_to_3', 'hurricane_4_plus']
        
        # Calculate stats for each variable
        data_vars = {}
        for var in variables:
            if var in ds:
                stats = calculate_stats(ds[var], dim='draw')
                data_vars[f'{var}_mean'] = stats['mean']
                data_vars[f'{var}_lower'] = stats['lower']
                data_vars[f'{var}_upper'] = stats['upper']
        
        # Create summary dataset
        summary_ds = xr.Dataset(
            data_vars,
            attrs=sanitize_attrs_for_netcdf({
                **ds.attrs,
                'description': 'Summary statistics across draws (mean, 2.5%, 97.5%)',
            })
        )
        
        # Copy coordinate attributes
        if 'country_name' in ds.coords:
            summary_ds = summary_ds.assign_coords(country_name=ds.coords['country_name'])
        
        if should_run(5):
            print(f"  Shape: {dict(summary_ds.dims)}")
        
            # Save Level 5
            output_path = (OUTPUT_DIR / f"level5_basin_summary_{MODEL}_{VARIANT}_{SCENARIO}_"
                           f"{time_period}_{basin}_admin{ADMIN_LEVEL}.nc")
            summary_ds.to_netcdf(output_path)
            print(f"  ✓ Saved: {output_path.name}")
        
        basin_summaries[(time_period, basin)] = summary_ds

# ============================================================================
# LEVEL 6: GLOBAL SUMMARY (COMBINE BASINS)
# ============================================================================

# Group by time period (needed for Level 6 and 7)
time_periods = sorted(set(tp for tp, _ in basin_summaries.keys())) if basin_summaries else []

if should_run(6):
    print("\n" + "=" * 80)
    print("Creating Level 6: Global summary (combine basins)")
    print("=" * 80)

    for time_period in time_periods:
        print(f"\nProcessing {time_period}...")
        
        # Get all basins for this time period
        tp_basins = {basin: ds for (tp, basin), ds in basin_summaries.items() if tp == time_period}
        
        if len(tp_basins) == 0:
            continue
        
        print(f"  Found {len(tp_basins)} basins: {list(tp_basins.keys())}")
        
        # Get union of all countries across basins
        all_countries = set()
        for ds in tp_basins.values():
            all_countries.update(ds.country.values)
        all_countries = sorted(all_countries)
        
        print(f"  Total unique countries: {len(all_countries)}")
        
        # For global summary, we sum counts across basins for countries that appear in multiple basins
        # This is correct because storms are unique within each basin
        
        # Initialize global arrays
        sample_ds = list(tp_basins.values())[0]
        years = sample_ds.year.values
        months = sample_ds.month.values
        severities = sample_ds.severity.values if 'severity' in sample_ds.dims else None
        
        # Variables to aggregate
        mean_vars = [v for v in sample_ds.data_vars if v.endswith('_mean')]
        
        global_data = {}
        for var in mean_vars:
            base_var = var.replace('_mean', '')
            
            # Initialize with zeros
            if severities is not None and 'severity' in sample_ds[var].dims:
                shape = (len(years), len(months), len(all_countries), len(severities))
                dims = ['year', 'month', 'country', 'severity']
            else:
                shape = (len(years), len(months), len(all_countries))
                dims = ['year', 'month', 'country']
            
            global_mean = np.zeros(shape, dtype=np.float32)
            global_lower = np.zeros(shape, dtype=np.float32)
            global_upper = np.zeros(shape, dtype=np.float32)
            
            # Sum across basins
            for basin, ds in tp_basins.items():
                basin_countries = ds.country.values
                for i, country in enumerate(all_countries):
                    if country in basin_countries:
                        country_idx = list(basin_countries).index(country)
                        
                        if severities is not None and 'severity' in ds[var].dims:
                            global_mean[:, :, i, :] += ds[var].values[:, :, country_idx, :]
                            global_lower[:, :, i, :] += ds[f'{base_var}_lower'].values[:, :, country_idx, :]
                            global_upper[:, :, i, :] += ds[f'{base_var}_upper'].values[:, :, country_idx, :]
                        else:
                            global_mean[:, :, i] += ds[var].values[:, :, country_idx]
                            global_lower[:, :, i] += ds[f'{base_var}_lower'].values[:, :, country_idx]
                            global_upper[:, :, i] += ds[f'{base_var}_upper'].values[:, :, country_idx]
            
            global_data[f'{base_var}_mean'] = (dims, global_mean)
            global_data[f'{base_var}_lower'] = (dims, global_lower)
            global_data[f'{base_var}_upper'] = (dims, global_upper)
        
        # Build coordinates
        coords = {
            'year': years,
            'month': months,
            'country': all_countries,
        }
        if severities is not None:
            coords['severity'] = severities
        
        # Create global dataset
        global_ds = xr.Dataset(
            global_data,
            coords=coords,
            attrs={
                'model': MODEL,
                'variant': VARIANT,
                'scenario': SCENARIO,
                'time_period': time_period,
                'basin': 'GLOBAL',
                'basins_combined': list(tp_basins.keys()),
                'admin_level': ADMIN_LEVEL,
                'description': 'Global summary combining all basins (summed where countries span multiple basins)',
            }
        )
        
        # Save Level 6
        output_path = (OUTPUT_DIR / f"level6_global_summary_{MODEL}_{VARIANT}_{SCENARIO}_"
                       f"{time_period}_admin{ADMIN_LEVEL}.nc")
        global_ds.to_netcdf(output_path)
        print(f"  ✓ Saved: {output_path.name}")

# ============================================================================
# LEVEL 7: SCENARIO SUMMARY (FOR PLOTTING)
# ============================================================================

if should_run(7):
    print("\n" + "=" * 80)
    print("Creating Level 7: Scenario summary (basin × year × severity)")
    print("=" * 80)

    # Aggregate across countries within each basin, keeping year and severity
    for (time_period, basin), ds in basin_summaries.items():
        print(f"\nProcessing {time_period} / {basin}...")
        
        # Sum across countries (this gives basin totals)
        scenario_ds = ds.sum(dim='country')
        
        # Update attributes
        scenario_ds.attrs['aggregation'] = 'summed across countries'
        
        # Also compute annual sums (sum across months)
        annual_ds = scenario_ds.sum(dim='month')
        annual_ds.attrs['aggregation'] = 'summed across countries and months (annual)'
        
        # Save monthly version
        output_path = (OUTPUT_DIR / f"level7_scenario_monthly_{MODEL}_{VARIANT}_{SCENARIO}_"
                       f"{time_period}_{basin}_admin{ADMIN_LEVEL}.nc")
        scenario_ds.to_netcdf(output_path)
        print(f"  ✓ Saved monthly: {output_path.name}")
        
        # Save annual version
        output_path = (OUTPUT_DIR / f"level7_scenario_annual_{MODEL}_{VARIANT}_{SCENARIO}_"
                       f"{time_period}_{basin}_admin{ADMIN_LEVEL}.nc")
        annual_ds.to_netcdf(output_path)
        print(f"  ✓ Saved annual: {output_path.name}")

    # Also create global scenario summary (requires Level 6 to exist)
    print("\nCreating global scenario summaries...")

    for time_period in time_periods:
        # Load global summary
        global_path = (OUTPUT_DIR / f"level6_global_summary_{MODEL}_{VARIANT}_{SCENARIO}_"
                       f"{time_period}_admin{ADMIN_LEVEL}.nc")
        
        if not global_path.exists():
            print(f"  Skipping {time_period} - Level 6 not found (run --level 6 first)")
            continue
        
        global_ds = xr.open_dataset(global_path)
        
        # Sum across countries
        scenario_ds = global_ds.sum(dim='country')
        
        # Monthly version
        output_path = (OUTPUT_DIR / f"level7_scenario_monthly_{MODEL}_{VARIANT}_{SCENARIO}_"
                       f"{time_period}_GLOBAL_admin{ADMIN_LEVEL}.nc")
        scenario_ds.to_netcdf(output_path)
        print(f"  ✓ Saved global monthly: {output_path.name}")
        
        # Annual version
        annual_ds = scenario_ds.sum(dim='month')
        output_path = (OUTPUT_DIR / f"level7_scenario_annual_{MODEL}_{VARIANT}_{SCENARIO}_"
                       f"{time_period}_GLOBAL_admin{ADMIN_LEVEL}.nc")
        annual_ds.to_netcdf(output_path)
        print(f"  ✓ Saved global annual: {output_path.name}")

# ============================================================================
# ALSO SAVE CSV VERSIONS FOR EASY INSPECTION (runs with Level 7 or all)
# ============================================================================

if should_run(7) or RUN_LEVEL is None:
    print("\n" + "=" * 80)
    print("Exporting CSV versions for compatibility")
    print("=" * 80)

    # Create basin_annual_stats CSV (like TC Risk format)
    csv_rows = []
    for (time_period, basin), ds in basin_summaries.items():
        # Get annual sums (sum over months, countries, and severity)
        annual = ds.sum(dim=['month', 'country'])
        if 'severity' in annual.dims:
            annual = annual.sum(dim='severity')
        
        for year in annual.year.values:
            row = {
                'model': MODEL,
                'variant': VARIANT,
                'scenario': SCENARIO,
                'time_period': time_period,
                'basin': basin,
                'year': int(year),
            }
            
            # Add each metric
            for var in annual.data_vars:
                value = float(annual[var].sel(year=year).values)
                row[var] = value
            
            csv_rows.append(row)

    if csv_rows:
        csv_df = pd.DataFrame(csv_rows)
        csv_path = OUTPUT_DIR / f"basin_annual_stats_{MODEL}_{VARIANT}_{SCENARIO}.csv"
        csv_df.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path.name}")

    # Create annual_stats_all_countries CSV (global)
    csv_rows = []
    for time_period in time_periods:
        global_path = (OUTPUT_DIR / f"level6_global_summary_{MODEL}_{VARIANT}_{SCENARIO}_"
                       f"{time_period}_admin{ADMIN_LEVEL}.nc")
        
        if not global_path.exists():
            continue
        
        global_ds = xr.open_dataset(global_path)
        
        # Annual sums across months, countries, and severity
        annual = global_ds.sum(dim=['month', 'country'])
        if 'severity' in annual.dims:
            annual = annual.sum(dim='severity')
        
        for year in annual.year.values:
            row = {
                'model': MODEL,
                'variant': VARIANT,
                'scenario': SCENARIO,
                'time_period': time_period,
                'year': int(year),
            }
            
            for var in annual.data_vars:
                value = float(annual[var].sel(year=year).values)
                row[var] = value
            
            csv_rows.append(row)

    if csv_rows:
        csv_df = pd.DataFrame(csv_rows)
        csv_path = OUTPUT_DIR / f"annual_stats_all_countries_{MODEL}_{VARIANT}_{SCENARIO}.csv"
        csv_df.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path.name}")

print("\n" + "=" * 80)
level_msg = f"Level {RUN_LEVEL}" if RUN_LEVEL else "all levels (4-7)"
print(f"✅ Summary aggregation complete: {level_msg}")
print(f"   Model: {MODEL}/{VARIANT}/{SCENARIO}")
print(f"   Output directory: {OUTPUT_DIR}")
print("=" * 80)
