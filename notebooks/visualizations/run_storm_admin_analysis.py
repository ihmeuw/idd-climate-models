"""
Worker script for storm-admin0 impact analysis.

Processes all draws assigned to a specific task_id and creates
comprehensive storm-admin0 impact dataframes.
"""

import argparse
import pandas as pd
import geopandas as gpd
import xarray as xr
import numpy as np
from pathlib import Path

import idd_climate_models.constants as rfc
from idd_climate_models.climate_file_functions import get_track_path
from idd_spatial_analysis.constants import ADMIN_SHP_FILENAME_TEMPLATE

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description="Storm-Admin0 impact analysis worker")
parser.add_argument("--data_source", required=True, help="Data source (e.g., cmip6)")
parser.add_argument("--task_id", required=True, type=int, help="Task ID from assignments file")
parser.add_argument("--output_dir", required=True, help="Output directory for results")

args = parser.parse_args()

DATA_SOURCE = args.data_source
TASK_ID = args.task_id
OUTPUT_DIR = Path(args.output_dir)

# ============================================================================
# LOAD TASK ASSIGNMENTS
# ============================================================================

print(f"=" * 80)
print(f"Task {TASK_ID}: Storm-Admin0 Impact Analysis")
print(f"=" * 80)

task_assignments_file = rfc.CLIMADA_INPUT_PATH / DATA_SOURCE / "level_4_task_assignments.csv"
df_assignments = pd.read_csv(task_assignments_file, keep_default_na=False)

# Get assignments for this task
task_rows = df_assignments[df_assignments['task_id'] == TASK_ID]

if task_rows.empty:
    print(f"ERROR: No assignments found for task_id {TASK_ID}")
    exit(1)

# Extract task details (same for all rows in this task)
MODEL = task_rows['model'].iloc[0]
VARIANT = task_rows['variant'].iloc[0]
SCENARIO = task_rows['scenario'].iloc[0]
TIME_PERIOD = task_rows['time_period'].iloc[0]
BASIN = task_rows['basin'].iloc[0]
DRAWS = sorted(task_rows['draw'].unique())

print(f"\nTask details:")
print(f"  Model: {MODEL}")
print(f"  Variant: {VARIANT}")
print(f"  Scenario: {SCENARIO}")
print(f"  Time period: {TIME_PERIOD}")
print(f"  Basin: {BASIN}")
print(f"  Draws: {len(DRAWS)} ({min(DRAWS)}-{max(DRAWS)})")

# ============================================================================
# LOAD ADMIN SHAPEFILE
# ============================================================================

print(f"\nLoading admin0 shapefile...")
ADMIN0_SHP_FILENAME = ADMIN_SHP_FILENAME_TEMPLATE.format(admin_level=0, simple_suffix="_simplified")
admin0_gdf = gpd.read_file(ADMIN0_SHP_FILENAME)
print(f"✓ Loaded {len(admin0_gdf)} admin 0 regions")

# Get basin bounds for clipping
basin_info = rfc.basin_dict.get(BASIN)
if basin_info is None:
    print(f"ERROR: Basin {BASIN} not found in basin_dict")
    exit(1)

basin_bounds_str = basin_info['basin_bounds']

def parse_basin_coordinate(coord_str):
    """Convert coordinate string like '260E' or '45S' to numeric value."""
    value = float(coord_str[:-1])
    direction = coord_str[-1]
    if direction in ['W', 'S']:
        value = -value
    if direction in ['E', 'W'] and value > 180:
        value = value - 360
    return value

basin_lon_min = parse_basin_coordinate(basin_bounds_str[0])
basin_lat_min = parse_basin_coordinate(basin_bounds_str[1])
basin_lon_max = parse_basin_coordinate(basin_bounds_str[2])
basin_lat_max = parse_basin_coordinate(basin_bounds_str[3])

from shapely.geometry import box
bbox_polygon = box(basin_lon_min, basin_lat_min, basin_lon_max, basin_lat_max)
admin0_clipped = admin0_gdf.clip(bbox_polygon)
print(f"✓ Clipped to {len(admin0_clipped)} admin 0 regions in basin bounds")

# ============================================================================
# DEFINE ANALYSIS FUNCTION
# ============================================================================

def create_storm_admin_impact_dataframe(ds, admin0_gdf, basin_bounds_polygon=None):
    """
    Create a dataframe of storm impacts on admin 0 regions.
    
    OPTIMIZED VERSION using vectorized operations and spatial join.
    """
    # Clip admin boundaries to basin if needed
    if basin_bounds_polygon is not None:
        admin_clipped = admin0_gdf.clip(basin_bounds_polygon)
    else:
        admin_clipped = admin0_gdf
    
    # Ensure admin GeoDataFrame has necessary columns
    admin_cols = ['ADM0_CODE', 'ADM0_NAME', 'loc_id', 'geometry']
    admin_clipped = admin_clipped[[col for col in admin_cols if col in admin_clipped.columns]]
    
    # XARRAY TRICK 1: Stack all dimensions into flat arrays at once
    lon_flat = ds['lon'].values.ravel()
    lat_flat = ds['lat'].values.ravel()
    wind_flat = ds['max_sustained_wind'].values.ravel()
    
    # Create arrays for storm indices (repeat for each time step)
    n_storms = ds.sizes['n_trk']
    n_time = ds.sizes['time']
    storm_indices = np.repeat(np.arange(n_storms), n_time)
    
    # Get storm metadata (year and month per storm)
    storm_years = ds['tc_years'].values
    storm_months = ds['tc_month'].values
    
    # Expand years and months to match flat structure
    years_flat = np.repeat(storm_years, n_time)
    months_flat = np.repeat(storm_months, n_time)
    
    # Remove NaN values (VECTORIZED)
    valid = ~(np.isnan(lon_flat) | np.isnan(lat_flat) | np.isnan(wind_flat))
    
    # XARRAY TRICK 2: Create GeoDataFrame from all points at once
    points_gdf = gpd.GeoDataFrame({
        'storm_track': storm_indices[valid],
        'year': years_flat[valid].astype(int),
        'month': months_flat[valid].astype(int),
        'wind_speed': wind_flat[valid],
        'geometry': gpd.points_from_xy(lon_flat[valid], lat_flat[valid])
    }, crs='EPSG:4326')
    
    # GEOSPATIAL TRICK: Use spatial join with spatial index (rtree)
    joined = gpd.sjoin(points_gdf, admin_clipped, how='inner', predicate='within')
    
    # Group by storm and admin region, find max windspeed
    result = joined.groupby(['storm_track', 'ADM0_CODE', 'ADM0_NAME', 'loc_id', 'year', 'month'], 
                            dropna=False).agg({
        'wind_speed': 'max'
    }).reset_index()
    
    # Rename and add storm category
    result.rename(columns={'wind_speed': 'max_wind_speed'}, inplace=True)
    result['storm_category'] = result['max_wind_speed'].apply(rfc.classify_storm)
    
    # Reorder columns
    cols_order = ['storm_track', 'year', 'month', 'ADM0_CODE', 'ADM0_NAME', 
                  'loc_id', 'max_wind_speed', 'storm_category']
    result = result[[col for col in cols_order if col in result.columns]]
    
    return result

# ============================================================================
# PROCESS DRAWS
# ============================================================================

print(f"\nProcessing {len(DRAWS)} draws...")

all_results = []

for i, draw in enumerate(DRAWS, 1):
    print(f"\n[{i}/{len(DRAWS)}] Processing draw {draw}...")
    
    # Construct file path
    from argparse import Namespace
    args_for_path = Namespace(
        data_source='cmip6',
        model=MODEL,
        variant=VARIANT,
        scenario=SCENARIO,
        time_period=TIME_PERIOD,
        basin=BASIN,
        draw=draw,
        input_data_type='tc_risk',
        input_io_data_type='output',
        output_data_type='climada',
        output_io_data_type='input'
    )
    
    track_path = get_track_path(args_for_path, source=False, extension=".nc")
    
    # Store the directory for output (first draw sets this)
    if i == 1:
        output_dir_for_task = track_path.parent
    
    if not track_path.exists():
        print(f"  ⚠️  File not found: {track_path}")
        continue
    
    # Load dataset
    ds = xr.open_dataset(track_path)
    
    # Run analysis
    try:
        df_result = create_storm_admin_impact_dataframe(ds, admin0_gdf, bbox_polygon)
        
        # Add draw information
        df_result['draw'] = draw
        df_result['model'] = MODEL
        df_result['variant'] = VARIANT
        df_result['scenario'] = SCENARIO
        df_result['time_period'] = TIME_PERIOD
        df_result['basin'] = BASIN
        
        all_results.append(df_result)
        print(f"  ✓ Found {len(df_result)} storm-admin0 impacts")
        
    except Exception as e:
        print(f"  ❌ Error processing draw {draw}: {e}")
        continue
    
    finally:
        ds.close()

# ============================================================================
# COMBINE AND SAVE RESULTS
# ============================================================================

if not all_results:
    print("\n⚠️  No results to save")
    exit(0)

print(f"\nCombining results from {len(all_results)} draws...")
final_df = pd.concat(all_results, ignore_index=True)

# Reorder columns for readability
cols_order = [
    'model', 'variant', 'scenario', 'time_period', 'basin', 'draw',
    'storm_track', 'year', 'month', 
    'ADM0_CODE', 'ADM0_NAME', 'loc_id',
    'max_wind_speed', 'storm_category'
]
final_df = final_df[[col for col in cols_order if col in final_df.columns]]

# Sort by draw and storm_track
final_df = final_df.sort_values(['draw', 'storm_track'])

# Create output filename - save in same directory as .nc files
output_file = output_dir_for_task / f"storm_admin_impacts_task_{TASK_ID:04d}.csv"

# Save
final_df.to_csv(output_file, index=False)

print(f"\n✅ SUCCESS")
print(f"   Output: {output_file}")
print(f"   Total rows: {len(final_df)}")
print(f"   Draws processed: {final_df['draw'].nunique()}")
print(f"   Storm-admin0 impacts: {len(final_df)}")

# Print summary statistics
print(f"\nSummary by category:")
category_counts = final_df['storm_category'].value_counts()
for cat, count in category_counts.items():
    print(f"  {cat}: {count}")
