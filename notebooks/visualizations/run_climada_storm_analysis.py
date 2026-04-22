"""
Worker script for CLIMADA intensity raster → admin impact analysis.

Processes all draws assigned to a specific task_id:
1. Load intensity zarr for each storm
2. Overlay with admin shapefile to get max wind per country
3. Create Level 3 summary (Year × Month × Country × Severity) as zarr

Output: zarr file per draw with storm counts by year/month/country/severity
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio import features
from affine import Affine
from shapely.geometry import box

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import idd_climate_models.constants as rfc
from idd_climate_models.storm_data import ClimadaBasinData, ClimadaStormData
from idd_spatial_analysis.constants import ADMIN_SHP_FILENAME_TEMPLATE

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description="CLIMADA storm-admin impact analysis worker")
parser.add_argument("--data_source", required=True, help="Data source (e.g., cmip6)")
parser.add_argument("--task_id", required=True, type=int, help="Task ID from assignments file")
parser.add_argument("--output_dir", default="/mnt/team/rapidresponse/pub/tropical-storms/climada_admin_summaries",
                    help="Output directory for results")
parser.add_argument("--admin_level", type=int, default=0, help="Admin level (0, 1, or 2)")
parser.add_argument("--simplified", action="store_true", default=True, 
                    help="Use simplified shapefile")
parser.add_argument("--save_storm_level", action="store_true", default=False,
                    help="Save storm-level data (Level 1/2)")

args = parser.parse_args()

DATA_SOURCE = args.data_source
TASK_ID = args.task_id
OUTPUT_DIR = Path(args.output_dir)
ADMIN_LEVEL = args.admin_level
USE_SIMPLIFIED = args.simplified
SAVE_STORM_LEVEL = args.save_storm_level

# ============================================================================
# LOAD TASK ASSIGNMENTS
# ============================================================================

print("=" * 80)
print(f"Task {TASK_ID}: CLIMADA Storm-Admin{ADMIN_LEVEL} Impact Analysis")
print("=" * 80)

task_assignments_file = rfc.CLIMADA_INPUT_PATH / DATA_SOURCE / "level_4_task_assignments.csv"
df_assignments = pd.read_csv(task_assignments_file, keep_default_na=False)

# Get assignments for this task
task_rows = df_assignments[df_assignments['task_id'] == TASK_ID]

if task_rows.empty:
    print(f"ERROR: No assignments found for task_id {TASK_ID}")
    sys.exit(1)

# Extract task details
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
print(f"  Admin level: {ADMIN_LEVEL}")
print(f"  Simplified shapefile: {USE_SIMPLIFIED}")

# ============================================================================
# LOAD ADMIN SHAPEFILE
# ============================================================================

print(f"\nLoading admin{ADMIN_LEVEL} shapefile...")
simple_suffix = "_simplified" if USE_SIMPLIFIED else ""
admin_shp_filename = ADMIN_SHP_FILENAME_TEMPLATE.format(
    admin_level=ADMIN_LEVEL, 
    simple_suffix=simple_suffix
)
admin_gdf = gpd.read_file(admin_shp_filename)
print(f"✓ Loaded {len(admin_gdf)} admin {ADMIN_LEVEL} regions")

# Get basin bounds for clipping
basin_info = rfc.basin_dict.get(BASIN)
if basin_info is None:
    print(f"ERROR: Basin {BASIN} not found in basin_dict")
    sys.exit(1)

basin_bounds_str = basin_info['basin_bounds']


def parse_basin_coordinate(coord_str: str) -> float:
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

bbox_polygon = box(basin_lon_min, basin_lat_min, basin_lon_max, basin_lat_max)
admin_clipped = admin_gdf.clip(bbox_polygon)
print(f"✓ Clipped to {len(admin_clipped)} admin {ADMIN_LEVEL} regions in basin bounds")

# Get the relevant admin code column name
if ADMIN_LEVEL == 0:
    admin_code_col = 'ADM0_CODE'
    admin_name_col = 'ADM0_NAME'
elif ADMIN_LEVEL == 1:
    admin_code_col = 'ADM1_CODE'
    admin_name_col = 'ADM1_NAME'
else:
    admin_code_col = 'ADM2_CODE'
    admin_name_col = 'ADM2_NAME'

# Build list of all countries in basin (for complete grid)
all_countries_in_basin = admin_clipped[[admin_code_col, admin_name_col]].drop_duplicates()
all_countries_in_basin = all_countries_in_basin.rename(columns={
    admin_code_col: 'country_code',
    admin_name_col: 'country_name'
})
print(f"✓ Found {len(all_countries_in_basin)} unique admin regions in basin")

# ============================================================================
# DEFINE ANALYSIS FUNCTIONS
# ============================================================================

# Severity categories (matching constants.py)
SEVERITY_CATEGORIES = [
    'tropical_depression', 'tropical_storm', 
    'category_1', 'category_2', 'category_3', 'category_4', 'category_5'
]


def rasterize_admin_regions(
    admin_gdf: gpd.GeoDataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    admin_code_col: str
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Rasterize admin regions to match intensity grid.
    
    Returns:
        raster: 2D array with admin codes as pixel values
        code_map: Dict mapping raster values to admin codes
    """
    # Create affine transform from coordinates
    lon_res = abs(lon[1] - lon[0])
    lat_res = abs(lat[1] - lat[0])
    
    # Determine if lat is increasing or decreasing
    lat_increasing = lat[1] > lat[0]
    
    if lat_increasing:
        transform = Affine.translation(lon.min() - lon_res/2, lat.min() - lat_res/2) * \
                    Affine.scale(lon_res, lat_res)
    else:
        transform = Affine.translation(lon.min() - lon_res/2, lat.max() + lat_res/2) * \
                    Affine.scale(lon_res, -lat_res)
    
    # Create shapes for rasterization
    # Assign unique integer IDs to each admin region
    admin_gdf = admin_gdf.copy()
    admin_gdf['raster_id'] = range(1, len(admin_gdf) + 1)
    
    shapes = [(geom, raster_id) for geom, raster_id in 
              zip(admin_gdf.geometry, admin_gdf['raster_id'])]
    
    # Rasterize
    out_shape = (len(lat), len(lon))
    raster = features.rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.int32
    )
    
    # Create mapping from raster_id to admin_code
    code_map = dict(zip(admin_gdf['raster_id'], admin_gdf[admin_code_col]))
    
    return raster, code_map


def get_max_wind_per_admin(
    intensity_da: xr.DataArray,
    admin_raster: np.ndarray,
    code_map: Dict[int, str]
) -> Dict[str, float]:
    """
    Get maximum wind speed per admin region from intensity raster.
    
    Returns:
        Dict mapping admin_code to max wind speed
    """
    intensity_values = intensity_da.values
    
    # Handle NaN values
    intensity_values = np.nan_to_num(intensity_values, nan=0.0)
    
    results = {}
    for raster_id, admin_code in code_map.items():
        mask = admin_raster == raster_id
        if mask.any():
            max_wind = intensity_values[mask].max()
            if max_wind > 0:
                results[admin_code] = float(max_wind)
    
    return results


def classify_wind_speed(wind_speed: float) -> str:
    """Classify wind speed into severity category."""
    return rfc.classify_storm(wind_speed) or 'none'


def process_storm(
    storm: ClimadaStormData,
    admin_raster: np.ndarray,
    code_map: Dict[int, str],
    admin_names: Dict[str, str]
) -> List[Dict]:
    """
    Process a single storm and return list of country impacts.
    
    Returns:
        List of dicts with storm, country, wind, category info
    """
    try:
        intensity_da = storm.get_intensity_array()
        max_winds = get_max_wind_per_admin(intensity_da, admin_raster, code_map)
        
        if not max_winds:
            return []
        
        results = []
        for admin_code, max_wind in max_winds.items():
            category = classify_wind_speed(max_wind)
            if category == 'none' or category == 'tropical_depression':
                continue  # Skip below tropical storm
            
            results.append({
                'storm_name': storm.storm_name,
                'storm_id': storm.storm_id,
                'year': storm.year,
                'month': storm.start_date.month if storm.start_date else None,
                'country_code': admin_code,
                'country_name': admin_names.get(admin_code, ''),
                'max_wind_speed': max_wind,
                'category': category,
            })
        
        return results
    
    except Exception as e:
        print(f"    ⚠️ Error processing {storm.storm_name}: {e}")
        return []


def create_summary_dataset(
    storm_impacts: List[Dict],
    all_countries: pd.DataFrame,
    time_period: str,
    draw: int,
    metadata: Dict
) -> xr.Dataset:
    """
    Create Level 3 summary dataset with complete grid.
    
    Dimensions: year × month × country × severity
    """
    # Parse time period
    start_year, end_year = map(int, time_period.split('-'))
    years = list(range(start_year, end_year + 1))
    months = list(range(1, 13))
    countries = sorted(all_countries['country_code'].unique())
    severities = ['tropical_storm', 'category_1', 'category_2', 'category_3', 
                  'category_4', 'category_5']
    
    # Initialize count array with zeros
    shape = (len(years), len(months), len(countries), len(severities))
    counts = np.zeros(shape, dtype=np.int16)
    
    # Create mappings for fast indexing
    year_idx = {y: i for i, y in enumerate(years)}
    month_idx = {m: i for i, m in enumerate(months)}
    country_idx = {c: i for i, c in enumerate(countries)}
    severity_idx = {s: i for i, s in enumerate(severities)}
    
    # Count storms (each storm counted once at its max category per country)
    for impact in storm_impacts:
        y = impact['year']
        m = impact['month']
        c = impact['country_code']
        s = impact['category']
        
        if y in year_idx and m in month_idx and c in country_idx and s in severity_idx:
            counts[year_idx[y], month_idx[m], country_idx[c], severity_idx[s]] += 1
    
    # Create country name mapping
    country_names = all_countries.set_index('country_code')['country_name'].to_dict()
    country_name_array = [country_names.get(c, '') for c in countries]
    
    # Create dataset
    ds = xr.Dataset(
        {
            'storm_count': (['year', 'month', 'country', 'severity'], counts),
        },
        coords={
            'year': years,
            'month': months,
            'country': countries,
            'country_name': ('country', country_name_array),
            'severity': severities,
        },
        attrs={
            'model': metadata['model'],
            'variant': metadata['variant'],
            'scenario': metadata['scenario'],
            'time_period': metadata['time_period'],
            'basin': metadata['basin'],
            'draw': draw,
            'admin_level': metadata['admin_level'],
            'simplified_shapefile': metadata['simplified'],
            'description': 'Storm counts by year, month, country, and severity category',
            'severity_categories': 'Saffir-Simpson scale: tropical_storm (17-33 m/s), category_1 (33-43), category_2 (43-50), category_3 (50-58), category_4 (58-70), category_5 (70+)',
        }
    )
    
    return ds


# ============================================================================
# PROCESS DRAWS
# ============================================================================

print(f"\nProcessing {len(DRAWS)} draws...")

# Pre-compute admin raster (same for all draws/storms in this basin)
print("\nPre-computing admin raster...")

# Use first draw to get coordinate grid
sample_basin = ClimadaBasinData(
    model=MODEL,
    variant=VARIANT,
    scenario=SCENARIO,
    time_period=TIME_PERIOD,
    basin=BASIN,
    draw=DRAWS[0],
)

sample_storms = sample_basin.list_storms()
if not sample_storms:
    print("ERROR: No storms found in sample basin")
    sys.exit(1)

sample_storm = sample_basin.get_storm(sample_storms[0])
sample_intensity = sample_storm.get_intensity_array()
lon = sample_intensity.lon.values
lat = sample_intensity.lat.values
sample_storm.close()

print(f"  Grid size: {len(lat)} × {len(lon)}")

# Rasterize admin regions
admin_raster, code_map = rasterize_admin_regions(admin_clipped, lon, lat, admin_code_col)
print(f"  ✓ Rasterized {len(code_map)} admin regions")

# Create admin name mapping
admin_names = dict(zip(admin_clipped[admin_code_col], admin_clipped[admin_name_col]))

# Metadata for output files
metadata = {
    'model': MODEL,
    'variant': VARIANT,
    'scenario': SCENARIO,
    'time_period': TIME_PERIOD,
    'basin': BASIN,
    'admin_level': ADMIN_LEVEL,
    'simplified': USE_SIMPLIFIED,
}

# Output directory structure
# Use OUTPUT_DIR from args, organized by model/variant/scenario/time_period/basin
# Default: /mnt/team/rapidresponse/pub/tropical-storms/climada_admin_summaries/
output_base = (OUTPUT_DIR / MODEL / VARIANT / SCENARIO / TIME_PERIOD / BASIN)
output_base.mkdir(parents=True, exist_ok=True)

for i, draw in enumerate(DRAWS, 1):
    print(f"\n[{i}/{len(DRAWS)}] Processing draw {draw}...")
    
    # Load basin data
    basin_data = ClimadaBasinData(
        model=MODEL,
        variant=VARIANT,
        scenario=SCENARIO,
        time_period=TIME_PERIOD,
        basin=BASIN,
        draw=draw,
    )
    
    # Validate paths
    valid, msg = basin_data.validate_paths()
    if not valid:
        print(f"  ⚠️ {msg}")
        continue
    
    storms = basin_data.list_storms()
    print(f"  Found {len(storms)} storms")
    
    # Process all storms
    all_impacts = []
    storms_with_impacts = 0
    
    for storm_name in storms:
        storm = basin_data.get_storm(storm_name)
        impacts = process_storm(storm, admin_raster, code_map, admin_names)
        
        if impacts:
            all_impacts.extend(impacts)
            storms_with_impacts += 1
        
        storm.close()
    
    print(f"  ✓ {storms_with_impacts}/{len(storms)} storms hit admin regions")
    print(f"  ✓ {len(all_impacts)} total storm-country impacts")
    
    if not all_impacts:
        print(f"  ⚠️ No impacts found, skipping output")
        continue
    
    # Create Level 3 summary dataset
    summary_ds = create_summary_dataset(
        all_impacts, 
        all_countries_in_basin, 
        TIME_PERIOD, 
        draw, 
        metadata
    )
    
    # Save as zarr
    output_path = output_base / f"draw_{draw:04d}_admin{ADMIN_LEVEL}_summary.zarr"
    summary_ds.to_zarr(output_path, mode='w')
    print(f"  ✓ Saved: {output_path}")
    
    # Optionally save storm-level data
    if SAVE_STORM_LEVEL:
        storm_df = pd.DataFrame(all_impacts)
        storm_df['draw'] = draw
        storm_df['model'] = MODEL
        storm_df['variant'] = VARIANT
        storm_df['scenario'] = SCENARIO
        storm_df['time_period'] = TIME_PERIOD
        storm_df['basin'] = BASIN
        
        storm_csv_path = output_base / f"draw_{draw:04d}_storm_impacts.csv"
        storm_df.to_csv(storm_csv_path, index=False)
        print(f"  ✓ Saved storm-level: {storm_csv_path}")
    
    # Clean up
    basin_data.close_all()

print(f"\n{'='*80}")
print(f"✅ Task {TASK_ID} complete")
print(f"   Output directory: {output_base}")
print(f"={'='*80}")
