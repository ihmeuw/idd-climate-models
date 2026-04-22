"""
Count ALL storms (landfall + non-landfall) from TC risk model output files.
"""

from pathlib import Path
import sys
import xarray as xr
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
import idd_climate_models.constants as rfc

# TC risk output path
TC_RISK_OUTPUT = Path(rfc.TC_RISK_OUTPUT_PATH) / "cmip6"

print("="*80)
print("COUNTING ALL STORMS FROM TC RISK MODEL OUTPUT")
print("="*80)

# Load task assignments to know all combinations
print("\nLoading task assignments...")
task_file = Path(rfc.CLIMADA_INPUT_PATH) / 'cmip6' / 'level_4_task_assignments.csv'
df_tasks = pd.read_csv(task_file, keep_default_na=False)
print(f"Found {len(df_tasks)} task assignments")

# Get unique combinations
unique_combos = df_tasks[['model', 'variant', 'scenario', 'time_period', 'basin', 'draw']].drop_duplicates()
print(f"Unique model/variant/scenario/time_period/basin/draw combinations: {len(unique_combos)}")

# Build file paths directly
print("\nConstructing file paths from task assignments...")
track_files = []
for _, row in unique_combos.iterrows():
    # Path pattern: TC_RISK_OUTPUT/{model}/{variant}/{scenario}/{time_period}/{basin}/tracks_{basin}_{model}_{scenario}_{variant}_{dates}_e{draw}.nc
    # Note: dates need to be converted from time_period format (e.g., 1965-1969 -> 196501_196912)
    time_period = row['time_period']
    start_year, end_year = time_period.split('-')
    dates = f"{start_year}01_{end_year}12"
    
    file_path = TC_RISK_OUTPUT / row['model'] / row['variant'] / row['scenario'] / time_period / row['basin'] / \
                f"tracks_{row['basin']}_{row['model']}_{row['scenario']}_{row['variant']}_{dates}_e{row['draw']}.nc"
    
    if file_path.exists():
        track_files.append(file_path)

print(f"Found {len(track_files)} existing track files (out of {len(unique_combos)} expected)")

if len(track_files) == 0:
    print("\nNo track files found!")
    sys.exit(1)

# Parse file paths to organize storms
print("\nCounting storms in each file...")
storm_counts = []
file_count = 0

for i, file_path in enumerate(track_files, 1):
    try:
        # Extract info from the row we already have
        # Find matching row in unique_combos based on file path
        filename = file_path.stem
        parts = filename.split("_")
        
        basin = parts[1]
        model = parts[2]
        scenario = parts[3]
        variant = parts[4]
        draw = int(parts[-1].replace('e', ''))
        
        # Time period from parent directory
        time_period = file_path.parent.parent.name
        
        # Open dataset and count storms
        with xr.open_dataset(file_path) as ds:
            n_storms = ds.dims.get('n_trk', 0)
        
        storm_counts.append({
            'model': model,
            'variant': variant,
            'scenario': scenario,
            'basin': basin,
            'time_period': time_period,
            'draw': draw,
            'n_storms': n_storms,
            'file': str(file_path.relative_to(TC_RISK_OUTPUT))
        })
        
        file_count += 1
        if file_count % 1000 == 0:
            print(f"  Processed {file_count}/{len(track_files)} files...")
            
    except Exception as e:
        print(f"  ⚠️  Error processing {file_path.name}: {e}")
        continue

print(f"\nSuccessfully processed {file_count} files")

# Convert to DataFrame
import pandas as pd
df = pd.DataFrame(storm_counts)

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nTotal files processed: {len(df)}")
print(f"Total storms across all files: {df['n_storms'].sum():,}")

# Aggregate by different groupings
print("\n1. Total storms by model/variant/scenario:")
by_combo = df.groupby(['model', 'variant', 'scenario'])['n_storms'].sum().reset_index()
print(by_combo.to_string(index=False))

print("\n2. Average storms per draw by scenario/basin/time_period:")
by_group = df.groupby(['scenario', 'basin', 'time_period'])['n_storms'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
print(by_group.head(20).to_string(index=False))

print("\n3. Total storms by basin:")
by_basin = df.groupby('basin')['n_storms'].sum().reset_index()
by_basin = by_basin.sort_values('n_storms', ascending=False)
print(by_basin.to_string(index=False))

print("\n4. Total storms by scenario:")
by_scenario = df.groupby('scenario')['n_storms'].sum().reset_index()
print(by_scenario.to_string(index=False))

print("\n5. Average storms per draw (across all dimensions):")
storms_per_draw = df.groupby('draw')['n_storms'].sum()
print(f"   Min: {storms_per_draw.min():,}")
print(f"   Max: {storms_per_draw.max():,}")
print(f"   Mean: {storms_per_draw.mean():,.1f}")
print(f"   Total draws: {len(storms_per_draw)}")

# Save detailed results
output_file = Path("/ihme/homes/bcreiner/repos/idd-climate-models/outputs/storm_admin_summaries/all_storms_detailed.csv")
df.to_csv(output_file, index=False)
print(f"\n✓ Saved detailed counts to: {output_file}")

summary_file = Path("/ihme/homes/bcreiner/repos/idd-climate-models/outputs/storm_admin_summaries/all_storms_summary.csv")
by_group.to_csv(summary_file, index=False)
print(f"✓ Saved summary to: {summary_file}")

print("\n" + "="*80)
print("NOTE: These counts include ALL storms (landfall + non-landfall)")
print("="*80)
