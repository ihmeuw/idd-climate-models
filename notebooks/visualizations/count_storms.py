"""
Count storms across all draws by model/variant/scenario/basin/time_period.
"""

from pathlib import Path
import pandas as pd

# Data directory
DATA_DIR = Path("/ihme/homes/bcreiner/repos/idd-climate-models/outputs/storm_admin_summaries")

# Load all storm_level_data files
print("Loading storm-level data files...")
storm_files = list(DATA_DIR.glob("storm_level_data_*.csv"))
print(f"Found {len(storm_files)} files")

dfs = []
for file in storm_files:
    parts = file.stem.replace("storm_level_data_", "").split("_")
    if len(parts) >= 3:
        scenario = parts[-1]
        variant = parts[-2]
        model = "_".join(parts[:-2])
        
        df = pd.read_csv(file, keep_default_na=False, na_values=[''])
        
        # Add model/variant/scenario if not already present
        if 'model' not in df.columns:
            df['model'] = model
        if 'variant' not in df.columns:
            df['variant'] = variant
        if 'scenario' not in df.columns:
            df['scenario'] = scenario
            
        dfs.append(df)

df_all_storms = pd.concat(dfs, ignore_index=True)
print(f"\nLoaded {len(df_all_storms):,} total storm records (landfall storms only)")

# Check columns
print(f"\nColumns: {list(df_all_storms.columns)}")

# ============================================================================
# Count storms by different groupings
# ============================================================================

print("\n" + "="*80)
print("STORM COUNTS BY DIFFERENT GROUPINGS")
print("="*80)

# 1. Total storms per draw (across everything)
storms_per_draw = df_all_storms.groupby('draw').size().reset_index(name='storm_count')
print(f"\n1. Storms per draw (across all models/variants/scenarios/basins/time_periods):")
print(f"   Min: {storms_per_draw['storm_count'].min()}")
print(f"   Max: {storms_per_draw['storm_count'].max()}")
print(f"   Mean: {storms_per_draw['storm_count'].mean():.1f}")
print(f"   Total unique draws: {len(storms_per_draw)}")

# 2. Storms by model/variant/scenario/basin/time_period/draw
storm_counts_detailed = df_all_storms.groupby(
    ['model', 'variant', 'scenario', 'basin', 'time_period', 'draw']
).size().reset_index(name='storm_count')

print(f"\n2. Storms by model/variant/scenario/basin/time_period/draw:")
print(f"   Total combinations: {len(storm_counts_detailed):,}")
print(f"   Storm count range: {storm_counts_detailed['storm_count'].min()} - {storm_counts_detailed['storm_count'].max()}")
print(f"\n   Sample:")
print(storm_counts_detailed.head(10).to_string(index=False))

# 3. Average storms per draw by scenario/basin/time_period (across models/variants)
storm_counts_summary = storm_counts_detailed.groupby(
    ['scenario', 'basin', 'time_period']
)['storm_count'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()

print(f"\n3. Average storms per draw by scenario/basin/time_period (across all model/variants):")
print(storm_counts_summary.to_string(index=False))

# 4. Total storms by scenario (summed across all draws/basins/time_periods)
by_scenario = df_all_storms.groupby('scenario').size().reset_index(name='total_storm_records')
print(f"\n4. Total storm records by scenario:")
print(by_scenario.to_string(index=False))

# 5. Total storms by basin (summed across all draws/scenarios/time_periods)
by_basin = df_all_storms.groupby('basin').size().reset_index(name='total_storm_records')
print(f"\n5. Total storm records by basin:")
print(by_basin.to_string(index=False))

# 6. Total storms by model/variant
by_model_variant = df_all_storms.groupby(['model', 'variant']).size().reset_index(name='total_storm_records')
print(f"\n6. Total storm records by model/variant:")
print(by_model_variant.to_string(index=False))

# ============================================================================
# Save detailed counts
# ============================================================================

output_file = DATA_DIR / "storm_counts_detailed.csv"
storm_counts_detailed.to_csv(output_file, index=False)
print(f"\n✓ Saved detailed counts to: {output_file}")

summary_file = DATA_DIR / "storm_counts_summary.csv"
storm_counts_summary.to_csv(summary_file, index=False)
print(f"✓ Saved summary counts to: {summary_file}")

print("\n" + "="*80)
print("IMPORTANT NOTE:")
print("These counts are for storms that hit at least one country (landfall only).")
print("Basin stats may include additional non-landfall storms from TC risk modeling.")
print("="*80)
