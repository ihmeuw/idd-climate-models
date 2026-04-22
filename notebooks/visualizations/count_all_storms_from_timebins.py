"""
Count ALL storms from the pre-computed TempestExtremes time bins file.
This is MUCH faster than opening individual NetCDF files!
"""

from pathlib import Path
import sys
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
import idd_climate_models.constants as rfc

print("="*80)
print("COUNTING ALL STORMS FROM TEMPESTEXTREMES TIME BINS")
print("="*80)

# Load the time bins file
print(f"\nLoading: {rfc.TIME_BINS_WIDE_DF_PATH}")
df = pd.read_csv(rfc.TIME_BINS_WIDE_DF_PATH)
print(f"Loaded {len(df)} time bins")

# The _int columns have the actual storm counts per basin per time bin
basin_cols = ['AU_int', 'EP_int', 'GL_int', 'NA_int', 'NI_int', 'SI_int', 'SP_int', 'WP_int']
basins = ['AU', 'EP', 'GL', 'NA', 'NI', 'SI', 'SP', 'WP']

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

# Total storms across everything
total_storms = df[basin_cols].sum().sum()
print(f"\nTotal storms across all basins/models/scenarios/time periods: {total_storms:,}")

# By basin
print("\n1. Total storms by basin:")
basin_totals = []
for basin in basins:
    col = f'{basin}_int'
    total = df[col].sum()
    basin_totals.append({'basin': basin, 'total_storms': total})
basin_df = pd.DataFrame(basin_totals).sort_values('total_storms', ascending=False)
print(basin_df.to_string(index=False))

# By scenario
print("\n2. Total storms by scenario:")
scenario_totals = df.groupby('scenario')[basin_cols].sum().sum(axis=1).reset_index()
scenario_totals.columns = ['scenario', 'total_storms']
print(scenario_totals.to_string(index=False))

# By model/variant
print("\n3. Total storms by model/variant:")
mv_totals = df.groupby(['model', 'variant'])[basin_cols].sum().sum(axis=1).reset_index()
mv_totals.columns = ['model', 'variant', 'total_storms']
print(mv_totals.to_string(index=False))

# By model/variant/scenario
print("\n4. Total storms by model/variant/scenario:")
mvs_totals = df.groupby(['model', 'variant', 'scenario'])[basin_cols].sum().sum(axis=1).reset_index()
mvs_totals.columns = ['model', 'variant', 'scenario', 'total_storms']
print(mvs_totals.to_string(index=False))

# Detailed counts by basin for each combination
print("\n5. Storms by model/variant/scenario/basin:")
detailed_data = []
for _, row in df.iterrows():
    for basin in basins:
        col = f'{basin}_int'
        detailed_data.append({
            'model': row['model'],
            'variant': row['variant'],
            'scenario': row['scenario'],
            'bin_idx': row['bin_idx'],
            'start_year': row['start_year'],
            'end_year': row['end_year'],
            'bin_size': row['bin_size'],
            'basin': basin,
            'n_storms': row[col]
        })

detailed_df = pd.DataFrame(detailed_data)

# Summary by scenario/basin
print("\nAverage storms per time bin by scenario/basin:")
summary = detailed_df.groupby(['scenario', 'basin'])['n_storms'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
print(summary.to_string(index=False))

# Save outputs
output_dir = Path("/ihme/homes/bcreiner/repos/idd-climate-models/outputs/storm_admin_summaries")
output_dir.mkdir(parents=True, exist_ok=True)

detailed_file = output_dir / "all_storms_timebins_detailed.csv"
detailed_df.to_csv(detailed_file, index=False)
print(f"\n✓ Saved detailed counts to: {detailed_file}")

summary_file = output_dir / "all_storms_timebins_summary.csv"
summary.to_csv(summary_file, index=False)
print(f"✓ Saved summary to: {summary_file}")

print("\n" + "="*80)
print("NOTE: These are ALL storms from TempestExtremes (landfall + non-landfall)")
print("These counts are per TIME BIN, not per draw.")
print("Each time bin has a Bayesian Poisson estimate of storm counts.")
print("="*80)
