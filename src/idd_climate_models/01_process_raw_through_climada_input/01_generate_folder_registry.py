"""
One-off script to generate folder_paths_registry.csv using the SAME logic as the orchestrator.
Uses chunked time bins (MAX_PERIOD_DURATION=5 years) to match what folders are created.
"""

import pandas as pd
from pathlib import Path
import idd_climate_models.constants as rfc
from idd_climate_models.time_period_functions import get_time_bins_path

def main():
    # Use same MAX_PERIOD_DURATION as orchestrator
    MAX_PERIOD_DURATION = 5  # Must match 00_orchestrator.py
    data_source = "cmip6"
    
    print("=" * 80)
    print(f"Generating folder registry using chunked time bins")
    print(f"  MAX_PERIOD_DURATION: {MAX_PERIOD_DURATION} years")
    print(f"  Data source: {data_source}")
    print("=" * 80)
    
    # Get time bins file (same logic as orchestrator)
    time_bins_path = get_time_bins_path(MAX_PERIOD_DURATION)
    print(f"\nLoading time bins from: {time_bins_path}")
    time_bins_df = pd.read_csv(time_bins_path)
    
    # Filter to BayesPoisson method only (already filtered if using chunked file)
    if MAX_PERIOD_DURATION is None:
        time_bins_df = time_bins_df[time_bins_df['method'] == 'BayesPoisson']
    
    # Create time_period column
    time_bins_df['time_period'] = time_bins_df['start_year'].astype(str) + '-' + time_bins_df['end_year'].astype(str)
    
    # Get unique combinations
    unique_combos = time_bins_df[['model', 'variant', 'scenario', 'time_period']].drop_duplicates()
    
    print(f"Found {len(unique_combos)} unique model/variant/scenario/time_period combinations")
    
    # Build paths dataframe
    rows = []
    
    for _, row in unique_combos.iterrows():
        model = row['model']
        variant = row['variant']
        scenario = row['scenario']
        time_period = row['time_period']
        
        tc_risk_input_path = rfc.TC_RISK_INPUT_PATH / data_source / model / variant / scenario / time_period
        tc_risk_output_path = rfc.TC_RISK_OUTPUT_PATH / data_source / model / variant / scenario / time_period
        climada_input_path = rfc.CLIMADA_INPUT_PATH / data_source / model / variant / scenario / time_period
        
        rows.append({
            'model': model,
            'variant': variant,
            'scenario': scenario,
            'time_period': time_period,
            'tc_risk_input_path': str(tc_risk_input_path),
            'tc_risk_output_path': str(tc_risk_output_path),
            'climada_input_path': str(climada_input_path),
        })
    
    registry_df = pd.DataFrame(rows)
    
    # Save to CSV
    log_file = rfc.TC_RISK_INPUT_PATH / data_source / "folder_paths_registry.csv"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    registry_df.to_csv(log_file, index=False)
    
    print(f"Generated folder_paths_registry.csv with {len(registry_df)} rows")
    print(f"Saved to: {log_file}")
    print(f"\nFirst few rows:")
    print(registry_df.head())

if __name__ == "__main__":
    main()