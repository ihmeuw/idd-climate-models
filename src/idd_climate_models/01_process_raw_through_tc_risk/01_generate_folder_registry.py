"""
One-off script to generate folder_paths_registry.csv from existing time bins.
This will create a CSV with all model/variant/scenario/time_period combinations
and their corresponding input/output paths.
"""

import pandas as pd
from pathlib import Path
import idd_climate_models.constants as rfc

def main():
    # Load time bins
    time_bins_df = pd.read_csv(rfc.TIME_BINS_DF_PATH)
    
    # Filter to BayesPoisson method only
    time_bins_df = time_bins_df[time_bins_df['method'] == 'BayesPoisson']
    
    # Create time_period column
    time_bins_df['time_period'] = time_bins_df['start_year'].astype(str) + '-' + time_bins_df['end_year'].astype(str)
    
    # Get unique combinations
    unique_combos = time_bins_df[['model', 'variant', 'scenario', 'time_period']].drop_duplicates()
    
    # Build paths dataframe
    data_source = "cmip6"
    rows = []
    
    for _, row in unique_combos.iterrows():
        model = row['model']
        variant = row['variant']
        scenario = row['scenario']
        time_period = row['time_period']
        
        input_path = rfc.TC_RISK_INPUT_PATH / data_source / model / variant / scenario / time_period
        output_path = rfc.TC_RISK_OUTPUT_PATH / data_source / model / variant / scenario / time_period
        
        rows.append({
            'model': model,
            'variant': variant,
            'scenario': scenario,
            'time_period': time_period,
            'input_path': str(input_path),
            'output_path': str(output_path),
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