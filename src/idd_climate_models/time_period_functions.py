"""
Functions for managing and chunking time periods.

This module handles:
- Splitting long time periods into smaller chunks
- Loading and caching chunked time bins files
- Time period utilities
"""

import math
import pandas as pd
from pathlib import Path
from idd_climate_models import constants as rfc


def split_period(start_year, end_year, max_width):
    """
    Split a time period into bins of at most max_width years,
    with bin sizes as equal as possible.
    
    Args:
        start_year: Starting year of period
        end_year: Ending year of period
        max_width: Maximum bin width in years
        
    Returns:
        List of (start_year, end_year) tuples
    
    Example:
        >>> split_period(2015, 2042, 5)
        [(2015, 2019), (2020, 2024), (2025, 2029), (2030, 2034), (2035, 2039), (2040, 2042)]
    """
    total_years = end_year - start_year + 1
    
    # Minimum bins needed
    n_bins = math.ceil(total_years / max_width)
    
    # Base size and remainder
    base_size = total_years // n_bins
    remainder = total_years % n_bins
    
    # Create bins: first 'remainder' bins get (base_size + 1), rest get base_size
    bins = []
    current_year = start_year
    
    for i in range(n_bins):
        bin_size = base_size + (1 if i < remainder else 0)
        bin_end = current_year + bin_size - 1
        bins.append((current_year, bin_end))
        current_year = bin_end + 1
    
    return bins


def get_time_bins_path(max_duration=None):
    """
    Get the path to the appropriate time bins CSV file.
    
    If max_duration is None, returns the base file path.
    If max_duration is specified, returns the chunked file path and creates it if needed.
    
    Args:
        max_duration: Maximum period duration in years (None = no chunking)
        
    Returns:
        Path to the time bins CSV file
    """
    if max_duration is None:
        return rfc.TIME_BINS_DF_PATH
    
    # Path for chunked file
    base_dir = rfc.TIME_BINS_DF_PATH.parent
    chunked_path = base_dir / f'bayespoisson_time_bins_max_bin_{max_duration}.csv'
    
    # If chunked file exists, use it
    if chunked_path.exists():
        print(f"Using existing chunked time bins: {chunked_path.name}")
        return chunked_path
    
    # Otherwise, create it
    print(f"Creating chunked time bins file with max duration {max_duration} years...")
    
    # Read base file
    df = pd.read_csv(rfc.TIME_BINS_DF_PATH)
    
    # Filter to BayesPoisson method
    df = df[df['method'] == 'BayesPoisson']
    
    # Process each row
    new_rows = []
    rows_split = 0
    rows_kept = 0
    
    for _, row in df.iterrows():
        start_year = int(row['start_year'])
        end_year = int(row['end_year'])
        period_length = end_year - start_year + 1
        
        if period_length <= max_duration:
            # Keep as-is
            new_rows.append(row.to_dict())
            rows_kept += 1
        else:
            # Split into smaller bins
            bins = split_period(start_year, end_year, max_duration)
            rows_split += 1
            for bin_start, bin_end in bins:
                new_row = row.to_dict()
                new_row['start_year'] = bin_start
                new_row['end_year'] = bin_end
                new_rows.append(new_row)
    
    # Create new dataframe and save LONG format
    chunked_df = pd.DataFrame(new_rows)
    chunked_df.to_csv(chunked_path, index=False)
    
    print(f"✅ Created long format: {chunked_path}")
    print(f"   Original rows: {len(df)}")
    print(f"   Rows kept as-is: {rows_kept}")
    print(f"   Rows split: {rows_split}")
    print(f"   New total rows: {len(chunked_df)}")
    
    # Also create WIDE format version (same structure as bayespoisson_time_bins_wide.csv)
    wide_path = base_dir / f'bayespoisson_time_bins_wide_max_bin_{max_duration}.csv'
    
    # Load the original wide format file to get basin storm counts
    original_wide_path = base_dir / 'bayespoisson_time_bins_wide.csv'
    if original_wide_path.exists():
        print(f"\n   Creating wide format version...")
        wide_df_original = pd.read_csv(original_wide_path)
        
        # For each row in chunked_df, find the matching original period and copy basin storm counts
        wide_rows = []
        for _, chunk_row in chunked_df.iterrows():
            chunk_start = int(chunk_row['start_year'])
            chunk_end = int(chunk_row['end_year'])
            model = chunk_row['model']
            variant = chunk_row['variant']
            scenario = chunk_row['scenario']
            
            # Find the original period that contains this chunk
            # The chunk must be fully contained within the original period
            matching_original = wide_df_original[
                (wide_df_original['model'] == model) &
                (wide_df_original['variant'] == variant) &
                (wide_df_original['scenario'] == scenario) &
                (wide_df_original['start_year'] <= chunk_start) &
                (wide_df_original['end_year'] >= chunk_end)
            ]
            
            if len(matching_original) > 0:
                # Copy the first matching original row
                wide_row = matching_original.iloc[0].to_dict()
                # Update only the years to match the chunk (keep all basin storm counts)
                wide_row['start_year'] = chunk_start
                wide_row['end_year'] = chunk_end
                wide_rows.append(wide_row)
            else:
                # No matching original period found - this shouldn't happen in normal operation
                print(f"   ⚠️  Warning: No original period found for chunk {model}/{variant}/{scenario}/{chunk_start}-{chunk_end}")
        
        if wide_rows:
            wide_df = pd.DataFrame(wide_rows)
            wide_df.to_csv(wide_path, index=False)
            print(f"✅ Created wide format: {wide_path}")
            print(f"   Rows in wide format: {len(wide_df)}")
        else:
            print(f"   ⚠️  Warning: Could not create wide format (no matching rows)")
    else:
        print(f"   ⚠️  Warning: Original wide format file not found at {original_wide_path}")
        print(f"                Skipping wide format creation")
    
    return chunked_path
