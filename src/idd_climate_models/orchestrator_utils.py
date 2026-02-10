"""
Utility functions for TC-risk pipeline orchestration.

This module provides helper functions for:
- Checking output completeness at each level
- Managing dynamic draw counting for Level 4
- Loading and filtering time bins
- Cleaning/deleting output folders
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import subprocess


# ============================================================================
# OUTPUT CHECKING FUNCTIONS
# ============================================================================

def check_level3_output_finished(
    model: str,
    variant: str,
    scenario: str,
    time_period: str,
    tc_risk_output_path: Path
) -> bool:
    """
    Check if Level 3 (global TC-risk) has finished successfully.
    
    Args:
        model: Model name (e.g., 'MPI-ESM1-2-HR')
        variant: Variant name (e.g., 'r1i1p1f1')
        scenario: Scenario name (e.g., 'ssp126')
        time_period: Time period string (e.g., '2015-2040')
        tc_risk_output_path: Base path to TC-risk outputs
    
    Returns:
        True if both env_wnd and thermo files exist, False otherwise
    """
    output_path = tc_risk_output_path / model / variant / scenario / time_period
    if not output_path.exists():
        return False
    
    # Check for required Level 3 output files
    env_wnd_files = list(output_path.glob("env_wnd_*.nc"))
    thermo_files = list(output_path.glob("thermo_*.nc"))
    
    return len(env_wnd_files) > 0 and len(thermo_files) > 0


def get_level4_basin_file_count(
    model: str,
    variant: str,
    scenario: str,
    time_period: str,
    basin: str,
    tc_risk_output_path: Path
) -> int:
    """
    Get the number of .nc files in a Level 4 basin output folder.
    
    Args:
        model: Model name
        variant: Variant name
        scenario: Scenario name
        time_period: Time period string
        basin: Basin code (e.g., 'NA', 'EP')
        tc_risk_output_path: Base path to TC-risk outputs
    
    Returns:
        Count of .nc files, or 0 if folder doesn't exist
    """
    output_path = tc_risk_output_path / model / variant / scenario / time_period / basin
    if not output_path.exists():
        return 0
    
    files = list(output_path.glob("*.nc"))
    return len(files)


def check_level4_output_complete(
    model: str,
    variant: str,
    scenario: str,
    time_period: str,
    basin: str,
    tc_risk_output_path: Path,
    num_draws: int
) -> bool:
    """
    Check if Level 4 (basin TC-risk) output is complete.
    
    Args:
        model: Model name
        variant: Variant name
        scenario: Scenario name
        time_period: Time period string
        basin: Basin code (e.g., 'NA', 'EP')
        tc_risk_output_path: Base path to TC-risk outputs
        num_draws: Expected number of draws
    
    Returns:
        True if has num_draws + 1 files (draws + ensemble mean), False otherwise
    """
    file_count = get_level4_basin_file_count(
        model, variant, scenario, time_period, basin, tc_risk_output_path
    )
    return file_count >= num_draws + 1


def get_existing_draw_numbers(
    model: str,
    variant: str,
    scenario: str,
    time_period: str,
    basin: str,
    tc_risk_output_path: Path
) -> List[int]:
    """
    Get list of existing draw numbers for a basin.
    
    Parses filenames like 'tracks_draw_042.nc' to extract draw numbers.
    
    Args:
        model: Model name
        variant: Variant name
        scenario: Scenario name
        time_period: Time period string
        basin: Basin code
        tc_risk_output_path: Base path to TC-risk outputs
    
    Returns:
        Sorted list of draw numbers that already exist, empty list if folder doesn't exist
    """
    output_path = tc_risk_output_path / model / variant / scenario / time_period / basin
    if not output_path.exists():
        return []
    
    draw_numbers = []
    for file in output_path.glob("tracks_draw_*.nc"):
        try:
            # Extract number from filename like 'tracks_draw_042.nc'
            draw_num = int(file.stem.split('_')[-1])
            draw_numbers.append(draw_num)
        except (ValueError, IndexError):
            continue
    
    return sorted(draw_numbers)


def determine_missing_draw_batches(
    existing_draws: List[int],
    total_draws: int,
    draws_per_batch: int
) -> List[Tuple[int, int]]:
    """
    Determine which draw batches need to be run based on existing draws.
    
    Args:
        existing_draws: List of draw numbers that already exist
        total_draws: Total number of draws needed (e.g., 250)
        draws_per_batch: Number of draws per batch (e.g., 10)
    
    Returns:
        List of (batch_start, batch_end) tuples for missing draws.
        Returns empty list if all draws exist.
    
    Example:
        If total_draws=250, draws_per_batch=10, and existing_draws=[0,1,2,10,11,12]:
        Returns [(3, 9), (13, 249)] - two batches covering missing draws
    """
    existing_set = set(existing_draws)
    needed_draws = [d for d in range(total_draws) if d not in existing_set]
    
    if not needed_draws:
        return []
    
    # Group consecutive draws into batches
    batches = []
    for i in range(0, len(needed_draws), draws_per_batch):
        batch_draws = needed_draws[i:i + draws_per_batch]
        batches.append((batch_draws[0], batch_draws[-1]))
    
    return batches


# ============================================================================
# TIME BINS LOADING
# ============================================================================

def load_time_bins(
    time_bins_path: Path,
    method: str = 'BayesPoisson'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load and process time bins from CSV.
    
    Args:
        time_bins_path: Path to time bins CSV file
        method: Changepoint detection method to filter by (default: 'BayesPoisson')
    
    Returns:
        Tuple of:
        - time_bins_df: Full dataframe with time_period column added
        - time_bins_lookup: Dict mapping (model, variant, scenario) -> list of time_period dicts
    
    Example lookup structure:
        {
            ('MPI-ESM1-2-HR', 'r1i1p1f1', 'ssp126'): [
                {'time_period': '2015-2040', 'start_year': 2015, 'end_year': 2040},
                {'time_period': '2041-2070', 'start_year': 2041, 'end_year': 2070},
            ]
        }
    """
    df = pd.read_csv(time_bins_path)
    
    # Filter to specified method
    df = df[df['method'] == method].copy()
    
    # Create time_period string from start/end years
    df['time_period'] = df['start_year'].astype(str) + '-' + df['end_year'].astype(str)
    
    # Create lookup dictionary
    lookup = {}
    for _, row in df.iterrows():
        key = (row['model'], row['variant'], row['scenario'])
        if key not in lookup:
            lookup[key] = []
        lookup[key].append({
            'time_period': row['time_period'],
            'start_year': int(row['start_year']),
            'end_year': int(row['end_year'])
        })
    
    return df, lookup


# ============================================================================
# CLEANING FUNCTIONS
# ============================================================================

def rename_path_for_deletion(path: Path, suffix: str = '_DELETE') -> Optional[Path]:
    """
    Rename a path by appending suffix to prepare for async deletion.
    
    This allows the workflow to continue while deletion happens in background.
    
    Args:
        path: Path to rename
        suffix: Suffix to append (default: '_DELETE')
    
    Returns:
        New path if successful, None if path doesn't exist
    """
    if not path.exists():
        return None
    
    new_path = path.parent / (path.name + suffix)
    
    # If renamed path already exists, remove it first
    if new_path.exists():
        import shutil
        shutil.rmtree(new_path)
    
    path.rename(new_path)
    return new_path


def delete_path_recursive(
    path: Path,
    timeout_seconds: int = 300,
    nice_priority: bool = True
) -> bool:
    """
    Delete a path recursively using system rm with timeout and optional nice priority.
    
    Uses ionice to be gentle on filesystem if nice_priority=True.
    
    Args:
        path: Path to delete
        timeout_seconds: Maximum time to allow for deletion (default: 300s = 5min)
        nice_priority: Use ionice for low I/O priority (default: True)
    
    Returns:
        True if successful or path doesn't exist, False if timeout or error
    """
    if not path.exists():
        return True
    
    try:
        cmd = ['timeout', str(timeout_seconds)]
        
        if nice_priority:
            cmd.extend(['ionice', '-c', '3'])
        
        cmd.extend(['rm', '-rf', str(path)])
        
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True
        )
        
        # Success if path is gone (even if command timed out)
        return not path.exists()
        
    except Exception:
        return False


def clean_basin_outputs(
    model: str,
    variant: str,
    scenario: str,
    time_period: str,
    basin: str,
    tc_risk_output_path: Path,
    climada_input_path: Path,
    timeout_per_path: int = 60
) -> Tuple[bool, bool]:
    """
    Clean both TC-Risk outputs and CLIMADA inputs for a specific basin.
    
    Args:
        model: Model name
        variant: Variant name
        scenario: Scenario name
        time_period: Time period string
        basin: Basin code
        tc_risk_output_path: Base path to TC-risk outputs
        climada_input_path: Base path to CLIMADA inputs
        timeout_per_path: Timeout in seconds for each deletion (default: 60)
    
    Returns:
        Tuple of (tc_risk_cleaned, climada_cleaned) booleans
    """
    # Clean TC-Risk output
    tc_path = tc_risk_output_path / model / variant / scenario / time_period / basin
    tc_cleaned = delete_path_recursive(tc_path, timeout_seconds=timeout_per_path)
    
    # Clean CLIMADA input
    climada_path = climada_input_path / model / variant / scenario / time_period / basin
    climada_cleaned = delete_path_recursive(climada_path, timeout_seconds=timeout_per_path)
    
    # Recreate empty CLIMADA directory
    if climada_cleaned and not climada_path.exists():
        climada_path.mkdir(parents=True, exist_ok=True)
    
    return tc_cleaned, climada_cleaned


def clean_all_basin_outputs_parallel(
    tasks: List[Dict],
    tc_risk_output_path: Path,
    climada_input_path: Path,
    max_workers: int = 10,
    verbose: bool = False
) -> int:
    """
    Clean basin outputs for multiple tasks in parallel.
    
    Args:
        tasks: List of task dicts with keys: model, variant, scenario, time_period, basin
        tc_risk_output_path: Base path to TC-risk outputs
        climada_input_path: Base path to CLIMADA inputs
        max_workers: Maximum number of parallel workers (default: 10)
        verbose: Print progress (default: False)
    
    Returns:
        Number of basins successfully cleaned
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if verbose:
        from tqdm import tqdm
        
    cleaned_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for task in tasks:
            future = executor.submit(
                clean_basin_outputs,
                task['model'], task['variant'], task['scenario'],
                task['time_period'], task['basin'],
                tc_risk_output_path, climada_input_path
            )
            futures.append(future)
        
        # Wait for all to complete
        iterator = as_completed(futures)
        if verbose:
            iterator = tqdm(iterator, total=len(futures), desc="Cleaning basin folders")
        
        for future in iterator:
            tc_cleaned, climada_cleaned = future.result()
            if tc_cleaned and climada_cleaned:
                cleaned_count += 1
    
    return cleaned_count


# ============================================================================
# FILTERING FUNCTIONS
# ============================================================================

def filter_time_bins_by_targets(
    time_bins_df: pd.DataFrame,
    targets: List[Dict],
    verbose: bool = False
) -> pd.DataFrame:
    """
    Filter time bins dataframe to only include specified targets.
    
    Args:
        time_bins_df: Full time bins dataframe
        targets: List of dicts with optional keys: model, variant, scenario, time_period, basin
                 Empty/None values are treated as wildcards
        verbose: Print filtering progress
    
    Returns:
        Filtered dataframe
    
    Example targets:
        [
            {'model': 'MPI-ESM1-2-HR', 'variant': 'r1i1p1f1', 'scenario': 'ssp126', 'time_period': '2015-2040'},
            {'model': 'MRI-ESM2-0', 'variant': 'r1i1p1f1', 'scenario': None, 'time_period': None},  # All scenarios/periods
        ]
    """
    if not targets:
        return time_bins_df
    
    # Build filter mask
    mask = pd.Series([False] * len(time_bins_df), index=time_bins_df.index)
    
    for target in targets:
        target_mask = pd.Series([True] * len(time_bins_df), index=time_bins_df.index)
        
        # Apply filters for non-None values
        if target.get('model'):
            target_mask &= (time_bins_df['model'] == target['model'])
        if target.get('variant'):
            target_mask &= (time_bins_df['variant'] == target['variant'])
        if target.get('scenario'):
            target_mask &= (time_bins_df['scenario'] == target['scenario'])
        if target.get('time_period'):
            target_mask &= (time_bins_df['time_period'] == target['time_period'])
        
        mask |= target_mask
    
    filtered_df = time_bins_df[mask].copy()
    
    if verbose:
        print(f"Filtered from {len(time_bins_df)} to {len(filtered_df)} time bin combinations")
    
    return filtered_df