import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import os
import re
from typing import List, Tuple, Dict, Any, Union # Added for better type hinting

# Import your constants module
import idd_climate_models.constants as rfc

# ============================================================================
# 1. CONSTANTS AND UTILITY FUNCTIONS
# ============================================================================

# --- Core Constants from rfc ---
# Define the core constants used in this script
MODELS_TO_RUN = ['ACCESS-CM2', 'EC-Earth3', 'EC-Earth3-Veg','EC-Earth3-Veg-LR', 'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0']
BIN_SIZE_YEARS = 20
DATA_SOURCE = "cmip6"
ssp_scenario_map = rfc.ssp_scenario_map
FUTURE_SCENARIOS = [s for s in ssp_scenario_map if s != 'historical']
NUM_DRAWS = 25
basin_dict = rfc.basin_dict
BASINS = [b for b in basin_dict if basin_dict[b]['most_detailed']]
VARIANT = 'r1i1p1f1' 

# Constants derived from rfc
TC_RISK_OUTPUT_PATH = rfc.TC_RISK_OUTPUT_PATH
THRESHOLD_DICT = rfc.threshold_dict # Renamed to uppercase for standard constant naming
TC_RISK_DATE_RANGES = rfc.build_tc_risk_date_ranges()
TIMESTEPS_PER_DAY = rfc.tc_time_steps_per_day


# Pre-calculate all time bins (This is efficient and kept)
TIME_BINS: Dict[str, List[Tuple[int, int]]] = {
    scenario: rfc.get_time_periods(scenario, BIN_SIZE_YEARS)
    for scenario in ssp_scenario_map
}

def get_output_dir(model: str, scenario: str, time_period: Tuple[int, int]) -> Path:
    """Constructs the parent directory for a time bin."""
    # Simplified Path construction
    time_period_str = f'{time_period[0]}-{time_period[1]}'
    return TC_RISK_OUTPUT_PATH / DATA_SOURCE / model / VARIANT / scenario / time_period_str

def get_track_path(model: str, scenario: str, time_period: Tuple[int, int], basin: str, draw: int) -> Path:
    """Constructs the full file path for a track file."""
    output_dir_parent = get_output_dir(model, scenario, time_period)
    
    # Draws are 1-based, array indices are 0-based (e.g., draw 1 is _e0)
    draw_text = f'_e{draw - 1}' if draw > 0 else ''
    
    # Use f-strings for time strings, ensuring 4-digit years
    time_start_str = f'{time_period[0]:04d}01' # Added :04d for explicit 4-digit formatting
    time_end_str = f'{time_period[1]:04d}12'
    
    # File name construction
    track_file = f'tracks_{basin}_{model}_{scenario}_{VARIANT}_{time_start_str}_{time_end_str}{draw_text}.nc'
    
    # The basin directory is a subdirectory of the time_period directory
    return output_dir_parent / basin / track_file


def process_track_file(file_path: Path, model: str, scenario: str, 
                       basin: str, draw: int, threshold_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Union[str, int, float]]]:
    results = []
    
    try:
        # Use a context manager to ensure the file is closed
        with xr.open_dataset(file_path) as ds:
            
            # Extract and validate required data
            vmax_trks = ds['vmax_trks'].values  # Shape: (n_trk, time)
            tc_years = ds['tc_years'].values    # Shape: (n_trk) 
            
            # 1. Pre-calculate track maximum Vmax
            # np.nanmax handles NaNs correctly
            track_max_vmax = np.nanmax(vmax_trks, axis=1)

            # Loop over all thresholds
            for threshold_key, threshold_data in threshold_dict.items():
                threshold_value = threshold_data['wind_speed']

                # 2. Identify tracks that ever exceed the threshold (boolean array)
                exceeds_threshold = track_max_vmax >= threshold_value
                
                # Identify unique years present in the dataset (exclude NaNs)
                # Use np.unique and mask in one step
                unique_years = np.unique(tc_years[~np.isnan(tc_years)]).astype(int)

                # 3. Calculate count PER YEAR AND Duration (DAYS)
                for year in unique_years:
                    # Mask for tracks that originated in this specific year AND exceed the threshold
                    year_mask = (tc_years == year)
                    
                    # Track indices that qualify for this year/threshold
                    qualifying_track_indices = np.where(year_mask & exceeds_threshold)[0]
                    count = len(qualifying_track_indices)
                    total_duration_days = 0.0

                    if count > 0:

                        vmax_subset = vmax_trks[qualifying_track_indices, :]
                        
                        # Boolean mask: True if Vmax >= threshold
                        # This works on the subset of tracks
                        duration_mask = vmax_subset >= threshold_value
                        
                        # Sum 'True' values (time steps) across the whole subset
                        total_timesteps = np.nansum(duration_mask)
                        
                        # Conversion: total_timesteps / TIMESTEPS_PER_DAY
                        total_duration_days = total_timesteps / TIMESTEPS_PER_DAY
                    
                    # Append the result
                    results.append({
                        'model': model,
                        'scenario': scenario,
                        'year': year,           
                        'basin': basin,
                        'draw': draw,
                        'threshold': threshold_key,
                        'count': count,
                        'days': total_duration_days
                    })
        
    except Exception as e:
        # Better error logging to show the location of the error
        print(f"Error processing file {file_path.name} "
              f"({model}, {scenario}, {basin}, draw {draw}): {e}")
        # Continue silently if an error occurs within the function
        
    return results

def process_model_scenario(model: str, scenario: str, 
                           threshold_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    all_results_list: List[Dict[str, Union[str, int, float]]] = []
    for scenario in ssp_scenario_map:
        print(f"  Scenario: {scenario}")
        scenario_time_periods = TIME_BINS.get(scenario, [])
        for time_period_tuple in scenario_time_periods:
            print(f"    Time bin: {time_period_tuple[0]}-{time_period_tuple[1]}")            
            for basin in BASINS:
                for draw in range(0, NUM_DRAWS):
                    file_path = get_track_path(model, scenario, time_period_tuple, basin, draw)
                    if not file_path.exists():
                        continue # Skip non-existent files
                    # Call the encapsulated processing function
                    results = process_track_file(
                        file_path=file_path, 
                        model=model, 
                        scenario=scenario, 
                        basin=basin, 
                        draw=draw, 
                        threshold_dict=THRESHOLD_DICT
                    )
                    # Extend the master list with results from the file
                    all_results_list.extend(results)
                    
    model_df = pd.DataFrame(all_results_list)

    group_columns = [
        'model', 'scenario', 'year', 'draw', 'threshold'
    ]

    # 2. Group the DataFrame by these columns and sum the numerical metrics
    global_df = model_df.groupby(group_columns, observed=True).agg(
        # Sum the storm count
        count=('count', 'sum'),
        # Sum the storm duration days
        days=('days', 'sum')
    ).reset_index()
    global_df['basin'] = 'GL'

    model_df = pd.concat([model_df, global_df], ignore_index=True)

def calculate_cumulative_convergence(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the full suite of running stability diagnostics across all draws (k=1 to max_draws) 
    for every unique combination of model/scenario/year/basin/threshold.
    
    Includes: Running Mean, STD, CV, Relative Change in Mean, STD, and CV.

    Args:
        model_df: The long DataFrame containing raw storm metrics.

    Returns:
        A long DataFrame containing the running diagnostics.
    """
    
    group_dims: List[str] = ['model', 'scenario', 'year', 'basin', 'threshold']
    metrics: List[str] = ['count', 'days']
    
    df_sorted = model_df.sort_values(by=group_dims + ['draw'])
    stability_results: List[pd.DataFrame] = []
    
    grouped_data = df_sorted.groupby(group_dims)
    
    for group_key, group_df in grouped_data:
        
        for metric in metrics:
            values = group_df[metric].values
            N = len(values)
            
            # --- 1. Calculate Core Running Statistics (Mean, STD) ---
            cumulative_sum = np.cumsum(values)
            running_mean = cumulative_sum / np.arange(1, N + 1)
            running_std = [np.std(values[:i+1], ddof=1) if i > 0 else 0.0 for i in range(N)]
            
            # --- 2. Calculate Running CV (Coefficient of Variation) ---
            running_cv = np.divide(running_std, running_mean, 
                                   out=np.zeros_like(running_mean, dtype=float), 
                                   where=running_mean != 0)
            
            # --- 3. Calculate Relative Change Metrics (The new required logic) ---
            
            relative_change_mean = np.zeros_like(running_mean)
            relative_change_std = np.zeros_like(running_std, dtype=float)
            relative_change_cv = np.zeros_like(running_cv, dtype=float)
            
            for k in range(1, N):
                # Relative Change in Mean
                if np.abs(running_mean[k]) > 1e-9: 
                    relative_change_mean[k] = np.abs((running_mean[k] - running_mean[k - 1]) / running_mean[k])
                
                # Relative Change in STD
                if running_std[k - 1] > 1e-9: # Compare to previous STD
                    relative_change_std[k] = np.abs((running_std[k] - running_std[k - 1]) / running_std[k - 1])
                
                # Relative Change in CV
                if running_cv[k - 1] > 1e-9: # Compare to previous CV
                    relative_change_cv[k] = np.abs((running_cv[k] - running_cv[k - 1]) / running_cv[k - 1])

            # --- 4. Compile Results ---
            
            temp_df = pd.DataFrame({
                'draw': np.arange(1, N + 1),
                'metric_type': metric,
                'running_mean': running_mean,
                'running_std': running_std,
                'running_cv': running_cv,
                'relative_change_mean': relative_change_mean,
                'relative_change_std': relative_change_std,
                'relative_change_cv': relative_change_cv
            })
            
            for dim, value in zip(group_dims, group_key):
                temp_df[dim] = value
                
            stability_results.append(temp_df)

    stability_df = pd.concat(stability_results, ignore_index=True)

    return stability_df