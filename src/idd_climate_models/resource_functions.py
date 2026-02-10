import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from idd_climate_models.climate_file_functions import is_curvilinear_grid
import idd_climate_models.constants as rfc

# ============================================================================
# RESOURCE PREDICTION FUNCTIONS (Empirical Models from SLURM Data Analysis)
# ============================================================================

# Model parameters from analyze_job_resources.ipynb
MEMORY_BUFFER = 0.25          # 25% safety margin
TIME_BUFFER = 0.25            # 25% safety margin

# Memory model: Fitted to max memory per duration (R² = 0.9988)
MEMORY_INTERCEPT = 2.4066     # GiB
MEMORY_SLOPE = 0.5090         # GiB per year
MAX_MEMORY_RESIDUAL = 1.3232  # Worst under-prediction (GiB)

# Runtime model: Fitted to max runtime per total_storms for DRAWS_PER_BATCH=2 (R² = 0.9994)
RUNTIME_INTERCEPT = 29.1829   # minutes
RUNTIME_SLOPE = 0.392948      # minutes per storm
MAX_RUNTIME_RESIDUAL = 6.5885 # Worst under-prediction (minutes)


def predict_memory_requirement_gib(duration_years: int) -> float:
    """
    Predict memory requirement based on time period duration.
    
    Memory usage is driven by duration (not total storms), as the entire
    time period's data must be held in memory during processing.
    
    Formula: ((regression_prediction) + max_residual) × (1 + buffer)
    This accounts for worst-case deviation from regression line.
    
    Args:
        duration_years: Length of time period in years
        
    Returns:
        Recommended memory allocation in GiB
    """
    regression_prediction = MEMORY_INTERCEPT + MEMORY_SLOPE * duration_years
    x = regression_prediction + MAX_MEMORY_RESIDUAL
    buffered_memory = x * (1 + MEMORY_BUFFER)
    return buffered_memory


def predict_runtime_requirement_minutes(total_storms: float, draws_per_batch: int = 2) -> float:
    """
    Predict runtime based on total storm count and batch size.
    
    Runtime scales linearly with total storms (storms are processed sequentially)
    and with draws_per_batch (more draws = proportionally more work).
    
    Formula: ((regression_prediction) + max_residual) × (batch/2) × (1 + buffer)
    This accounts for worst-case deviation and scales with batch size.
    
    Args:
        total_storms: Total number of storms in the dataset
        draws_per_batch: Number of draws to process per batch (default=2)
        
    Returns:
        Estimated runtime in minutes
    """
    # Base runtime for draws_per_batch=2
    regression_prediction = RUNTIME_INTERCEPT + RUNTIME_SLOPE * total_storms
    x = regression_prediction + MAX_RUNTIME_RESIDUAL
    
    # Scale by actual batch size (linear scaling)
    scaled = x * (draws_per_batch / 2)
    
    # Add safety buffer
    buffered_runtime = scaled * (1 + TIME_BUFFER)
    
    return buffered_runtime


# ============================================================================
# RESOURCE ESTIMATION FUNCTIONS 
# ============================================================================

def get_file_size_gb(file_path: Path) -> float:
    """Gets file size in GB."""
    return os.path.getsize(file_path) / (1024**3)

from typing import Dict, Any

def get_resource_info(
    file_path: Path,
    representative: str = 'first',
    num_files: int = 1,
    REQUIRED_MEM_FACTOR: float = 12.0,  # Increased from 8.0 to account for .compute() loading all data
    MIN_MEM_GB: float = 8.0,  # Increased minimum for monthly data (hus/ta failed at 4GB, succeeded at 6GB)
    MAX_MEM_GB: float = 120.0
) -> tuple[Dict, bool]:
    """
    Allocates resources based on estimated total file size (rep_size * bin_size).
    """
    
    file_size_gb, _ = get_rep_file_size_gb(file_path=file_path, representative=representative)
    size_gb = file_size_gb * num_files  # Total estimated size for the time bin
    # 1. Memory Calculation
    required_mem_gb = int(size_gb * REQUIRED_MEM_FACTOR) + 4

    final_mem_value = min(MAX_MEM_GB, max(MIN_MEM_GB, required_mem_gb))
    memory = f"{int(final_mem_value)}G"

    
    # 2. Runtime/Core Calculation (Adjusted from previous tiers for safety)
    if final_mem_value < 10.0:
        runtime = "15m"  
        cores = 2
    elif final_mem_value < 50.0:
        runtime = "30m"  
        cores = 4
    else: 
        runtime = "1h"
        cores = 4

    cores = 4
    resource_request = {
            "memory": memory,
            "cores": cores,
            "runtime": runtime
        }

    return resource_request

def get_rep_file_size_gb(file_path: Path, representative: str = 'first') -> tuple[float, Path]:
    """
    Gets the file size in GB for a representative file in a directory.
    """
    if file_path.is_dir():
        all_input_files = sorted(os.listdir(file_path))
        if not all_input_files:
            raise FileNotFoundError(f"No files found in directory: {file_path}")
        if representative == 'first':
            representative_file_name = all_input_files[0]
        elif representative == 'last':
            representative_file_name = all_input_files[-1]
        else:
            raise ValueError("Representative must be 'first' or 'last'")
        full_file_path = file_path / representative_file_name
    else:
        full_file_path = file_path

    return get_file_size_gb(full_file_path), full_file_path


# ============================================================================
# DYNAMIC RESOURCE ALLOCATION FOR TC-RISK PIPELINE
# ============================================================================

# Cache for time bins wide dataframe
_TIME_BINS_WIDE_CACHE = None

def _load_time_bins_wide_for_resources():
    """Load and cache the wide time bins file with basin-specific storm counts.
    
    Tries to load the chunked wide file (max_bin_5) first if it exists,
    otherwise falls back to the original wide file.
    """
    global _TIME_BINS_WIDE_CACHE
    if _TIME_BINS_WIDE_CACHE is None:
        # Try loading the chunked wide file first (created by get_time_bins_path with max_duration=5)
        chunked_wide_path = rfc.TIME_BINS_WIDE_DF_PATH.parent / 'bayespoisson_time_bins_wide_max_bin_5.csv'
        
        if chunked_wide_path.exists():
            _TIME_BINS_WIDE_CACHE = pd.read_csv(chunked_wide_path)
        else:
            _TIME_BINS_WIDE_CACHE = pd.read_csv(rfc.TIME_BINS_WIDE_DF_PATH)
    return _TIME_BINS_WIDE_CACHE


def predict_memory_gib(duration_years: float) -> float:
    """
    Predict memory requirements in GiB for TC-risk processing based on time period duration.
    
    Uses a linear regression model fitted to actual SLURM usage data:
        memory_gib = intercept + slope × duration_years
    
    Adds max residual (worst under-prediction) and 25% buffer for conservatism.
    
    Model Performance:
        - R² = 0.9770 (excellent fit)
        - Fitted to 1043 completed jobs
        - Max residual = 1.32 GiB
    
    Args:
        duration_years: Length of time period in years (e.g., 5 for 2015-2019)
    
    Returns:
        Predicted memory requirement in GiB (rounded up to next integer)
    
    Example:
        >>> predict_memory_gib(5)  # 5-year period
        8  # GiB
        >>> predict_memory_gib(10)  # 10-year period
        13  # GiB
    """
    # Fitted parameters from SLURM usage analysis
    INTERCEPT = 2.41
    SLOPE = 0.5090
    MAX_RESIDUAL = 1.32  # GiB, worst under-prediction
    BUFFER = 0.25  # 25% safety margin
    
    # Base prediction from regression
    base_prediction = INTERCEPT + SLOPE * duration_years
    
    # Add max residual (conservative: covers worst case)
    conservative_prediction = base_prediction + MAX_RESIDUAL
    
    # Apply buffer (e.g., 25% extra headroom)
    final_prediction = conservative_prediction * (1.0 + BUFFER)
    
    # Round up to ensure sufficient memory
    return int(np.ceil(final_prediction))


def predict_runtime_minutes(total_storms: float, draws_per_batch: int = 2) -> float:
    """
    Predict runtime in minutes for TC-risk processing based on total storm count.
    
    Uses a linear regression model fitted to actual SLURM usage data:
        runtime_minutes = intercept + slope × total_storms
    
    Model fitted to DRAWS_PER_BATCH=2 baseline. Scales linearly for other batch sizes.
    Adds max residual (worst under-prediction) and 25% buffer for conservatism.
    
    Model Performance:
        - R² = 0.9429 (good fit)
        - Fitted to 1043 completed jobs with DRAWS_PER_BATCH=2
        - Max residual = 6.59 minutes
    
    Args:
        total_storms: Total storms to process (storms_per_year × duration_years)
        draws_per_batch: Number of draws per batch (default=2, baseline for model)
    
    Returns:
        Predicted runtime in minutes (rounded up to next integer)
    
    Example:
        >>> predict_runtime_minutes(100, draws_per_batch=2)  # 100 storms, 2 draws
        67  # minutes
        >>> predict_runtime_minutes(100, draws_per_batch=5)  # 100 storms, 5 draws
        167  # minutes (scales 5/2 = 2.5x)
    """
    # Fitted parameters from SLURM usage analysis (for DRAWS_PER_BATCH=2 baseline)
    INTERCEPT = 29.18
    SLOPE = 0.3929
    MAX_RESIDUAL = 6.59  # minutes, worst under-prediction
    BUFFER = 0.25  # 25% safety margin
    BASELINE_BATCH_SIZE = 2  # Model was fitted to batch=2 jobs
    
    # Base prediction from regression (for batch=2)
    base_prediction = INTERCEPT + SLOPE * total_storms
    
    # Add max residual (conservative: covers worst case)
    conservative_prediction = base_prediction + MAX_RESIDUAL
    
    # Scale for different batch sizes (linear scaling)
    batch_scaled = conservative_prediction * (draws_per_batch / BASELINE_BATCH_SIZE)
    
    # Apply buffer (e.g., 25% extra headroom)
    final_prediction = batch_scaled * (1.0 + BUFFER)
    
    # Round up to ensure sufficient time
    return int(np.ceil(final_prediction))


# ============================================================================
# REGRESSION-BASED BATCH SIZING AND RUNTIME ESTIMATION
# ============================================================================

class BatchSizeRegression:
    """
    Encapsulates regression model for batch size optimization based on storm counts.
    
    The model predicts minutes_per_draw = intercept + slope × total_storms
    Then adds a conservative buffer = (100 + buffer_pct)% of max residual
    """
    
    def __init__(self):
        self.slope: Optional[float] = None
        self.intercept: Optional[float] = None
        self.r_squared: Optional[float] = None
        self.buffer: Optional[float] = None
        self.buffer_pct: Optional[float] = None
        self.model: Optional[LinearRegression] = None
        
    def fit(self, complete_df: pd.DataFrame, buffer_pct: float = 10.0) -> None:
        """
        Fit linear regression model from completed task data.
        
        Args:
            complete_df: DataFrame with columns 'total_storms' and 'median_minutes_per_draw'
            buffer_pct: Percentage buffer to add to max residual (e.g., 10 = 110% of max residual)
        """
        X = complete_df[['total_storms']].values
        y = complete_df['median_minutes_per_draw'].values
        
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        self.slope = self.model.coef_[0]
        self.intercept = self.model.intercept_
        self.r_squared = self.model.score(X, y)
        
        # Calculate conservative buffer
        y_pred = self.model.predict(X)
        residuals = y - y_pred
        max_residual = residuals.max()
        
        self.buffer_pct = buffer_pct
        self.buffer = (1.0 + buffer_pct / 100.0) * max_residual
        
    def predict_minutes_per_draw(self, total_storms: float, conservative: bool = True) -> float:
        """
        Predict minutes per draw for a given total storm count.
        
        Args:
            total_storms: Total storms in the batch (storms_per_year × period_length)
            conservative: If True, add buffer. If False, use raw regression prediction.
        
        Returns:
            Predicted minutes per draw
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        base_prediction = self.intercept + self.slope * total_storms
        
        if conservative:
            return base_prediction + self.buffer
        else:
            return base_prediction
    
    def calculate_optimal_batch_size(
        self,
        storms_per_year: float,
        period_length: int,
        target_minutes: float = 110.0,
        safety_factor: float = 1.05,
        min_batch_size: int = 1,
        max_batch_size: int = 250
    ) -> Tuple[int, float]:
        """
        Calculate optimal batch size to hit target runtime.
        
        Args:
            storms_per_year: Average storms per year for this basin
            period_length: Number of years in time period
            target_minutes: Target runtime in minutes (default 110 = 2hr with buffer)
            safety_factor: Additional safety multiplier (default 1.05 = 5% buffer)
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
        
        Returns:
            Tuple of (optimal_batch_size, predicted_minutes_per_draw)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Total storms per draw = storms_per_year × period_length
        total_storms_per_draw = storms_per_year * period_length
        
        # Predicted minutes per draw (with buffer)
        predicted_minutes = self.predict_minutes_per_draw(
            total_storms_per_draw, 
            conservative=True
        ) * safety_factor
        
        # Optimal batch size to hit target
        optimal_batch_size = target_minutes / predicted_minutes
        
        # Clip to valid range and convert to int
        optimal_batch_size = int(np.clip(optimal_batch_size, min_batch_size, max_batch_size))
        
        return optimal_batch_size, predicted_minutes
    
    def calculate_batch_runtime(
        self,
        storms_per_year: float,
        period_length: int,
        num_draws: int,
        safety_factor: float = 1.05
    ) -> float:
        """
        Calculate expected runtime in minutes for a batch of draws.
        
        Args:
            storms_per_year: Average storms per year for this basin
            period_length: Number of years in time period
            num_draws: Number of draws in this batch
            safety_factor: Additional safety multiplier (default 1.05 = 5% buffer)
        
        Returns:
            Expected runtime in minutes
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Total storms per draw
        total_storms_per_draw = storms_per_year * period_length
        
        # Predicted minutes per draw (with buffer)
        minutes_per_draw = self.predict_minutes_per_draw(
            total_storms_per_draw,
            conservative=True
        ) * safety_factor
        
        # Total batch runtime
        return minutes_per_draw * num_draws
    
    def format_runtime_for_jobmon(self, minutes: float) -> str:
        """
        Convert minutes to jobmon runtime string format.
        
        Args:
            minutes: Runtime in minutes
        
        Returns:
            String like "30m", "2h", "90m"
        """
        if minutes < 60:
            return f"{int(np.ceil(minutes))}m"
        else:
            hours = minutes / 60
            if hours == int(hours):
                return f"{int(hours)}h"
            else:
                # Keep as minutes if not a clean hour value
                return f"{int(np.ceil(minutes))}m"
    
    def get_summary(self) -> str:
        """Return formatted summary of the regression model."""
        if self.model is None:
            return "Model not fitted."
        
        summary = "="*80 + "\n"
        summary += "BATCH SIZE REGRESSION MODEL\n"
        summary += "="*80 + "\n"
        summary += f"minutes_per_draw = {self.intercept:.2f} + {self.slope:.5f} × total_storms\n"
        summary += f"Buffer: {self.buffer_pct:.1f}% of max residual = +{self.buffer:.2f} minutes\n"
        summary += f"R² = {self.r_squared:.4f}\n"
        summary += f"\nInterpretation:\n"
        summary += f"  Base overhead: {self.intercept:.1f} minutes per draw\n"
        summary += f"  Per-storm cost: {self.slope:.5f} minutes per storm\n"
        summary += f"  Model explains {self.r_squared*100:.1f}% of variance\n"
        summary += f"  Conservative buffer: +{self.buffer:.2f} minutes\n"
        return summary


def get_storms_per_year(
    model: str,
    variant: str,
    scenario: str,
    time_period: str,
    basin: str
) -> float:
    """
    Look up average storms per year for a given model/variant/scenario/time_period/basin.
    
    Args:
        model: Model name
        variant: Variant name
        scenario: Scenario name
        time_period: Time period string like "1970-1999"
        basin: Basin code like "EP", "NA"
    
    Returns:
        Average storms per year for this basin
    """
    time_bins_wide = _load_time_bins_wide_for_resources()
    
    start_year, end_year = map(int, time_period.split('-'))
    
    # Find matching row
    mask = (
        (time_bins_wide['model'] == model) &
        (time_bins_wide['variant'] == variant) &
        (time_bins_wide['scenario'] == scenario) &
        (time_bins_wide['start_year'] == start_year) &
        (time_bins_wide['end_year'] == end_year)
    )
    
    matching = time_bins_wide[mask]
    
    if len(matching) == 0:
        raise ValueError(
            f"No time bins data for {model}/{variant}/{scenario}/{time_period}"
        )
    
    row = matching.iloc[0]
    basin_col = f"{basin}_int"
    
    if basin_col not in row:
        raise ValueError(f"Basin {basin} not found in time bins data")
    
    return float(row[basin_col])


def calculate_batch_splits_for_basin(
    model: str,
    variant: str,
    scenario: str,
    time_period: str,
    basin: str,
    total_draws: int,
    regression_model: BatchSizeRegression,
    optimal_time_per_task: float = 110.0,
    safety_factor: float = 1.05,
    min_batch_size: int = 1,
    max_batch_size: int = 250
) -> Tuple[int, int, int]:
    """
    Calculate batch sizing for a basin: optimal batch size, number of batches, last batch size.
    
    Args:
        model: Model name
        variant: Variant name
        scenario: Scenario name
        time_period: Time period string like "1970-1999"
        basin: Basin code like "EP", "NA"
        total_draws: Total number of draws (e.g., 250)
        regression_model: Fitted BatchSizeRegression instance
        optimal_time_per_task: Target runtime in minutes (default 110)
        safety_factor: Safety multiplier for runtime prediction (default 1.05)
        min_batch_size: Minimum batch size (default 1)
        max_batch_size: Maximum batch size (default 250)
    
    Returns:
        Tuple of (optimal_batch_size, num_batches, last_batch_size)
    """
    # Get storms per year
    storms_per_year = get_storms_per_year(model, variant, scenario, time_period, basin)
    
    # Parse period length
    start_year, end_year = map(int, time_period.split('-'))
    period_length = end_year - start_year + 1
    
    # Calculate optimal batch size
    optimal_batch_size, _ = regression_model.calculate_optimal_batch_size(
        storms_per_year=storms_per_year,
        period_length=period_length,
        target_minutes=optimal_time_per_task,
        safety_factor=safety_factor,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size
    )
    
    # Calculate number of batches and last batch size
    num_batches = int(np.ceil(total_draws / optimal_batch_size))
    last_batch_size = total_draws - (num_batches - 1) * optimal_batch_size
    
    return optimal_batch_size, num_batches, last_batch_size


# ============================================================================
# LEGACY RESOURCE ALLOCATION FUNCTIONS (kept for backward compatibility)
# ============================================================================


def get_level2_resources(time_period: str, variable: str, frequency: str) -> Dict[str, Any]:
    """
    Calculate resources for Level 2 (process variable) based on time period length.
    
    Args:
        time_period: str like "1970-1999"
        variable: str like "ua", "tos"
        frequency: str like "day", "Amon"
    
    Returns:
        dict: Resource allocation with keys: memory, cores, runtime
    """
    start_year, end_year = map(int, time_period.split('-'))
    period_length = end_year - start_year + 1
    
    # Base resources by frequency
    if 'day' in frequency.lower():
        # Daily data scales with time period length
        if period_length <= 10:
            mem, cores, runtime = "80G", 6, "30m"
        elif period_length <= 20:
            mem, cores, runtime = "120G", 8, "1h"
        elif period_length <= 30:
            mem, cores, runtime = "150G", 8, "90m"
        else:
            mem, cores, runtime = "180G", 8, "2h"
    else:
        # Monthly data is lighter
        if period_length <= 20:
            mem, cores, runtime = "12G", 4, "10m"
        elif period_length <= 40:
            mem, cores, runtime = "16G", 6, "15m"
        else:
            mem, cores, runtime = "20G", 8, "20m"
    
    return {
        "memory": mem,
        "cores": cores,
        "runtime": runtime
    }


def get_level3_resources(time_period: str) -> Dict[str, Any]:
    """
    Calculate resources for Level 3 (global TC-risk) based on time period length.
    
    Args:
        time_period: str like "1970-1999"
    
    Returns:
        dict: Resource allocation with keys: memory, cores, runtime
    """
    start_year, end_year = map(int, time_period.split('-'))
    period_length = end_year - start_year + 1
    
    # Conservative fixed allocation for short periods (no training data below 9 years)
    # Based on analysis in analyze_global_job_resources.ipynb showing extrapolation risk
    if period_length <= 10:
        mem, runtime = "10G", "10m"
    elif period_length <= 20:
        mem, runtime = "40G", "1h"
    elif period_length <= 30:
        mem, runtime = "50G", "90m"
    else:
        mem, runtime = "60G", "2h"
    
    cores = rfc.tc_risk_n_procs + 1  # Always use configured cores
    
    return {
        "memory": mem,
        "cores": cores,
        "runtime": runtime
    }


def get_level4_resources(
    model: str, 
    variant: str, 
    scenario: str, 
    time_period: str, 
    basin: str, 
    draws_per_batch: int,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Calculate resources for Level 4 (basin TC-risk) using empirical models.
    
    Uses prediction functions based on SLURM data analysis:
    - Memory: Depends on time period duration (years of data in memory)
    - Runtime: Depends on total storms and scales linearly with draws_per_batch
    
    Both models include max residual correction and 25% safety buffers.
    
    Args:
        model: Model name
        variant: Variant name
        scenario: Scenario name
        time_period: Time period string like "1970-1999"
        basin: Basin code like "EP", "NA"
        draws_per_batch: Number of draws in this batch (e.g., 2)
        verbose: Print calculation details
    
    Returns:
        dict: Resource allocation with keys: memory, cores, runtime
    """
    # Load wide time bins to get basin-specific storm counts
    time_bins_wide = _load_time_bins_wide_for_resources()
    
    start_year, end_year = map(int, time_period.split('-'))
    period_length = end_year - start_year + 1
    
    # Find matching row
    mask = (
        (time_bins_wide['model'] == model) &
        (time_bins_wide['variant'] == variant) &
        (time_bins_wide['scenario'] == scenario) &
        (time_bins_wide['start_year'] == start_year) &
        (time_bins_wide['end_year'] == end_year)
    )
    
    matching = time_bins_wide[mask]
    
    if len(matching) == 0:
        # Fallback to conservative resources
        if verbose:
            print(f"  WARNING: No time bins data for {model}/{variant}/{scenario}/{time_period}, using default resources")
        return {
            "memory": "35G",
            "cores": rfc.tc_risk_n_procs + 1,
            "runtime": "4h"
        }
    
    row = matching.iloc[0]
    basin_col = f"{basin}_int"
    
    if basin_col not in row:
        if verbose:
            print(f"  WARNING: Basin {basin} not found in time bins, using default resources")
        return {
            "memory": "35G",
            "cores": rfc.tc_risk_n_procs + 1,
            "runtime": "4h"
        }
    
    storms_per_year = float(row[basin_col])
    
    # Calculate total storms this job will process
    total_storms = storms_per_year * period_length
    
    # Use empirical prediction models based on SLURM data analysis
    # Memory depends on duration (years of data held in memory)
    memory_gib = predict_memory_requirement_gib(period_length)
    mem = f"{int(np.ceil(memory_gib))}G"
    
    # Runtime depends on total storms and scales with batch size
    runtime_minutes = predict_runtime_requirement_minutes(total_storms, draws_per_batch)
    
    # Convert to jobmon format (round up to nearest hour)
    runtime_hours = int(np.ceil(runtime_minutes / 60))
    runtime = f"{runtime_hours}h"
    
    if verbose:
        print(f"  {basin}: {period_length}yr × {storms_per_year:.1f}storms/yr × {draws_per_batch}draws")
        print(f"    → {total_storms:.0f} total storms → {mem} memory, {runtime} runtime ({runtime_minutes:.1f}min)")
    
    cores = rfc.tc_risk_n_procs + 1
    
    return {
        "memory": mem,
        "cores": cores,
        "runtime": runtime
    }


def _get_tiered_runtime(total_storms_in_batch: float) -> str:
    """Helper function for legacy tiered runtime allocation."""
    if total_storms_in_batch <= 500:
        return "1h"
    elif total_storms_in_batch <= 1000:
        return "2h"
    elif total_storms_in_batch <= 2000:
        return "4h"
    elif total_storms_in_batch <= 4000:
        return "6h"
    else:
        return "8h"


def get_level4_resources_adaptive(
    model: str,
    variant: str,
    scenario: str,
    time_period: str,
    basin: str,
    num_draws: int,
    regression_model: BatchSizeRegression,
    optimal_time_per_task: float = 110.0,
    safety_factor: float = 1.05,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Calculate adaptive resources for Level 4 (basin TC-risk) using regression model.
    
    This function calculates resources based on actual runtime predictions from
    the regression model, accounting for the specific number of draws in this batch.
    
    Args:
        model: Model name
        variant: Variant name
        scenario: Scenario name
        time_period: Time period string like "1970-1999"
        basin: Basin code like "EP", "NA"
        num_draws: Number of draws in this specific batch
        regression_model: Fitted BatchSizeRegression instance
        optimal_time_per_task: Target runtime in minutes (default 110)
        safety_factor: Safety multiplier for runtime prediction (default 1.05)
        verbose: Print calculation details
    
    Returns:
        dict: Resource allocation with keys: memory, cores, runtime
    """
    try:
        # Get storms per year
        storms_per_year = get_storms_per_year(model, variant, scenario, time_period, basin)
        
        # Parse period length
        start_year, end_year = map(int, time_period.split('-'))
        period_length = end_year - start_year + 1
        
        # Calculate expected runtime for this batch
        runtime_minutes = regression_model.calculate_batch_runtime(
            storms_per_year=storms_per_year,
            period_length=period_length,
            num_draws=num_draws,
            safety_factor=safety_factor
        )
        
        # Format runtime for jobmon
        runtime_str = regression_model.format_runtime_for_jobmon(runtime_minutes)
        
        # Calculate memory based on total storms in batch
        total_storms_in_batch = storms_per_year * period_length * num_draws
        
        if total_storms_in_batch <= 500:
            memory = "35G"
        elif total_storms_in_batch <= 1000:
            memory = "40G"
        elif total_storms_in_batch <= 2000:
            memory = "45G"
        elif total_storms_in_batch <= 4000:
            memory = "50G"
        else:
            memory = "60G"
        
        cores = rfc.tc_risk_n_procs + 1
        
        if verbose:
            print(f"  {basin} batch: {num_draws} draws, {storms_per_year:.1f} storms/yr, "
                  f"{period_length}yr → {runtime_minutes:.1f}min ({runtime_str}), {memory}")
        
        return {
            "memory": memory,
            "cores": cores,
            "runtime": runtime_str
        }
        
    except Exception as e:
        # Fallback to conservative resources
        if verbose:
            print(f"  WARNING: Error calculating resources for {model}/{basin}/{time_period}: {e}")
        return {
            "memory": "40G",
            "cores": rfc.tc_risk_n_procs + 1,
            "runtime": "4h"
        }