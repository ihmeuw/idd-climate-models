import os
from pathlib import Path
from typing import Dict, Any

from idd_climate_models.climate_file_functions import is_curvilinear_grid

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