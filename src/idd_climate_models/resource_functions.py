import os
from pathlib import Path
from typing import Dict, Any

# ============================================================================
# RESOURCE ESTIMATION FUNCTIONS 
# ============================================================================

def get_file_size_gb(file_path: Path) -> float:
    """Gets file size in GB."""
    return os.path.getsize(file_path) / (1024**3)

from typing import Dict, Any

def get_resource_tier(file_size_gb: float, REQUIRED_MEM_FACTOR: float = 3.0,
                      MIN_MEM_GB: float = 8.0, MAX_MEM_GB: float = 80.0) -> Dict[str, Any]:
    """
    Allocates resources based on estimated total file size (rep_size * bin_size).
    """
    
    # 1. Memory Calculation
    required_mem_gb = int(file_size_gb * REQUIRED_MEM_FACTOR) + 4

    final_mem_value = min(MAX_MEM_GB, max(MIN_MEM_GB, required_mem_gb))
    memory = f"{int(final_mem_value)}G"

    # 2. Runtime/Core Calculation (Adjusted from previous tiers for safety)
    if file_size_gb < 100.0:
        runtime = "15m"  
        cores = 2
    elif file_size_gb < 400.0:
        runtime = "30m"  
        cores = 4
    else: 
        runtime = "1h"
        cores = 4
        
    return {
        "memory": memory,
        "cores": cores,
        "runtime": runtime
    }

def get_rep_file_size_gb(file_path: Path, representative: str = 'first') -> float:
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

    return get_file_size_gb(full_file_path)