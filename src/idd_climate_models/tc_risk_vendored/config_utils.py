"""
Configuration utilities for TC-risk vendored code.

NO NAMELIST IMPORTS! All configuration via JSON/dict.
"""
import json
from pathlib import Path


def load_default_config():
    """Load default configuration from JSON file."""
    config_path = Path(__file__).parent / 'default_config.json'
    with open(config_path, 'r') as f:
        return json.load(f)


def create_tc_risk_config(
    model: str,
    variant: str,
    scenario: str,
    time_period: str,
    basin: str,
    base_directory: str,
    output_directory: str,
    n_procs: int = 10,
    dataset_type: str = "CMIP6"
):
    """
    Create a TC-risk configuration dictionary for a specific job.
    
    Args:
        model: Climate model name (e.g., 'CMCC-ESM2')
        variant: Variant label (e.g., 'r1i1p1f1')
        scenario: Scenario name (e.g., 'historical', 'ssp245')
        time_period: Time period string (e.g., '1986-2014')
        basin: Basin code (e.g., 'GL', 'NA')
        base_directory: Path to input climate data
        output_directory: Path to output directory
        n_procs: Number of processes for parallelization
        dataset_type: 'CMIP6' or 'ERA5'
    
    Returns:
        dict: Complete configuration dictionary ready to pass to vendored functions
    """
    # Load defaults
    config = load_default_config()
    
    # Parse time period
    start_year, end_year = map(int, time_period.split('-'))
    
    # Create experiment prefix
    exp_prefix = f"{model}_{scenario}_{variant}"
    
    # Update configuration
    config.update({
        'base_directory': base_directory,
        'output_directory': output_directory,
        'exp_prefix': exp_prefix,
        'exp_name': basin,
        'start_year': start_year,
        'start_month': 1,
        'end_year': end_year,
        'end_month': 12,
        'n_procs': n_procs,
        'dataset_type': dataset_type,
        'basin': basin,
        'model': model,
        'variant': variant,
        'scenario': scenario,
        'time_period': time_period,
    })
    
    return config


def save_config(config_dict, output_path):
    """Save configuration dictionary to JSON file (for debugging/logging)."""
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
