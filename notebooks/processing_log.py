import os
import json
from datetime import datetime

import idd_climate_models.constants as rfc
from idd_climate_models.model_validation import validate_model

# ============================================================================
# PROCESSING LOG FUNCTIONS
# ============================================================================

def get_processing_log_path(processed_data_path, data_source="cmip6"):
    """Get the path to the processing log JSON file."""
    return os.path.join(processed_data_path, data_source, f"{data_source}_processing_log.json")


def load_processing_log(processed_data_path, data_source="cmip6"):
    """Load the processing log."""
    log_path = get_processing_log_path(processed_data_path, data_source)
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read processing log: {e}")
    return {'models': {}, 'last_updated': None}


def save_processing_log(log_data, processed_data_path, data_source="cmip6"):
    """Save the processing log to disk."""
    log_path = get_processing_log_path(processed_data_path, data_source)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save processing log: {e}")


def update_processing_log(model, variant, scenarios, variables, processed_data_path, data_source="cmip6"):
    """Update the processing log to mark a model/variant as fully processed."""
    log_data = load_processing_log(processed_data_path, data_source)
    if model not in log_data['models']:
        log_data['models'][model] = {'variants': {}}
    log_data['models'][model]['variants'][variant] = {
        'processed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'scenarios': sorted(scenarios),
        'variables': sorted(variables)
    }
    save_processing_log(log_data, processed_data_path, data_source)


def is_variant_processed(model, variant, processing_log):
    """Check if a model/variant is marked as fully processed."""
    return (model in processing_log.get('models', {}) and 
            variant in processing_log['models'][model].get('variants', {}))


def find_unprocessed_models(data_dir, processed_data_path, data_source="cmip6", verbose=True):
    """Find models that need processing by checking the processing log first."""
    processing_log = load_processing_log(processed_data_path, data_source)
    
    if verbose:
        processed_count = sum(len(m['variants']) for m in processing_log.get('models', {}).values())
        print(f"Processing log: {processed_count} variants marked as processed")
        print(f"Last updated: {processing_log.get('last_updated', 'Never')}")
        print("=" * 80)
    
    raw_data_dir = os.path.join(data_dir, "raw")
    source_path = os.path.join(raw_data_dir, data_source)
    
    if not os.path.exists(source_path):
        print(f"Error: Data source directory does not exist: {source_path}")
        return {}
    
    try:
        model_names = sorted([f.name for f in os.scandir(source_path) if f.is_dir()])
    except OSError as e:
        print(f"Error reading data source directory: {e}")
        return {}
    
    if verbose:
        print(f"Found {len(model_names)} models. Validating unprocessed variants...")
        print("=" * 80)
    
    all_models = {}
    stats = {'validated': 0, 'skipped': 0, 'complete': 0}
    
    for i, model_name in enumerate(model_names, 1):
        if verbose:
            print(f"[{i}/{len(model_names)}] {model_name}...", end=" ")
        
        model_result = validate_model(model_name, raw_data_dir, data_source)
        if not model_result.get('variants'):
            if verbose:
                print("✗ NO VARIANTS")
            continue
        
        filtered_variants = {
            vname: vdata for vname, vdata in model_result['variants'].items()
            if not is_variant_processed(model_name, vname, processing_log)
        }
        
        stats['skipped'] += len(model_result['variants']) - len(filtered_variants)
        stats['validated'] += len(filtered_variants)
        stats['complete'] += sum(1 for v in filtered_variants.values() if v.get('complete'))
        
        if filtered_variants:
            model_result['variants'] = filtered_variants
            all_models[model_name] = model_result
            status = "✓" if model_result['complete'] else "✗"
            if verbose:
                print(f"{status} ({len(filtered_variants)} unprocessed)")
        else:
            if verbose:
                print("⊙ ALL PROCESSED")
    
    if verbose:
        print("=" * 80)
        print(f"Models with unprocessed variants: {len(all_models)}")
        print(f"Variants: {stats['validated']} validated, {stats['skipped']} skipped")
        if stats['validated']:
            print(f"Complete: {stats['complete']}/{stats['validated']} ({100*stats['complete']/stats['validated']:.1f}%)")
    
    return all_models

import os
import re
import subprocess
from pathlib import Path
from datetime import datetime 

import idd_climate_models.constants as rfc
from idd_climate_models.utility_functions import is_monthly
from idd_climate_models.model_validation import validate_all_models
from idd_climate_models.processing_log import (
    find_unprocessed_reorganizations, 
    update_reorganization_log
)

# ============================================================================
# DATA REORGANIZATION FUNCTIONS  
# ============================================================================

# ...existing functions (get_time_bins, find_yearly_files_in_range, recombine_variable_files)...

def create_recombined_structure(validation_results, target_base_path, 
                               bin_size_years=5, data_source="cmip6", dry_run=False, verbose=True, rerun=False):
    """
    Create a new folder structure and recombine NetCDF files by variable within time bins.
    Uses the output from the yearly split processing (PROCESSED_DATA_PATH).
        
    Returns:
    --------
    dict : Summary of created structure
    """
    
    # Convert paths to Path objects for consistency
    target_base_path = Path(target_base_path)
    processed_data_path = Path(rfc.PROCESSED_DATA_PATH)
    
    # Filter out already processed bins (unless rerun=True)
    validation_results = find_unprocessed_reorganizations(
        validation_results, target_base_path, data_source, bin_size_years, 
        rerun=rerun, verbose=verbose
    )
    
    if not validation_results:
        if verbose:
            print("✅ All time bins have been reorganized! Nothing to do.")
        return {'created_dirs': 0, 'combined_files': 0, 'errors': [], 'structure': {}}
    
    # ...existing code...
    
    # Inside the processing loop, after successfully processing each bin:
    for bin_start, bin_end in time_bins:
        bin_name = f"{bin_start}-{bin_end}"
        
        # ...existing processing code...
        
        processed_variables = []
        bin_has_files = False
        
        for var_name, var_data in variables_data.items():
            # ...existing variable processing...
            
            if variable_files_in_bin:
                # ...existing recombination code...
                
                if success:
                    processed_variables.append(var_name)
                    bin_has_files = True
                    # ...existing success handling...
        
        # Update log after processing this bin
        if not dry_run and bin_has_files and processed_variables:
            update_reorganization_log(
                model_name, variant_name, scenario_name, frequency_name, bin_name, 
                processed_variables, target_base_path, data_source
            )
    
    # ...rest of existing code...


def reorganize_climate_data(target_dir, bin_size_years=5, data_source="cmip6", 
                           dry_run=True, verbose=True, rerun=False):
    """
    Complete workflow to validate models and reorganize processed yearly data.
    
    Parameters:
    -----------
    target_dir : str or Path
        Target directory for reorganized structure
    bin_size_years : int
        Years per time bin
    data_source : str
        Data source name
    dry_run : bool
        If True, only show what would be done
    verbose : bool
        Print detailed progress
    rerun : bool
        If True, reprocess all bins regardless of log
    """
    # ...existing validation code...
    
    summary = create_recombined_structure(
        validation_results, target_dir, 
        bin_size_years, data_source, dry_run, verbose, rerun
    )
    
    return validation_results, summary