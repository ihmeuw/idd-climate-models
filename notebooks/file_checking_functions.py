import os
import re
import json
from datetime import datetime, timedelta

import idd_climate_models.constants as rfc
from idd_climate_models.utility_functions import get_subfolders, get_nc_files, int_to_date, is_monthly
from idd_climate_models.model_structure_utils import iterate_model_files
from idd_climate_models.processing_log import update_processing_log

# ============================================================================
# FILE CHECKING FUNCTIONS
# ============================================================================

def check_processed_files_exist(model, variant, scenario, variable, grid, frequency, 
                                file_path, processed_data_path, data_source="cmip6"):
    """Check if all expected yearly processed files exist."""
    dest_dir = os.path.join(processed_data_path, data_source, model, variant, scenario, variable, grid, frequency)
    if not os.path.exists(dest_dir):
        return False
    
    monthly = is_monthly(frequency)
    date_pattern = r"(\d{6})-(\d{6})" if monthly else r"(\d{8})-(\d{8})"
    match = re.search(date_pattern, os.path.basename(file_path))
    if not match:
        return False
    
    try:
        start_date = int_to_date(int(match.group(1)), monthly)
        end_date = int_to_date(int(match.group(2)), monthly)
    except ValueError:
        return False
    
    MIN_YEAR, MAX_YEAR = 1950, 2100
    start_year = max(start_date.year, MIN_YEAR)
    end_year = min(end_date.year, MAX_YEAR)
    
    if start_year > end_year:
        return True
    
    base_name = os.path.basename(file_path)
    for year in range(start_year, end_year + 1):
        if monthly:
            expected_fname = re.sub(r'_(\d{6})-(\d{6})\.nc$', f'_{year}01-{year}12.nc', base_name)
        else:
            expected_fname = re.sub(r'_(\d{8})-(\d{8})\.nc$', f'_{year}0101-{year}1231.nc', base_name)
        if not os.path.exists(os.path.join(dest_dir, expected_fname)):
            return False
    return True


def filter_already_processed(validation_results, processed_data_path, data_source="cmip6", 
                            rerun=False, verbose=False):
    """Filter out tasks that have already been processed."""
    if rerun:
        if verbose:
            print("Rerun=True: Returning all tasks")
        return validation_results
    
    if verbose:
        print("Filtering already processed files...")
        print("=" * 80)
    
    filtered_results = {}
    stats = {'total': 0, 'remaining': 0, 'variants_done': []}
    
    for model, model_data in validation_results.items():
        if not model_data.get('complete'):
            continue
        
        # Use the iterator to simplify checking
        files_to_process = []
        variant_stats = {'total': 0, 'scenarios': set(), 'variables': set()}
        
        # Updated to use the new iterate_model_files function
        for model_name, variant, scenario, variable, grid, frequency, file_info in iterate_model_files({model: model_data}):
            # Extract file path and fill_required from file_info
            if isinstance(file_info, dict):
                file_path = file_info.get('path', '')
                fill_required = file_info.get('fill_required', False)
            else:
                file_path = file_info
                fill_required = False
            
            stats['total'] += 1
            variant_stats['total'] += 1
            variant_stats['scenarios'].add(scenario)
            variant_stats['variables'].add(variable)
            
            if not check_processed_files_exist(model, variant, scenario, variable, grid, 
                                              frequency, file_path, processed_data_path, data_source):
                stats['remaining'] += 1
                files_to_process.append({
                    'variant': variant, 'scenario': scenario, 'variable': variable, 'grid': grid,
                    'frequency': frequency, 'path': file_path, 'fill_required': fill_required
                })

        # If no files are left to process for this variant, log it as complete
        if variant_stats['total'] > 0 and not files_to_process and model_data.get('variants'):
            # This assumes one variant per model_data structure in this loop context
            variant_name = list(model_data['variants'].keys())[0]
            stats['variants_done'].append({
                'model': model, 'variant': variant_name,
                'scenarios': list(variant_stats['scenarios']),
                'variables': list(variant_stats['variables'])
            })
            continue

        # Rebuild the nested dictionary from the list of files to process
        if files_to_process:
            filtered_model = {'complete': True, 'issues': [], 'variants': {}}
            for f in files_to_process:
                v_level = filtered_model['variants'].setdefault(f['variant'], {'complete': True, 'issues': [], 'scenarios': {}})
                s_level = v_level['scenarios'].setdefault(f['scenario'], {'complete': True, 'issues': [], 'variables': {}})
                var_level = s_level['variables'].setdefault(f['variable'], {'complete': True, 'issues': [], 'grids': {}})
                g_level = var_level['grids'].setdefault(f['grid'], {'complete': True, 'issues': [], 'frequencys': {}})
                t_level = g_level['frequencys'].setdefault(f['frequency'], {'complete': True, 'issues': [], 'files': []})
                t_level['files'].append({'path': f['path'], 'fill_required': f['fill_required']})
            
            filtered_results[model] = filtered_model

    # Update log for completed variants
    for entry in stats['variants_done']:
        update_processing_log(entry['model'], entry['variant'], entry['scenarios'], 
                            entry['variables'], processed_data_path, data_source)
    
    if verbose:
        print("=" * 80)
        print(f"Files: {stats['remaining']}/{stats['total']} remaining")
        print(f"Models with tasks: {len(filtered_results)}")
        if stats['variants_done']:
            print(f"Marked {len(stats['variants_done'])} variants complete")
    
    return filtered_results