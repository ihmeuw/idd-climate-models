import os
import re
import json
from datetime import datetime, timedelta

# ============================================================================
# VALIDATION RULES CONFIGURATION
# ============================================================================

VALIDATION_RULES = {
    'model': {
        'child_name': 'variants',
        'required_children': None,
        'exact_count': None,
    },
    'variant': {
        'child_name': 'scenarios',
        'required_children': ["historical", "ssp126", "ssp245", "ssp585"],
        'exact_count': None,
    },
    'scenario': {
        'child_name': 'variables',
        'required_children': ["ua", "va", "tos", "psl", "hus", "ta"],
        'exact_count': None,
    },
    'variable': {
        'child_name': 'grids',
        'required_children': None,
        'require_at_least_one': ['gn', 'gr', 'gr1', 'gr2'],
        'exact_count': None,
    },
    'grid': {
        'child_name': 'time_periods',
        'required_children': None,
        'exact_count': 1,
        'validator': lambda name: name == 'day' or 'mon' in name.lower()
    },
    'time': {
        'child_name': None,
        'date_ranges': {
            'historical': {'monthly': (197001, 201412), 'daily': (19700101, 20141231)},
            'ssp126': {'monthly': (201501, 210012), 'daily': (20150101, 21001231)},
            'ssp245': {'monthly': (201501, 210012), 'daily': (20150101, 21001231)},
            'ssp585': {'monthly': (201501, 210012), 'daily': (20150101, 21001231)}
        }
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def int_to_date(date_int, monthly=False):
    """Convert integer date representation to datetime object."""
    return datetime.strptime(str(date_int), "%Y%m" if monthly else "%Y%m%d")

def is_monthly(time_folder):
    """Check if time folder represents monthly data."""
    return 'mon' in time_folder.lower()

def get_children(path):
    """Get all subdirectories in a path."""
    try:
        return sorted([f.name for f in os.scandir(path) if f.is_dir()]), []
    except OSError as e:
        return None, [f"Cannot read directory: {e}"]

def get_nc_files(path):
    """Get all NetCDF files in a path."""
    try:
        return sorted([f for f in os.listdir(path) if f.endswith('.nc')]), []
    except OSError as e:
        return [], [f"Cannot read directory: {e}"]

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def check_folder_rules(level, children, rules):
    """Check if folder contents meet the rules for this level."""
    issues = []
    
    if rules.get('required_children'):
        missing = [c for c in rules['required_children'] if c not in children]
        if missing:
            issues.append(f"Missing required folders: {missing}")
    
    if rules.get('require_at_least_one'):
        required_options = rules['require_at_least_one']
        if not any(c in required_options for c in children):
            issues.append(f"Must have at least one of: {required_options}, found: {children}")
    
    if rules.get('exact_count') is not None and len(children) != rules['exact_count']:
        issues.append(f"Expected {rules['exact_count']} folder(s), found {len(children)}: {children}")
    
    if rules.get('validator'):
        for child in children:
            if not rules['validator'](child):
                issues.append(f"Invalid folder name: '{child}'")
    
    return issues

def extract_date_ranges(nc_files, model_name, scenario, variant, grid, var, time_folder):
    """Extract date ranges from NetCDF filenames."""
    issues = []
    monthly = is_monthly(time_folder)
    date_ranges = []
    date_pattern = r"(\d{6})-(\d{6})" if monthly else r"(\d{8})-(\d{8})"
    regex = rf"{var}_{time_folder}_{model_name}_{scenario}_{variant}_{grid}_{date_pattern}\.nc"
    
    for fname in nc_files:
        m = re.match(regex, fname)
        if not m:
            issues.append(f"File '{fname}' doesn't match naming convention")
            continue
        start, end = m.groups()
        try:
            int_to_date(int(start), monthly)
            int_to_date(int(end), monthly)
            date_ranges.append((int(start), int(end)))
        except ValueError as e:
            issues.append(f"File '{fname}' has invalid date format: {e}")
    
    return sorted(date_ranges, key=lambda x: x[0]), monthly, issues

def check_date_coverage(date_ranges, scenario, monthly, time_rules):
    """Check if date ranges cover required period with no gaps."""
    if not date_ranges:
        return False, ["No valid date ranges found"]
    
    issues = []
    range_type = 'monthly' if monthly else 'daily'
    expected_start, expected_end = time_rules['date_ranges'][scenario][range_type]
    actual_start, actual_end = date_ranges[0][0], date_ranges[-1][1]
    
    if actual_start > expected_start:
        issues.append(f"Coverage starts too late: {actual_start}, expected by: {expected_start}")
        return False, issues
    if actual_end < expected_end:
        issues.append(f"Coverage ends too early: {actual_end}, expected until: {expected_end}")
        return False, issues
    
    # Check for gaps
    for i in range(1, len(date_ranges)):
        prev_end = int_to_date(date_ranges[i-1][1], monthly)
        curr_start = int_to_date(date_ranges[i][0], monthly)
        
        if monthly:
            year = prev_end.year + (prev_end.month // 12)
            month = prev_end.month % 12 + 1
            next_expected = prev_end.replace(year=year, month=month) if month <= 12 else prev_end.replace(year=year+1, month=1)
            if next_expected != curr_start:
                issues.append(f"Gap between {prev_end.strftime('%Y%m')} and {curr_start.strftime('%Y%m')}")
                return False, issues
        else:
            if prev_end + timedelta(days=1) != curr_start:
                issues.append(f"Gap between {prev_end.strftime('%Y%m%d')} and {curr_start.strftime('%Y%m%d')}")
                return False, issues
    
    return True, issues

def validate_time_level(path, context, time_folder):
    """Validate the time level (leaf node with files)."""
    nc_files, issues = get_nc_files(path)
    if not nc_files:
        return {'complete': False, 'files': [], 'issues': issues + ["No NetCDF files found"]}
    
    date_ranges, monthly, naming_issues = extract_date_ranges(
        nc_files, context['model'], context['scenario'], context['variant'],
        context['grid'], context['variable'], time_folder
    )
    complete, coverage_issues = check_date_coverage(
        date_ranges, context['scenario'], monthly, VALIDATION_RULES['time']
    )
    
    return {
        'complete': complete and len(naming_issues) == 0,
        'files': [os.path.join(path, f) for f in nc_files],
        'issues': issues + naming_issues + coverage_issues
    }

def validate_level(path, level, context, rules):
    """Recursively validate a level in the folder hierarchy."""
    child_dict_name = rules.get('child_name', 'children')
    children_names, issues = get_children(path)
    
    if children_names is None:
        result = {'complete': False, 'issues': issues}
        if child_dict_name:
            result[child_dict_name] = {}
        return result
    
    folder_issues = check_folder_rules(level, children_names, rules)
    issues.extend(folder_issues)
    
    level_order = ['model', 'variant', 'scenario', 'variable', 'grid', 'time']
    current_idx = level_order.index(level)
    next_level = level_order[current_idx + 1] if current_idx < len(level_order) - 1 else None
    
    children = {}
    all_children_complete = True
    
    for child_name in children_names:
        child_path = os.path.join(path, child_name)
        new_context = context.copy()
        if next_level:
            new_context[next_level] = child_name
        
        if next_level == 'time':
            child_result = validate_time_level(child_path, new_context, child_name)
        else:
            next_rules = VALIDATION_RULES.get(next_level, {})
            child_result = validate_level(child_path, next_level, new_context, next_rules)
        
        children[child_name] = child_result
        if not child_result.get('complete', False):
            all_children_complete = False
    
    result = {
        'complete': len(folder_issues) == 0 and all_children_complete,
        'issues': issues
    }
    result[child_dict_name] = children
    return result

# ============================================================================
# MAIN API FUNCTIONS
# ============================================================================

def validate_model(model_name, raw_data_dir, data_source="cmip6"):
    """Validate a single climate model's complete structure."""
    base_path = os.path.join(raw_data_dir, data_source, model_name)
    if not os.path.exists(base_path):
        return {'complete': False, 'variants': {}, 'issues': [f"Model directory does not exist: {base_path}"]}
    return validate_level(base_path, 'model', {'model': model_name}, VALIDATION_RULES['model'])

def validate_all_models(data_dir, data_source="cmip6", verbose=True):
    """Validate all climate models in the data directory."""
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
        print(f"Validating {len(model_names)} climate models...")
        print("=" * 80)
    
    all_models = {}
    complete_count = 0
    
    for i, model_name in enumerate(model_names, 1):
        if verbose:
            print(f"[{i}/{len(model_names)}] Validating {model_name}...", end=" ")
        
        model_result = validate_model(model_name, raw_data_dir, data_source)
        all_models[model_name] = model_result
        
        if model_result['complete']:
            complete_count += 1
            if verbose:
                print("✓ COMPLETE")
        else:
            if verbose:
                print("✗ INCOMPLETE")
    
    if verbose:
        print("=" * 80)
        print(f"\nSummary: {complete_count}/{len(model_names)} models complete ({100*complete_count/len(model_names):.1f}%)")
    
    return all_models

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

# ============================================================================
# FILE CHECKING FUNCTIONS
# ============================================================================

def check_processed_files_exist(model, variant, scenario, variable, grid, time_period, 
                                file_path, processed_data_path, data_source="cmip6"):
    """Check if all expected yearly processed files exist."""
    dest_dir = os.path.join(processed_data_path, data_source, model, variant, scenario, variable, grid, time_period)
    if not os.path.exists(dest_dir):
        return False
    
    monthly = is_monthly(time_period)
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
        
        filtered_model = {'complete': model_data['complete'], 'issues': model_data.get('issues', []), 'variants': {}}
        
        for variant, vdata in model_data.get('variants', {}).items():
            filtered_variant = {'complete': vdata['complete'], 'issues': vdata.get('issues', []), 'scenarios': {}}
            variant_stats = {'total': 0, 'remaining': 0, 'scenarios': set(), 'variables': set()}
            
            for scenario, sdata in vdata.get('scenarios', {}).items():
                variant_stats['scenarios'].add(scenario)
                filtered_scenario = {'complete': sdata['complete'], 'issues': sdata.get('issues', []), 'variables': {}}
                scenario_has_tasks = False
                
                for variable, vardata in sdata.get('variables', {}).items():
                    variant_stats['variables'].add(variable)
                    filtered_var = {'complete': vardata['complete'], 'issues': vardata.get('issues', []), 'grids': {}}
                    var_has_tasks = False
                    
                    for grid, gdata in vardata.get('grids', {}).items():
                        filtered_grid = {'complete': gdata['complete'], 'issues': gdata.get('issues', []), 'time_periods': {}}
                        grid_has_tasks = False
                        
                        for time_period, tdata in gdata.get('time_periods', {}).items():
                            filtered_time = {'complete': tdata['complete'], 'issues': tdata.get('issues', []), 'files': []}
                            
                            for file_path in tdata.get('files', []):
                                stats['total'] += 1
                                variant_stats['total'] += 1
                                if not check_processed_files_exist(model, variant, scenario, variable, grid, 
                                                                  time_period, file_path, processed_data_path, data_source):
                                    filtered_time['files'].append(file_path)
                                    stats['remaining'] += 1
                                    variant_stats['remaining'] += 1
                            
                            if filtered_time['files']:
                                filtered_grid['time_periods'][time_period] = filtered_time
                                grid_has_tasks = True
                        
                        if grid_has_tasks:
                            filtered_var['grids'][grid] = filtered_grid
                            var_has_tasks = True
                    
                    if var_has_tasks:
                        filtered_scenario['variables'][variable] = filtered_var
                        scenario_has_tasks = True
                
                if scenario_has_tasks:
                    filtered_variant['scenarios'][scenario] = filtered_scenario
            
            if filtered_variant['scenarios']:
                filtered_model['variants'][variant] = filtered_variant
            elif variant_stats['total'] > 0 and variant_stats['remaining'] == 0:
                stats['variants_done'].append({
                    'model': model, 'variant': variant,
                    'scenarios': list(variant_stats['scenarios']),
                    'variables': list(variant_stats['variables'])
                })
        
        if filtered_model['variants']:
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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_complete_models(validation_results):
    """Extract list of complete model names."""
    return [name for name, data in validation_results.items() if data['complete']]

def get_incomplete_models(validation_results):
    """Extract list of incomplete model names."""
    return [name for name, data in validation_results.items() if not data['complete']]

def iterate_model_files(model_structure, filter_complete=True):
    """
    Iterator that yields all files from a model in organized order.
    
    Parameters:
    -----------
    model_structure : dict
        The nested validation structure for a model
    filter_complete : bool
        If True, only yield files from complete branches
    
    Yields:
    -------
    tuple : (variant, scenario, variable, grid, time_period, filepath, is_complete)
    """
    for variant_name, variant_data in sorted(model_structure.get('variants', {}).items()):
        if filter_complete and not variant_data['complete']:
            continue
            
        for scenario_name, scenario_data in sorted(variant_data.get('scenarios', {}).items()):
            if filter_complete and not scenario_data['complete']:
                continue
                
            for var_name, var_data in sorted(scenario_data.get('variables', {}).items()):
                if filter_complete and not var_data['complete']:
                    continue
                    
                for grid_name, grid_data in sorted(var_data.get('grids', {}).items()):
                    if filter_complete and not grid_data['complete']:
                        continue
                        
                    for time_name, time_data in sorted(grid_data.get('time_periods', {}).items()):
                        if filter_complete and not time_data['complete']:
                            continue
                            
                        for filepath in sorted(time_data.get('files', [])):
                            yield (variant_name, scenario_name, var_name, grid_name, 
                                   time_name, filepath, time_data['complete'])

def print_model_tree(model_structure, model_name, max_depth=None, show_complete=False):
    """Print a tree view of the model structure."""
    def print_node(data, name, depth=0):
        if max_depth is not None and depth >= max_depth:
            return
        if not show_complete and data.get('complete', False) and depth > 0:
            return
        
        status = "✓" if data.get('complete', False) else "✗"
        print(f"{'  ' * depth}{name} [{status}]")
        
        for issue in data.get('issues', []):
            print(f"{'  ' * depth}  ⚠ {issue}")
        
        if 'files' in data:
            print(f"{'  ' * depth}  📁 {len(data['files'])} files")
        
        for child_key in ['variants', 'scenarios', 'variables', 'grids', 'time_periods']:
            if child_key in data:
                for cname, cdata in sorted(data[child_key].items()):
                    print_node(cdata, cname, depth + 1)
                break
    
    print(f"\n{'='*80}\nModel: {model_name}")
    print(f"Status: {'✓ COMPLETE' if model_structure['complete'] else '✗ INCOMPLETE'}\n{'='*80}")
    for vname, vdata in sorted(model_structure.get('variants', {}).items()):
        print_node(vdata, f"Variant: {vname}", depth=0)