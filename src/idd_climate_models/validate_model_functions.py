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
        'required_children': None,  # Any variants are ok
        'exact_count': None,  # Multiple variants allowed
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
        'required_children': None,  # Any grid name is ok
        'exact_count': 1,  # Must have exactly 1 grid
    },
    'grid': {
        'child_name': 'time_periods',
        'required_children': None,  # Any valid time folder
        'exact_count': 1,  # Must have exactly 1 time folder
        'validator': lambda name: name == 'day' or 'mon' in name.lower()
    },
    'time': {
        'child_name': None,  # Leaf node, no children
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
    if monthly:
        return datetime.strptime(str(date_int), "%Y%m")
    else:
        return datetime.strptime(str(date_int), "%Y%m%d")


def is_monthly(time_folder):
    """Check if time folder represents monthly data."""
    return 'mon' in time_folder.lower()


def get_children(path):
    """Get all subdirectories in a path."""
    try:
        children = sorted([f.name for f in os.scandir(path) if f.is_dir()])
        return children, []
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
    
    # Check required children
    if rules.get('required_children'):
        missing = [child for child in rules['required_children'] if child not in children]
        if missing:
            issues.append(f"Missing required folders: {missing}")
    
    # Check exact count
    if rules.get('exact_count') is not None:
        if len(children) != rules['exact_count']:
            issues.append(f"Expected {rules['exact_count']} folder(s), found {len(children)}: {children}")
    
    # Check validator function
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
    
    # Build regex - accept any valid date format in filename
    date_pattern = r"(\d{6})-(\d{6})" if monthly else r"(\d{8})-(\d{8})"
    regex = rf"{var}_{time_folder}_{model_name}_{scenario}_{variant}_{grid}_{date_pattern}\.nc"
    
    for fname in nc_files:
        m = re.match(regex, fname)
        if not m:
            issues.append(f"File '{fname}' doesn't match naming convention")
            continue
        start, end = m.groups()
        
        # Validate date format
        try:
            int_to_date(int(start), monthly)
            int_to_date(int(end), monthly)
        except ValueError as e:
            issues.append(f"File '{fname}' has invalid date format: {e}")
            continue
            
        date_ranges.append((int(start), int(end)))
    
    return sorted(date_ranges, key=lambda x: x[0]), monthly, issues


def check_date_coverage(date_ranges, scenario, monthly, time_rules):
    """Check if date ranges cover required period with no gaps."""
    issues = []
    
    if not date_ranges:
        return False, ["No valid date ranges found"]
    
    # Check coverage
    range_type = 'monthly' if monthly else 'daily'
    expected_start, expected_end = time_rules['date_ranges'][scenario][range_type]
    
    actual_start, actual_end = date_ranges[0][0], date_ranges[-1][1]
    
    # Only fail if we don't have enough coverage
    # It's OK to have MORE data (earlier start or later end)
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
            try:
                next_expected = prev_end.replace(year=year, month=month)
            except ValueError:
                next_expected = prev_end.replace(year=prev_end.year + 1, month=1)
            
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
    issues = []
    
    # Get files
    nc_files, file_issues = get_nc_files(path)
    issues.extend(file_issues)
    
    if not nc_files:
        return {
            'complete': False,
            'files': [],
            'issues': issues + ["No NetCDF files found"]
        }
    
    # Extract and validate date ranges
    date_ranges, monthly, naming_issues = extract_date_ranges(
        nc_files, context['model'], context['scenario'], context['variant'],
        context['grid'], context['variable'], time_folder
    )
    issues.extend(naming_issues)
    
    # Check coverage
    complete, coverage_issues = check_date_coverage(
        date_ranges, context['scenario'], monthly, VALIDATION_RULES['time']
    )
    issues.extend(coverage_issues)
    
    file_paths = [os.path.join(path, f) for f in nc_files]
    
    return {
        'complete': complete and len(naming_issues) == 0,
        'files': file_paths,
        'issues': issues
    }


def validate_level(path, level, context, rules):
    """
    Recursively validate a level in the folder hierarchy.
    
    Parameters:
    -----------
    path : str
        Current directory path
    level : str
        Current level name ('model', 'variant', 'scenario', 'variable', 'grid', 'time')
    context : dict
        Dictionary tracking current position in hierarchy
    rules : dict
        Validation rules for this level
        
    Returns:
    --------
    dict
        Nested structure with 'complete', child_name, and 'issues' keys
    """
    issues = []
    
    # Get the name for this level's children
    child_dict_name = rules.get('child_name', 'children')
    
    # Get children folders
    children_names, child_issues = get_children(path)
    if children_names is None:
        result = {'complete': False, 'issues': child_issues}
        if child_dict_name:
            result[child_dict_name] = {}
        return result
    issues.extend(child_issues)
    
    # Check folder-level rules
    folder_issues = check_folder_rules(level, children_names, rules)
    issues.extend(folder_issues)
    
    # Determine next level
    level_order = ['model', 'variant', 'scenario', 'variable', 'grid', 'time']
    current_idx = level_order.index(level)
    next_level = level_order[current_idx + 1] if current_idx < len(level_order) - 1 else None
    
    # Process children
    children = {}
    all_children_complete = True
    
    for child_name in children_names:
        child_path = os.path.join(path, child_name)
        
        # Update context - assign child_name to the NEXT level
        new_context = context.copy()
        if next_level:
            new_context[next_level] = child_name
        
        # Time level is special - it's the leaf with files
        if next_level == 'time':
            child_result = validate_time_level(child_path, new_context, child_name)
        else:
            next_rules = VALIDATION_RULES.get(next_level, {})
            child_result = validate_level(child_path, next_level, new_context, next_rules)
        
        children[child_name] = child_result
        
        if not child_result.get('complete', False):
            all_children_complete = False
    
    # Level is complete if no folder issues and all children are complete
    complete = len(folder_issues) == 0 and all_children_complete
    
    result = {
        'complete': complete,
        'issues': issues
    }
    result[child_dict_name] = children
    
    return result


# ============================================================================
# MAIN API FUNCTIONS
# ============================================================================

def validate_model(model_name, raw_data_dir, data_source="cmip6"):
    """
    Validate a single climate model's complete structure.
    """
    base_path = os.path.join(raw_data_dir, data_source, model_name)
    
    if not os.path.exists(base_path):
        return {
            'complete': False,
            'variants': {},
            'issues': [f"Model directory does not exist: {base_path}"]
        }
    
    context = {'model': model_name}
    return validate_level(base_path, 'model', context, VALIDATION_RULES['model'])


def validate_all_models(data_dir, data_source="cmip6", verbose=True):
    """
    Validate all climate models in the data directory.
    
    Parameters:
    -----------
    data_dir : str
        Base data directory path
    data_source : str, default "cmip6"
        Data source subdirectory name
    verbose : bool, default True
        If True, prints progress information
        
    Returns:
    --------
    dict
        Dictionary with model names as keys, each containing complete structure
        with completeness flags and issues at every level
    """
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
        print(f"\nSummary: {complete_count}/{len(model_names)} models complete "
              f"({100*complete_count/len(model_names):.1f}%)")
    
    return all_models


# ============================================================================
# PROCESSED FILE CHECKING FUNCTIONS
# ============================================================================

def check_processed_files_exist(model, variant, scenario, variable, grid, time_period, file_path, processed_data_path):
    """
    Check if all expected yearly processed files exist for a given input file.
    
    Parameters:
    -----------
    model, variant, scenario, variable, grid, time_period : str
        Model parameters defining the path structure
    file_path : str
        Path to the original input file
    processed_data_path : str
        Base path for processed data
        
    Returns:
    --------
    bool
        True if all expected yearly files exist, False otherwise
    """
    # Construct the destination directory
    dest_dir = os.path.join(processed_data_path, model, variant, scenario, variable, grid, time_period)
    
    # If the directory doesn't exist, files definitely don't exist
    if not os.path.exists(dest_dir):
        return False
    
    # Determine if this is monthly or daily data
    monthly = is_monthly(time_period)
    
    # Extract the date range from the filename
    base_name = os.path.basename(file_path)
    date_pattern = r"(\d{6})-(\d{6})" if monthly else r"(\d{8})-(\d{8})"
    match = re.search(date_pattern, base_name)
    
    if not match:
        # Can't determine date range, assume files don't exist
        return False
    
    start_date_int, end_date_int = int(match.group(1)), int(match.group(2))
    
    try:
        start_date = int_to_date(start_date_int, monthly)
        end_date = int_to_date(end_date_int, monthly)
    except ValueError:
        return False
    
    # Determine the year range, constrained by MIN_YEAR and MAX_YEAR
    MIN_YEAR = 1950
    MAX_YEAR = 2100
    
    start_year = max(start_date.year, MIN_YEAR)
    end_year = min(end_date.year, MAX_YEAR)
    
    # If there are no valid years in range, consider it "complete"
    if start_year > end_year:
        return True
    
    # Check if all expected yearly files exist
    for year in range(start_year, end_year + 1):
        if monthly:
            expected_fname = re.sub(r'_(\d{6})-(\d{6})\.nc$', f'_{year}01-{year}12.nc', base_name)
        else:  # daily
            expected_fname = re.sub(r'_(\d{8})-(\d{8})\.nc$', f'_{year}0101-{year}1231.nc', base_name)
        
        expected_path = os.path.join(dest_dir, expected_fname)
        
        if not os.path.exists(expected_path):
            return False
    
    return True


def filter_already_processed(validation_results, processed_data_path, rerun=False, rerun_scope='file', verbose=False):
    """
    Filter out tasks that have already been processed.
    
    Parameters:
    -----------
    validation_results : dict
        Results from validate_all_models()
    processed_data_path : str
        Base path where processed files are stored
    rerun : bool, default False
        If True, return all tasks. If False, filter out already processed tasks.
    rerun_scope : str, default 'file'
        'file': Rerun only missing files.
        'model': If any file is missing, rerun the entire model.
    verbose : bool, default False
        If True, print filtering progress
        
    Returns:
    --------
    dict
        Filtered validation results with the same structure, but only including
        tasks that need to be processed
    """
    if rerun:
        if verbose:
            print("Rerun=True: Returning all tasks without filtering")
        return validation_results
    
    if verbose:
        print(f"Filtering out already processed files (scope: {rerun_scope})...")
        print("=" * 80)
    
    # --- New Logic: First, identify models with any missing files ---
    models_to_process = set()
    total_files_checked = 0
    
    for model, model_data in validation_results.items():
        if not model_data.get('complete', False):
            continue
        
        for variant, scenario, variable, grid, time_period, file_path, _ in iterate_model_files(model_data):
            total_files_checked += 1
            if not check_processed_files_exist(model, variant, scenario, variable, grid, time_period, file_path, processed_data_path):
                models_to_process.add(model)
                break  # Found a missing file, no need to check the rest of this model
        if model in models_to_process:
            continue # Move to the next model

    if verbose:
        print(f"Identified {len(models_to_process)} models with at least one missing file.")

    # --- Second, build the final results based on the scope ---
    filtered_results = {}
    if rerun_scope == 'model':
        # Return the original, complete data for any model that needs processing
        for model_name in models_to_process:
            filtered_results[model_name] = validation_results[model_name]
        
        if verbose:
            print(f"Scope is 'model': Returning full data for {len(filtered_results)} models.")
        return filtered_results

    # --- Logic for 'file' scope (original behavior) ---
    for model in models_to_process:
        model_data = validation_results[model]
        filtered_model = {'complete': True, 'issues': [], 'variants': {}}
        
        for variant, scenario, variable, grid, time_period, file_path, _ in iterate_model_files(model_data):
            if not check_processed_files_exist(model, variant, scenario, variable, grid, time_period, file_path, processed_data_path):
                # Rebuild the nested structure only for the missing file
                filtered_model.setdefault('variants', {}).setdefault(variant, {'complete': True, 'issues': [], 'scenarios': {}})
                filtered_model['variants'][variant].setdefault('scenarios', {}).setdefault(scenario, {'complete': True, 'issues': [], 'variables': {}})
                filtered_model['variants'][variant]['scenarios'][scenario].setdefault('variables', {}).setdefault(variable, {'complete': True, 'issues': [], 'grids': {}})
                filtered_model['variants'][variant]['scenarios'][scenario]['variables'][variable].setdefault('grids', {}).setdefault(grid, {'complete': True, 'issues': [], 'time_periods': {}})
                filtered_model['variants'][variant]['scenarios'][scenario]['variables'][variable]['grids'][grid].setdefault('time_periods', {}).setdefault(time_period, {'complete': True, 'issues': [], 'files': []})
                filtered_model['variants'][variant]['scenarios'][scenario]['variables'][variable]['grids'][grid]['time_periods'][time_period]['files'].append(file_path)

        if filtered_model['variants']:
            filtered_results[model] = filtered_model

    if verbose:
        print("=" * 80)
        print(f"Filtering complete:")
        print(f"  Models with remaining tasks: {len(filtered_results)}")

    return filtered_results


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_complete_models(validation_results):
    """Extract list of complete model names from validation results."""
    return [name for name, data in validation_results.items() if data['complete']]


def get_incomplete_models(validation_results):
    """Extract list of incomplete model names from validation results."""
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


def get_completeness_summary(model_structure, level_names=None):
    """
    Get a summary of completeness at each level.
    
    Returns a dictionary with counts at each level.
    """
    if level_names is None:
        level_names = ['variants', 'scenarios', 'variables', 'grids', 'time_periods']
    
    def count_level(data, depth=0):
        if depth >= len(level_names):
            return {'total': 1, 'complete': 1 if data.get('complete', False) else 0}
        
        # Find the children key for this level
        children_key = level_names[depth]
        if children_key not in data:
            return {'total': 1, 'complete': 1 if data.get('complete', False) else 0}
        
        counts = {'total': 0, 'complete': 0}
        for child in data[children_key].values():
            child_counts = count_level(child, depth + 1)
            counts['total'] += child_counts['total']
            counts['complete'] += child_counts['complete']
        
        return counts
    
    return count_level(model_structure)


def print_model_tree(model_structure, model_name, max_depth=None, show_complete=False):
    """Print a tree view of the model structure with issues."""
    
    def print_node(data, name, level=0, depth=0):
        if max_depth is not None and depth >= max_depth:
            return
            
        if not show_complete and data.get('complete', False) and depth > 0:
            return
        
        indent = "  " * depth
        status = "✓" if data.get('complete', False) else "✗"
        print(f"{indent}{name} [{status}]")
        
        # Print issues
        if data.get('issues'):
            for issue in data['issues']:
                print(f"{indent}  ⚠ {issue}")
        
        # Print file count for leaf nodes
        if 'files' in data:
            print(f"{indent}  📁 {len(data['files'])} files")
        
        # Recurse into children - try each possible child key
        for child_key in ['variants', 'scenarios', 'variables', 'grids', 'time_periods']:
            if child_key in data:
                for child_name, child_data in sorted(data[child_key].items()):
                    print_node(child_data, child_name, level + 1, depth + 1)
                break
    
    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"Status: {'✓ COMPLETE' if model_structure['complete'] else '✗ INCOMPLETE'}")
    print(f"{'='*80}")
    
    for variant_name, variant_data in sorted(model_structure.get('variants', {}).items()):
        print_node(variant_data, f"Variant: {variant_name}", depth=0)