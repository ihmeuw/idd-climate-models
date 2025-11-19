import os
import re
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
import xarray as xr
import copy

import idd_climate_models.constants as rfc
from idd_climate_models.constants import select_priority_grid, GRID_PRIORITY_ORDER 
from idd_climate_models.dictionary_utils import snip_validation_results

FOLDER_STRUCTURE = rfc.FOLDER_STRUCTURE
# VALIDATION_RULES = rfc.VALIDATION_RULES
MODEL_ROOT = rfc.MODEL_ROOT

build_tc_risk_rules = rfc.build_tc_risk_rules
DATA_RULES = rfc.DATA_RULES
# CRITICAL FIX: Only flag 'ncells' as forbidden, as 'vertices' is used by structured grids.
FORBIDDEN_UNSTRUCTURED_DIMS = {'ncells'}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_validation_dict(DATA_TYPE, IO_DATA_TYPE, DATA_SOURCE, validation_results = None, strict_grid_check=False):
    """Adds the strict_grid_check flag to the validation dictionary."""
    depth = -1 if IO_DATA_TYPE == 'output' and DATA_TYPE == 'tc_risk' else -2
    validation_dict = {
        'data_type': DATA_TYPE,
        'io_data_type': IO_DATA_TYPE,
        'data_source': DATA_SOURCE,
        'folder_structure': FOLDER_STRUCTURE[DATA_TYPE][IO_DATA_TYPE],
        'detail_level': FOLDER_STRUCTURE[DATA_TYPE][IO_DATA_TYPE][depth],
        'strict_grid_check': strict_grid_check # <-- FLAG ADDED
    }
    if validation_results is not None:
        validation_dict['validation_results'] = validation_results
    return validation_dict

def int_to_date(date_int, monthly=False):
    """Convert integer date representation to datetime object."""
    # Handles YYYYMM (monthly) or YYYYMMDD (daily)
    return datetime.strptime(str(date_int), "%Y%m" if monthly else "%Y%m%d")

def is_monthly(time_folder):
    """Check if time folder represents monthly data."""
    return 'mon' in time_folder.lower()

def get_subfolders(path):
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
    
def check_grid_structure(path, context, time_folder, rules):
    """
    Placeholder for the old list-based checker. The logic is now embedded in validate_frequency_level.
    """
    issues = []
    return {'complete': True, 'issues': issues}

# ============================================================================
# GAP CHECKER CALLBACKS (Specific Logic)
# ============================================================================

def check_gap_monthly(prev_end_int, curr_start_int):
    """Checks for a gap between two monthly date integers (YYYYMM)."""
    prev_end = int_to_date(prev_end_int, monthly=True)
    curr_start = int_to_date(curr_start_int, monthly=True)
    
    # Calculate expected next month
    if prev_end.month == 12:
        next_expected = prev_end.replace(year=prev_end.year + 1, month=1)
    else:
        next_expected = prev_end.replace(month=prev_end.month + 1)
        
    if next_expected.date() != curr_start.date():
        return f"Gap between {prev_end.strftime('%Y%m')} and {curr_start.strftime('%Y%m')}"
    return None

def check_gap_daily(prev_end_int, curr_start_int):
    """Checks for a gap between two daily date integers (YYYYMMDD)."""
    prev_end = int_to_date(prev_end_int, monthly=False)
    curr_start = int_to_date(curr_start_int, monthly=False)
    
    if prev_end + timedelta(days=1) != curr_start:
        return f"Gap between {prev_end.strftime('%Y%m%d')} and {curr_start.strftime('%Y%m%d')}"
    return None

def check_gap_yearly(prev_end_int, curr_start_int):
    """Checks for a gap between two year-only integers (in YYYY00 format)."""
    prev_end_year = prev_end_int // 100
    curr_start_year = curr_start_int // 100
    
    if prev_end_year + 1 != curr_start_year:
        return f"Gap between period ending {prev_end_year} and period starting {curr_start_year}"
    return None

# ============================================================================
# UNIVERSAL COVERAGE CHECKER (General Logic)
# ============================================================================

def check_coverage(date_ranges, expected_range, gap_checker_func, is_year_only=False):
    """
    Check if date ranges cover the required period with no gaps.
    """
    if not date_ranges:
        return False, ["No valid date ranges found"]

    issues = []
    expected_start, expected_end = expected_range
    actual_start, actual_end = date_ranges[0][0], date_ranges[-1][1]
    complete = True

    # --- BOUNDARY CHECKS ---
    if is_year_only:
        if actual_start > expected_start or actual_end < expected_end:
            actual_start_year = actual_start // 100
            actual_end_year = actual_end // 100
            expected_start_year = expected_start // 100
            expected_end_year = expected_end // 100
            
            if actual_start > expected_start:
                issues.append(f"Coverage starts too late: {actual_start_year}, expected by: {expected_start_year}")
                complete = False
            if actual_end < expected_end:
                issues.append(f"Coverage ends too early: {actual_end_year}, expected until: {expected_end_year}")
                complete = False
            
            if not complete: return False, issues
            
    else:
        expected_start_year = int(str(expected_start)[:4])
        actual_start_year = int(str(actual_start)[:4])
        expected_end_year = int(str(expected_end)[:4])
        actual_end_year = int(str(actual_end)[:4])

        if actual_start > expected_start:
            if expected_start_year == 2015 and actual_start_year <= 2020:
                issues.append(f"Coverage starts a little late: {actual_start}, expected by: {expected_start}")
            else:
                issues.append(f"Coverage starts too late: {actual_start}, expected by: {expected_start}")
                complete = False
        
        if complete and actual_end < expected_end:
            if expected_end_year == 2100 and actual_end_year >= 2095:
                issues.append(f"Coverage ends a little early: {actual_end}, expected until: {expected_end}")
            else:
                issues.append(f"Coverage ends too early: {actual_end}, expected until: {expected_end}")
                complete = False
        
        if not complete:
            return False, issues

    # --- GAP CHECK ---
    for i in range(1, len(date_ranges)):
        prev_end, curr_start = date_ranges[i-1][1], date_ranges[i][0]
        gap_issue = gap_checker_func(prev_end, curr_start)
        if gap_issue:
            issues.append(gap_issue)
            return False, issues

    return True, issues

def extract_date_ranges(nc_files, model_name, scenario, variant, grid, var, time_folder):
    """
    Extract date ranges AND filename from NetCDF filenames.
    """
    issues = []
    # NOTE: is_monthly() must be defined globally/locally for this function
    monthly = is_monthly(time_folder)
    # List stores tuples of ( (start_int, end_int), filename )
    date_filename_pairs = [] 
    
    # Monthly: YYYYMM-YYYYMM (6 digits). Daily: YYYYMMDD-YYYYMMDD (8 digits)
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
            
            # Return date tuple and the filename
            date_filename_pairs.append(((int(start), int(end)), fname)) 
            
        except ValueError as e:
            issues.append(f"File '{fname}' has invalid date format: {e}")

    # Return only the date/filename pairs and issues list (monthly is no longer returned)
    return sorted(date_filename_pairs, key=lambda x: x[0]), issues

def filter_files_by_time_bin(source_dir, variable, bin_start, bin_end, frequency, model, scenario, variant, grid): 
    """
    Find and filter all NetCDF files for a variable that overlap with the year bin.
    """
    files_in_range = []
    
    try:
        all_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.nc')])
    except OSError as e:
        print(f"Error reading directory {source_dir}: {e}")
        return []
    
    # Use extract_date_ranges to properly parse the filenames and dates
    # date_filename_pairs is a list of [ ((start_int, end_int), filename), ... ]
    date_filename_pairs, issues = extract_date_ranges(
        all_files, model, scenario, variant, grid, variable, frequency
    )
    
    if issues:
        print(f"  Warning: Issues parsing files: {issues}")
    
    # Filter files by year range
    for (start_int, end_int), filename in date_filename_pairs: 
        
        # Extract year from the date integer
        start_year = int(str(start_int)[:4])
        end_year = int(str(end_int)[:4])
        
        # Check if this file overlaps with our bin range
        if start_year <= bin_end and end_year >= bin_start:
            # Append the full path using the returned filename
            files_in_range.append(os.path.join(source_dir, filename))
    
    return files_in_range


def extract_folder_ranges(subfolders, scenario):
    """
    Extract year ranges from time-period folder names (e.g., '1970-1989').
    Returns date ranges in YYYY00-YYYY00 format for use with check_coverage.
    """
    issues = []
    date_ranges = []
    regex = r"(\d{4})-(\d{4})"
    
    for folder_name in subfolders:
        m = re.match(regex, folder_name)
        if not m:
            issues.append(f"Time period folder '{folder_name}' doesn't match YYYY-YYYY convention.")
            continue
            
        start_year, end_year = m.groups()
        
        try:
            start_int = int(start_year)
            end_int = int(end_year)
            
            if start_int > end_int:
                issues.append(f"Time period folder '{folder_name}' has start year after end year.")
                continue
                
            # Convert to YYYY00 for comparison consistency with check_coverage's date structure
            date_ranges.append((start_int * 100, end_int * 100)) 
        except ValueError:
            issues.append(f"Time period folder '{folder_name}' contains non-integer years.")
            
    return sorted(date_ranges, key=lambda x: x[0]), issues

# ============================================================================
# FINAL HANDLER FUNCTIONS
# ============================================================================

def validate_frequency_level(path, context, time_folder, rules):
    """
    Validate the 'frequency' level (file-level checks).
    This function conditionally runs the strict grid check if required by the workflow.
    """
    issues = []
    # Retrieve the flag from the context
    strict_grid_check = context.get('strict_grid_check', False)
    
    # 1. Get files for validation (must be done here for both checks)
    nc_files, name_issues = get_nc_files(path)
    if not nc_files:
        return {'complete': False, 'files': [], 'issues': issues + ["No NetCDF files found"]}

    # 2. CONDITIONAL GRID STRUCTURE CHECK (Only run if flag is TRUE)
    if strict_grid_check:
        try:
            # We already confirmed nc_files is not empty above.
            first_file = os.path.join(path, nc_files[0])
            # Check for forbidden dimensions
            with xr.open_dataset(first_file) as ds:
                # FIXING FUTURE WARNING: Use ds.sizes.keys() for stability and dimension names
                present_dims = set(ds.sizes.keys()) 
                unstructured_dims_found = present_dims.intersection(FORBIDDEN_UNSTRUCTURED_DIMS)

                if unstructured_dims_found:
                    issues.append(
                        f"Forbidden unstructured grid dimensions found: {list(unstructured_dims_found)}. "
                        "Model is incompatible with the target TC risk grid."
                    )
                    # Fail fast if grid is bad, before proceeding to date checks
                    return {'complete': False, 'files': [], 'issues': issues}
        except Exception as e:
            # If we fail to read the file during the grid check, that's an issue too
            issues.append(f"Error during strict grid check: {e}")
            return {'complete': False, 'files': [], 'issues': issues}

    # 3. STANDARD FILE VALIDATION LOGIC (The original file and date checking)
    
    # Extract Dates and Naming Issues (use the list of files obtained in step 1)
    monthly = is_monthly(time_folder)
    date_filename_pairs, naming_issues = extract_date_ranges( 
        nc_files, # <-- Use the list from Step 1
        context['model'], context['scenario'], context['variant'],
        context['grid'], context['variable'], time_folder
    )
    issues.extend(name_issues) # Use the correct variable name for naming issues
    
    # Get just the date ranges
    date_ranges = [date_tuple for date_tuple, filename in date_filename_pairs]
    
    # Determine Required Range and Gap Checker
    range_data_type = 'monthly' if monthly else 'daily'
    expected_range = rules['date_ranges'][context['scenario']][range_data_type]
    gap_checker = check_gap_monthly if monthly else check_gap_daily
    
    # Check Coverage using the unified function
    complete, coverage_issues = check_coverage(
        date_ranges, 
        expected_range, 
        gap_checker,
        is_year_only=False 
    )
    issues.extend(coverage_issues)

    # Process Files (for 'fill_required' metadata)
    files_with_metadata = []
    fill_start_flagged = False
    fill_end_flagged = False
    
    for i, (date_tuple, filename) in enumerate(date_filename_pairs):
        full_path = os.path.join(path, filename)
        is_flagged = False
        
        # Check if the file is the first or last valid file in the *valid set*
        if i == 0 and "starts a little late" in " ".join(coverage_issues) and not fill_start_flagged:
            is_flagged = True
            fill_start_flagged = True
        elif i == len(date_filename_pairs) - 1 and "ends a little early" in " ".join(coverage_issues) and not fill_end_flagged:
            is_flagged = True
            fill_end_flagged = True
        
        # NOTE: The fill_required flag is relevant for data/raw validation
        files_with_metadata.append({'path': full_path, 'fill_required': is_flagged})
    
    return {
        'complete': complete and len(name_issues) == 0,
        'files': files_with_metadata,
        'issues': issues
    }


def validate_time_period_level(path, context, time_folder_name, rules):
    """
    Validate the 'time-period' level for TC_RISK (folder must contain ALL required variable files).
    
    This version ensures all 6 required variables are present and omits the 
    'fill_required' flag, as requested.
    """
    issues = []
    
    # Get the expected variables for this data source (e.g., cmip6)
    data_source = context.get('data_source', 'cmip6')
    # Fetch the required variables from your constants file
    required_variables = rfc.VARIABLES.get(data_source, [])
    
    # 1. Get NetCDF files
    try:
        nc_files = sorted([f for f in os.listdir(path) if f.endswith('.nc')])
    except OSError as e:
        return {'complete': False, 'issues': [f"Cannot read directory: {e}"]}
        
    if not nc_files:
        return {'complete': False, 'issues': ["No NetCDF files found in final directory."]}

    # 2. Extract present variables from filenames and build metadata
    present_variables = set()
    files_with_metadata = []
    
    # Regex to capture the variable name at the start of the file: ^(var)_
    variable_pattern = r"^([a-zA-Z0-9]+)_" 
    
    for fname in nc_files:
        m = re.match(variable_pattern, fname)
        
        if m:
            var_name = m.group(1)
            present_variables.add(var_name)
            
            # OMITTING 'fill_required' as requested
            files_with_metadata.append({'path': os.path.join(path, fname)})
        else:
            issues.append(f"File '{fname}' does not match expected variable name convention (e.g., var_freq_...): variable not found.")

    # 3. Check for missing variables
    required_set = set(required_variables)
    missing_variables = required_set - present_variables
    
    if missing_variables:
        issues.append(f"Missing required variable files: {sorted(list(missing_variables))}")

    # 4. Determine Final Completion Status
    is_complete = not issues and not missing_variables
    
    return {
        'complete': is_complete,
        'files': files_with_metadata, 
        'issues': issues
    }

def validate_basin_level(path, context, basin_folder_name, rules):
    """
    Validate the 'basin' level for TC_RISK (folder must contain NUM_DRAWS .nc files).
    """
    issues = []
    
    # Get the expected variables for this data source (e.g., cmip6)
    data_source = context.get('data_source', 'cmip6')
    # Fetch the required variables from your constants file
    required_variables = rfc.VARIABLES.get(data_source, [])
    
    # 1. Get NetCDF files
    try:
        nc_files = sorted([f for f in os.listdir(path) if f.endswith('.nc')])
    except OSError as e:
        return {'complete': False, 'issues': [f"Cannot read directory: {e}"]}

    # 2. Count the number of draw files
    num_draws_found = len(nc_files)
    if basin_folder_name == 'GL':
        if num_draws_found != 0:
            issues.append(f"Expected 0 draw files, found {num_draws_found}.")
    else:
        if num_draws_found < rfc.NUM_DRAWS:
            issues.append(f"Expected at least {rfc.NUM_DRAWS} draw files, found {num_draws_found}.")

    
    

    # 4. Determine Final Completion Status
    is_complete = not issues
    
    return {
        'complete': is_complete,
        'files': nc_files, 
        'issues': issues
    }

VALIDATION_FUNCTION_MAP = {
    'frequency_file_validator': validate_frequency_level,
    'time_period_file_validator': validate_time_period_level,
    'grid_structure_checker': check_grid_structure,
    'basin_level_validator': validate_basin_level
}


# ============================================================================
# MODULAR VALIDATION FUNCTIONS (The 4 Steps + Dispatcher)
# ============================================================================

def check_folder_rules(level, children, rules, data_source):
    """
    STEP 1: Check if folder contents meet the rules for this level.
    """
    issues = []

    required_children_list = rules.get('required_children')
    
    # DYNAMIC VARIABLE LOOKUP LOGIC
    if level == 'scenario' and required_children_list == 'VARIABLES_BY_DATA_SOURCE':
        # Resolve the actual list using the data_source (e.g., 'cmip6')
        required_children_list = rfc.VARIABLES.get(data_source, [])

    if required_children_list:
        missing = [c for c in required_children_list if c not in children]
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

def _determine_next_level(level, data_type, io_data_type):
    """
    STEP 2: Dynamically determine the next level in the hierarchy.
    """
    level_order = FOLDER_STRUCTURE.get(data_type, {}).get(io_data_type, [])
    try:
        current_idx = level_order.index(level)
        return level_order[current_idx + 1] if current_idx < len(level_order) - 1 else None
    except ValueError:
        return None

def _filter_children_by_priority(level, children_names, rules):
    """
    STEP 3: Apply Grid Priority filtering if applicable. 
    """
    children_to_validate = children_names.copy()
    children_results_placeholders = {}
    priority_issues = []

    # Check if we are at the 'variable' level and the priority rule is active
    if level == 'variable' and rules.get('select_priority_grid') is True:
        
        # Find the highest priority grid available on disk.
        highest_priority_grid = select_priority_grid(children_names)
        
        if highest_priority_grid:
            children_to_validate = [highest_priority_grid]
            priority_issues.append(f"Grid Priority: Found multiple grids. Selecting highest priority: '{highest_priority_grid}'")
            
            ignored_grids = [c for c in children_names if c != highest_priority_grid]
            for ignored_grid in ignored_grids:
                children_results_placeholders[ignored_grid] = {
                    'complete': True,
                    'issues': [f"Grid Priority: Validation skipped; lower priority than selected grid '{highest_priority_grid}'"],
                    'children': {}
                }
        else:
            pass
            
    return children_to_validate, children_results_placeholders, priority_issues


def validate_level(path, level, context, data_type, io_data_type, data_source):
    """
    The main recursive dispatcher function.
    """
    TC_RISK_RULES = build_tc_risk_rules(io_data_type)
    VALIDATION_RULES = {
        'data': DATA_RULES,
        'tc_risk': TC_RISK_RULES
    }
    rules = VALIDATION_RULES[data_type][level]
    child_dict_name = rules.get('child_name', 'children')
    children_names, issues = get_subfolders(path)

    if children_names is None:
        result = {'complete': False, 'issues': issues}
        if child_dict_name:
            result[child_dict_name] = {}
        return result

    # --- Step 1: Validate Current Folder ---
    folder_issues = check_folder_rules(level, children_names, rules, data_source) 
    issues.extend(folder_issues)

    # --- Step 2: Determine Next Level ---
    next_level = _determine_next_level(level, data_type, io_data_type)
        
    # --- Step 3: Filter Children (Priority Grid Logic) ---
    children_to_validate, children, priority_issues = _filter_children_by_priority(
        level, 
        children_names, 
        rules
    )
    issues.extend(priority_issues)
    
    all_children_complete = True

    # --- Step 4: Iterate and Delegate (Process only the selected children) ---
    for child_name in children_to_validate:
        child_path = os.path.join(path, child_name)
        new_context = context.copy()
        
        child_result = {'complete': True, 'issues': []}
        
        if next_level:
            new_context[next_level] = child_name
            
            child_result = validate_next_level(
                child_path,
                next_level,
                new_context,
                data_type,
                io_data_type,
                child_name,
                data_source # <-- PASS data_source
            )

        children[child_name] = child_result
        if not child_result.get('complete', False):
            all_children_complete = False

    # --- Step 5: Aggregate Results ---
    result = {
        'complete': len(folder_issues) == 0 and all_children_complete,
        'issues': issues
    }
    result[child_dict_name] = children
    return result   


def validate_next_level(child_path, next_level, new_context, data_type, io_data_type, child_name, data_source):
    """
    Dispatches validation to the correct function (specialized handler or recursive call).
    """
    
    TC_RISK_RULES = build_tc_risk_rules(io_data_type)
    VALIDATION_RULES = {
        'data': DATA_RULES,
        'tc_risk': TC_RISK_RULES
    }
    next_rules = VALIDATION_RULES[data_type][next_level]
    handler_key = next_rules.get('handler')

    if handler_key:
        special_handler = VALIDATION_FUNCTION_MAP.get(handler_key)
        if special_handler is None:
            return {
                'complete': False, 
                'issues': [f"Error: Unknown handler key '{handler_key}' for level '{next_level}'"]
            }
        child_result = special_handler(child_path, new_context, child_name, next_rules)
    else:
        # Recursive call: Must pass data_source again
        child_result = validate_level(child_path, next_level, new_context, data_type, io_data_type, data_source)
        
    return child_result

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

def _convert_paths_to_strings(obj):
    """Recursively convert all pathlib.Path objects in a structure to strings."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _convert_paths_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_paths_to_strings(i) for i in obj]
    else:
        return obj
    
def log_validation_results(validation_dict, detail_level='variant'):
    """
    Writes the aggregated validation results to a single JSON log file.
    """
    source_path = Path(MODEL_ROOT) / validation_dict['data_type'] / validation_dict['io_data_type'] / validation_dict['data_source']
    complete_count = sum(1 for res in validation_dict['validation_results'].values() if res['complete'])
    total_count = len(validation_dict['validation_results'])

    snipped_results = snip_validation_results(validation_dict, detail_level=detail_level)
    cache_data = {
        'source_path': str(source_path), 
        'data_type': validation_dict['data_type'],
        'timestamp': datetime.now().isoformat(),
        "results": _convert_paths_to_strings(snipped_results) 
    }

    os.makedirs(os.path.dirname(validation_dict['log_path']), exist_ok=True)
    with open(validation_dict['log_path'], 'w') as f:
        json.dump(cache_data, f, indent=4)

    
    print(f"\nValidation complete for: {validation_dict['data_type']}, {validation_dict['io_data_type']}, {validation_dict['data_source']}")
    print(f"Summary: {complete_count}/{total_count} models complete. Parsed log (up to '{detail_level}') written to {validation_dict['log_path']}")

# ============================================================================
# FINAL WRAPPER FUNCTIONS
# ============================================================================

def validate_model_in_source(model_name, source_path, data_type, io_data_type, data_source, strict_grid_check=False):
    """
    Validates a single climate model's structure.
    This wrapper now accepts and passes the strict_grid_check flag.
    """
    model_path = os.path.join(source_path, model_name)
    if not os.path.exists(model_path):
        return {'complete': False, 'issues': [f"Model directory does not exist: {model_path}"]}
    
    initial_level = 'model'

    initial_context = {
        initial_level: model_name,
        # INJECT data_source into context to satisfy handlers that expect it there
        'data_source': data_source,
        'strict_grid_check': strict_grid_check # <-- CRITICAL INJECTION
    }
    
    return validate_level(
        path=model_path,
        level=initial_level,
        context=initial_context, # Passes context with the strict_grid_check flag
        data_type=data_type,
        io_data_type=io_data_type,
        data_source=data_source # Passes data_source as separate argument
    )

def validate_all_models_in_source(validation_dict, verbose=True, strict_grid_check=False): 
    """
    Wrapper to run validation across all models and log results.
    Retrieves and passes the 'strict_grid_check' flag.
    """
    
    validation_dict = copy.deepcopy(validation_dict)
    data_type = validation_dict['data_type']
    data_source = validation_dict['data_source']
    detail_level = validation_dict['detail_level']
    io_data_type = validation_dict['io_data_type']
    
    # We rely on the flag being passed here, but also use the dict version for safety printing.

    LOG_FILENAME = "validation_log.json" 

    source_path = os.path.join(MODEL_ROOT, data_type, io_data_type, data_source)
    log_path = os.path.join(source_path, LOG_FILENAME)
    validation_dict['log_path'] = log_path
    
    if verbose:
        print(f"[{data_type.upper()}] Starting full validation for: {os.path.basename(source_path)} (Strict Check: {strict_grid_check})...")

    try:
        EXCLUDED_FOLDERS = {'.git', '__pycache__'} 
        model_names = sorted([f.name for f in os.scandir(source_path) 
                               if f.is_dir() and f.name not in EXCLUDED_FOLDERS])
    except OSError as e:
        print(f"Error reading source directory {source_path}: {e}")
        return {}

    all_model_results = {}
    
    # 3. Validation Execution Loop
    for i, model_name in enumerate(model_names, 1):
        if verbose:
            print(f"[{i}/{len(model_names)}] Validating {model_name}")

        # CRITICAL CHANGE: Pass the strict_grid_check flag to the next function
        model_result = validate_model_in_source(
            model_name, 
            source_path, 
            data_type, 
            io_data_type,
            data_source,
            strict_grid_check # <-- PASSES THE FLAG
        ) 
        all_model_results[model_name] = model_result

    validation_dict['validation_results'] = all_model_results
    
    # 4. Write new log
    log_validation_results(validation_dict, detail_level=detail_level)

    return validation_dict