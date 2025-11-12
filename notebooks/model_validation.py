import os
import re
from datetime import datetime, timedelta

import idd_climate_models.constants as rfc
from idd_climate_models.utility_functions import get_subfolders, get_nc_files, int_to_date, is_monthly

DATA_VALIDATION_RULES = rfc.DATA_VALIDATION_RULES

# ============================================================================

def validate_grid(grid_path, grid_name, rules):
    """Validate a single grid directory."""
    return validate_level(grid_path, 'grid', {'grid': grid_name}, rules)

def validate_grids_for_variable(variable_path, variable_name, rules):
    """Validate grids for a variable, selecting only the highest priority grid."""
    from pathlib import Path
    if isinstance(variable_path, str):
        variable_path = Path(variable_path)

    grids = {}
    available_grid_names = [d.name for d in variable_path.iterdir() if d.is_dir()]

    if rules.get('select_priority_grid', False):
        priority_grid = rfc.select_priority_grid(available_grid_names)
        if priority_grid:
            grid_path = variable_path / priority_grid
            grid_validation = validate_grid(grid_path, priority_grid, rules)
            if grid_validation:
                grids[priority_grid] = grid_validation
        else:
            return None
    else:
        for grid_dir in variable_path.iterdir():
            if not grid_dir.is_dir():
                continue
            grid_validation = validate_grid(grid_dir, grid_dir.name, rules)
            if grid_validation:
                grids[grid_dir.name] = grid_validation

    return grids if grids else None

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
    range_data_type = 'monthly' if monthly else 'daily'
    expected_start, expected_end = time_rules['date_ranges'][scenario][range_data_type]
    actual_start, actual_end = date_ranges[0][0], date_ranges[-1][1]

    expected_start_year = int(str(expected_start)[:4])
    actual_start_year = int(str(actual_start)[:4])
    expected_end_year = int(str(expected_end)[:4])
    actual_end_year = int(str(actual_end)[:4])

    if actual_start > expected_start:
        if expected_start_year == 2015 and actual_start_year <= 2020:
            issues.append(f"Coverage starts a little late: {actual_start}, expected by: {expected_start}")
            return True, issues
        else:
            issues.append(f"Coverage starts too late: {actual_start}, expected by: {expected_start}")
            return False, issues
    if actual_end < expected_end:
        if expected_end_year == 2100 and actual_end_year >= 2095:
            issues.append(f"Coverage ends a little early: {actual_end}, expected until: {expected_end}")
            return True, issues
        else:
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
        date_ranges, context['scenario'], monthly, DATA_VALIDATION_RULES['time']
    )

    files_with_metadata = []
    fill_start_flagged = False
    fill_end_flagged = False
    sorted_files = sorted(nc_files, key=lambda f: re.search(r'(\d{6,8})-(\d{6,8})\.nc', f).group(1))

    for issue in coverage_issues:
        if "starts a little late" in issue and not fill_start_flagged:
            if sorted_files:
                files_with_metadata.append({'path': os.path.join(path, sorted_files[0]), 'fill_required': True})
                fill_start_flagged = True
        elif "ends a little early" in issue and not fill_end_flagged:
            if sorted_files:
                files_with_metadata.append({'path': os.path.join(path, sorted_files[-1]), 'fill_required': True})
                fill_end_flagged = True

    for f in sorted_files:
        full_path = os.path.join(path, f)
        if not any(d['path'] == full_path for d in files_with_metadata):
            files_with_metadata.append({'path': full_path, 'fill_required': False})

    return {
        'complete': complete and len(naming_issues) == 0,
        'files': files_with_metadata,
        'issues': issues + naming_issues + coverage_issues
    }

def validate_level(path, level, context, rules):
    """Recursively validate a level in the folder hierarchy."""
    child_dict_name = rules.get('child_name', 'children')
    children_names, issues = get_subfolders(path)

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
            next_rules = DATA_VALIDATION_RULES.get(next_level, {})
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

def validate_model(model_name, raw_data_dir, data_source="cmip6"):
    """Validate a single climate model's complete structure."""
    base_path = os.path.join(raw_data_dir, data_source, model_name)
    if not os.path.exists(base_path):
        return {'complete': False, 'variants': {}, 'issues': [f"Model directory does not exist: {base_path}"]}
    return validate_level(base_path, 'model', {'model': model_name}, DATA_VALIDATION_RULES['model'])

def validate_all_models(raw_data_dir, data_source="cmip6", verbose=True):
    """Validate all climate models in the data directory."""
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