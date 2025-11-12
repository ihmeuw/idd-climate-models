import os
from datetime import datetime, timedelta

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def int_to_date(date_int, monthly=False):
    """Convert integer date representation to datetime object."""
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

def extract_folder_ranges(subfolders, scenario):
    issues = []
    date_ranges = []
    
    # Pattern to match YYYY-YYYY format
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
            
            # Simple check for logic: start must be before or equal to end
            if start_int > end_int:
                issues.append(f"Time period folder '{folder_name}' has start year after end year.")
                continue
                
            # Use YYYY as the date representation (YYYY00 to signal year-only range)
            date_ranges.append((start_int * 100, end_int * 100)) 
        except ValueError:
            issues.append(f"Time period folder '{folder_name}' contains non-integer years.")
            
    # Sort by start year
    return sorted(date_ranges, key=lambda x: x[0]), issues

def check_coverage(date_ranges, expected_range, gap_checker_func, is_year_only=False):
    if not date_ranges:
        return False, ["No valid date ranges found"]

    issues = []
    
    # Standard Boundary Check Setup
    expected_start, expected_end = expected_range
    actual_start, actual_end = date_ranges[0][0], date_ranges[-1][1]

    if is_year_only:
        # Tighter checks for time-period (no leniency)
        if actual_start > expected_start:
            issues.append(f"Coverage starts too late: {actual_start // 100}, expected by: {expected_start // 100}")
            return False, issues
        if actual_end < expected_end:
            issues.append(f"Coverage ends too early: {actual_end // 100}, expected until: {expected_end // 100}")
            return False, issues
    else:
        # Lenient checks for frequency (Original Logic)
        expected_start_year = int(str(expected_start)[:4])
        actual_start_year = int(str(actual_start)[:4])
        expected_end_year = int(str(expected_end)[:4])
        actual_end_year = int(str(actual_end)[:4])

        complete = True
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

    for i in range(1, len(date_ranges)):
        prev_end, curr_start = date_ranges[i-1][1], date_ranges[i][0]
        gap_issue = gap_checker_func(prev_end, curr_start)
        if gap_issue:
            issues.append(gap_issue)
            return False, issues

    return True, issues

def check_gap_monthly(prev_end_int, curr_start_int):
    """Checks for a gap between two monthly date integers."""
    prev_end = int_to_date(prev_end_int, monthly=True)
    curr_start = int_to_date(curr_start_int, monthly=True)
    
    # Calculate expected next month
    if prev_end.month == 12:
        next_expected = prev_end.replace(year=prev_end.year + 1, month=1)
    else:
        next_expected = prev_end.replace(month=prev_end.month + 1)
        
    if next_expected != curr_start:
        return f"Gap between {prev_end.strftime('%Y%m')} and {curr_start.strftime('%Y%m')}"
    return None

def check_gap_daily(prev_end_int, curr_start_int):
    """Checks for a gap between two daily date integers."""
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