from pathlib import Path

MODEL_ROOT = "/mnt/team/rapidresponse/pub/tropical-storms"
REPO_ROOT = Path("/mnt/share/homes/bcreiner/repos")

DATA_PATH = Path(MODEL_ROOT) / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
TC_RISK_PATH = Path(MODEL_ROOT) / 'tc_risk'
TC_RISK_INPUT_PATH = TC_RISK_PATH / 'input'
TC_RISK_OUTPUT_PATH = TC_RISK_PATH / 'output'


repo_name = "idd-climate-models"
package_name = "idd_climate_models"

# ============================================================================
# CONSTANTS
# ============================================================================
SCENARIOS = ["historical", "ssp126", "ssp245", "ssp585"]
VARIABLES = {
    "cmip6": ["ua", "va", "tos", "psl", "hus", "ta"]
}
START_YEAR = {
    "historical": 1970,
    "ssp126": 2015,
    "ssp245": 2015,
    "ssp585": 2015
}
END_YEAR = {
    "historical": 2014,
    "ssp126": 2100,
    "ssp245": 2100,
    "ssp585": 2100
}

# ============================================================================
# VALIDATION RULES CONFIGURATION
# ============================================================================

# Dynamically build date_ranges for DATA_RULES
def build_data_date_ranges():
    ranges = {}
    for scenario in SCENARIOS:
        start = START_YEAR[scenario]
        end = END_YEAR[scenario]
        ranges[scenario] = {
            'monthly': (start * 100 + 1, end * 100 + 12),      # YYYYMM format
            'daily': (start * 10000 + 101, end * 10000 + 1231) # YYYYMMDD format
        }
    return ranges

# Dynamically build date_ranges for TC_RISK_RULES
def build_tc_risk_date_ranges():
    return {scenario: (START_YEAR[scenario], END_YEAR[scenario]) for scenario in SCENARIOS}

GRID_PRIORITY_ORDER = ['gn', 'gr', 'gr1', 'gr2']

FOLDER_STRUCTURE = {
    'data': ['model', 'variant', 'scenario', 'variable', 'grid', 'frequency'],
    'tc_risk': ['model', 'variant', 'scenario', 'time-period']
}

BASE_RULES = {
    'model': {'child_name': 'variant', 'required_children': None, 'exact_count': None},
    'variant': {'child_name': 'scenario', 'required_children': SCENARIOS, 'exact_count': None},
}

DATA_RULES = {
    **BASE_RULES,
    'scenario': {'child_name': 'variable', 'required_children': 'VARIABLES_BY_DATA_SOURCE', 'exact_count': None},
    'variable': {'child_name': 'grid', 'required_children': None, 'exact_count': 1, 'validator': lambda name: name in GRID_PRIORITY_ORDER},
    'grid': {'child_name': 'frequency', 'required_children': None, 'exact_count': 1, 'validator': lambda name: name == 'day' or 'mon' in name.lower()},
    'frequency': {
        'child_name': None,
        'date_ranges': build_data_date_ranges(),
        'handler':'frequency_file_validator'
    }
}

TC_RISK_RULES = {
    **BASE_RULES,
    'scenario': {'child_name': 'time-period', 'required_children': None, 'exact_count': None},
    'time-period': {
        'child_name': None,
        'date_ranges': build_tc_risk_date_ranges(),
        'exact_count': None,
        'handler': 'time_period_file_validator'
    }
}

VALIDATION_RULES = {
    'data': DATA_RULES,
    'tc_risk': TC_RISK_RULES
}

def select_priority_grid(available_grids):
    """
    Select the highest priority grid from available grids.
    Returns the grid name if found, None otherwise.
    """
    for priority_grid in GRID_PRIORITY_ORDER:
        if priority_grid in available_grids:
            return priority_grid
    return None