from pathlib import Path

MODEL_ROOT = "/mnt/team/rapidresponse/pub/tropical-storms"

REPO_ROOT = Path("/mnt/share/homes/bcreiner/repos")
SCRIPT_ROOT = Path(MODEL_ROOT) / "src" / "idd_climate_models"

DATA_PATH = Path(MODEL_ROOT) / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
TC_RISK_PATH = Path(MODEL_ROOT) / 'tc_risk'
TC_RISK_INPUT_PATH = TC_RISK_PATH / 'input'
TC_RISK_OUTPUT_PATH = TC_RISK_PATH / 'output'

TC_RISK_REPO_ROOT_DIR = REPO_ROOT / "tropical_cyclone_risk"

repo_name = "idd-climate-models"
package_name = "idd_climate_models"



# ============================================================================
# CONSTANTS
# ============================================================================
NUM_DRAWS = 100


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


tc_risk_file_type = 'netcdf'
tc_risk_n_procs = 4
tc_time_steps_per_day = 4
tc_risk_total_track_time_days = 30
tc_risk_tracks_per_basin = {'EP': 17,
                'NA': 15,
                'NI': 6,
                'SI': 11,
                'AU': 11,
                'SP': 7,
                'WP': 26,
                'GL': 1}

threshold_dict = {
    'any': {
        'wind_speed': 0,
        'name': 'Any wind speed (≥0 m/s)'
    },
    'tropical_storm': {
        'wind_speed': 17,
        'name': 'Tropical Storm or worse (≥17 m/s)'
    },
    'hurricane_minor': {
        'wind_speed': 33,
        'name': 'Hurricane or worse (≥33 m/s)'
    },
    'hurricane_major': {
        'wind_speed': 58,
        'name': 'Major Hurricane (≥58 m/s)'
    }
}

basin_dict = {
    'EP': {
        'name': 'Eastern Pacific',
        'most_detailed': True
    },
    'NA': {
        'name': 'North Atlantic',
        'most_detailed': True
    },
    'NI': {
        'name': 'North Indian',
        'most_detailed': True
    },
    'SI': {
        'name': 'South Indian',
        'most_detailed': True
    },
    'SP': {
        'name': 'South Pacific',
        'most_detailed': True
    },
    'WP': {
        'name': 'Western Pacific',
        'most_detailed': True
    },
    'GL': {
        'name': 'Global',
        'most_detailed': False
    }
}

ssp_scenario_map = {
    "ssp126": {
        "name": "RCP2.6",
        "rcp_scenario": 2.6,
        "color": "#046C9A",
        "dhs_scenario": 66,
        "dhs_vbd_scenario": 75
    },
    "ssp245": {
        "name": "RCP4.5",
        "rcp_scenario": 4.5,
        "color": "#E58601",
        "dhs_scenario": 0,
        "dhs_vbd_scenario": 0
    },
    "ssp585": {
        "name": "RCP8.5",
        "rcp_scenario": 8.5,
        "color": "#A42820",
        "dhs_scenario": 54,
        "dhs_vbd_scenario": 76
    },
    "historical": { 
        "name": "Historical", 
        "color": "#000000"
    }
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
    'data': {
        'raw': ['model', 'variant', 'scenario', 'variable', 'grid', 'frequency'],
        'processed': ['model', 'variant', 'scenario', 'variable', 'grid', 'frequency']
    },
    'tc_risk': {
        'input': ['model', 'variant', 'scenario', 'time-period'],
        'output': ['model', 'variant', 'scenario', 'time-period', 'basin']
    }
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

def build_tc_risk_rules(io_data_type):
    base = {
        **BASE_RULES,
        'scenario': {'child_name': 'time-period', 'required_children': None, 'exact_count': None},
        'basin': {'child_name': None, 'required_children': None, 'exact_count': None, 'handler': 'basin_level_validator'}
    }
    if io_data_type == "output":
        time_period_rule = {
            'child_name': 'basin',
            'date_ranges': build_tc_risk_date_ranges(),
            'required_children': list(basin_dict.keys())
        }
    else:
        time_period_rule = {
            'child_name': None,
            'date_ranges': build_tc_risk_date_ranges(),
            'required_children': None,
            'handler': 'time_period_file_validator'
        }
    base['time-period'] = time_period_rule
    return base

# TC_RISK_RULES = {
#     **BASE_RULES,
#     'scenario': {'child_name': 'time-period', 'required_children': None, 'exact_count': None},
#     'time-period': {
#         'child_name': None,
#         'date_ranges': build_tc_risk_date_ranges(),
#         'exact_count': None,
#         'handler': 'time_period_file_validator'
#     },
#     'basin': {'child_name': None, 'required_children': list(basin_dict.keys()), 'exact_count': None}
# }

# VALIDATION_RULES = {
#     'data': DATA_RULES,
#     'tc_risk': TC_RISK_RULES
# }

def select_priority_grid(available_grids):
    """
    Select the highest priority grid from available grids.
    Returns the grid name if found, None otherwise.
    """
    for priority_grid in GRID_PRIORITY_ORDER:
        if priority_grid in available_grids:
            return priority_grid
    return None

def get_time_bins(scenario_name, bin_size_years):
    TC_RISK_RULES = build_tc_risk_rules('output')
    date_ranges = TC_RISK_RULES['time-period']['date_ranges']
    if scenario_name not in date_ranges:
        print(f"Warning: No date range found for scenario '{scenario_name}'")
        return []
    start_year, end_year = date_ranges[scenario_name]
    return [(y, min(y + bin_size_years - 1, end_year)) for y in range(start_year, end_year + 1, bin_size_years)]