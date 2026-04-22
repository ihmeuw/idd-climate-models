"""
constants.py
Project-wide constants for TC mortality modeling.

All file paths and fixed analytical parameters live here.
To update a data vintage, move to a new server, or change a filter,
edit this file only — data.py and engine.py read from here.
"""

from pathlib import Path

# ── Root directories ──────────────────────────────────────────────────────────
_DATA_ROOT   = Path('/mnt/team/rapidresponse/pub/tropical-storms/data')
_IBTRACS_DIR = _DATA_ROOT / 'ibtracs_deaths'

# ── Input data ────────────────────────────────────────────────────────────────
TC_DATA_PATH = (
    _IBTRACS_DIR
    / 'combined_ibtracs_with_deaths_deduplicated_with_sdi_updated_island.csv'
)

# ── Output directories ────────────────────────────────────────────────────────
_DIRECT_RISK_ROOT = Path('/mnt/team/rapidresponse/pub/tropical-storms/direct_risk')

OUTPUT_DIR  = _DIRECT_RISK_ROOT
MODELS_DIR  = _DIRECT_RISK_ROOT / 'models'
RESULTS_DIR = _DIRECT_RISK_ROOT / 'results'
FIGURES_DIR = _DIRECT_RISK_ROOT / 'figures'

# Stage-level cache (new architecture)
# Can be overridden via environment variables for separate runs
import os
STAGE_MODELS_DIR  = Path(os.environ.get('STAGE_MODELS_DIR', _DIRECT_RISK_ROOT / 'stage_models'))
STAGE_RESULTS_DIR = Path(os.environ.get('STAGE_RESULTS_DIR', _DIRECT_RISK_ROOT / 'stage_results'))

# ── Columns to drop on load ───────────────────────────────────────────────────
# These are pre-processing artifacts, redundant transforms, and unused metadata
# that are present in the raw file but not needed by the modeling engine.
COLUMNS_TO_DROP = [
    # Redundant death transforms
    'ln_total_deaths',
    'deaths_rate',
    'ln_death_rate',
    'ln_death_rate_pop_weighted',
    'exists_deaths',
    'ln_exists_deaths',
    # Location hierarchy metadata (not used as covariates)
    'location_set_version_id',
    'location_set_id',
    'parent_id',
    'is_estimate',
    'most_detailed',
    'sort_order',
    'location_ascii_name',
    'location_name_short',
    'location_name_medium',
    'location_type_id',
    'location_type',
    'map_id',
    'ihme_loc_id',
    'local_id',
    # Sociodemographic / labeling extras
    'developed',
    'lancet_label',
    'who_label',
    'sdi_ln',
    # Storm metadata not used in models
    'wind_mph',
    'storm_category',
    # Date duplicates (renamed on load)
    'start_date_y',
    'end_date_y',
    # Audit columns
    'date_inserted',
    'last_updated',
    'last_updated_by',
    'last_updated_action',
    # Pre-computed log exposure (we recompute to ensure consistency)
    'log_exposed_population',
]

# ── Column renames applied on load ────────────────────────────────────────────
COLUMN_RENAMES = {
    'start_date_x': 'start_date',
    'end_date_x':   'end_date',
}

# ── Integer columns (filled with 0 before casting) ───────────────────────────
INTEGER_COLS = ['is_island', 'level', 'super_region_id', 'region_id']

# ── Row filters ───────────────────────────────────────────────────────────────
MIN_EXPOSED_POPULATION = 1        # drop rows with zero or missing exposure
MAX_DATA_YEAR          = 2023     # exclude 2024+ (incomplete data)
LOCATION_LEVEL         = 3        # most-detailed admin level only

# ── CV defaults (used by engine.py) ──────────────────────────────────────────
DEFAULT_SEEDS = (123, 456, 789, 101112, 131415)
DEFAULT_FOLDS = 5