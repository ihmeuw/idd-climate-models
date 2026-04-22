"""
tc_models/__init__.py
Top-level package. Import from here for a clean API.

    from idd_climate_models.tc_models import (
        load_tc_data,
        run_model_evaluation,
        run_multiple_models,
        run_pot_threshold_sweep,
        DISTRIBUTIONS,
        STRUCTURES,
    )
"""

from .data import load_tc_data, prepare_tc_data
from .engine import (
    run_model_evaluation,
    run_multiple_models,
    run_pot_threshold_sweep,
)
from .metrics import calc_metrics, calc_tail_metrics
from .constants import (
    TC_DATA_PATH,
    OUTPUT_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    DEFAULT_SEEDS,
    DEFAULT_FOLDS,
)
from .distributions import DISTRIBUTIONS, validate_stage_compat
from .structures import STRUCTURES
from .cache import (
    spec_to_id,
    model_exists,
    save_model,
    load_model,
    log_to_manifest,
    is_logged,
    load_manifest,
)
from .coefficients import (
    load_model_coefficients,
    display_dh_model_coefficients,
    coefficients_to_dataframe,
)

__all__ = [
    # Data
    'load_tc_data',
    'prepare_tc_data',
    # Modeling
    'run_model_evaluation',
    'run_multiple_models',
    'run_pot_threshold_sweep',
    # Registries
    'DISTRIBUTIONS',
    'STRUCTURES',
    'validate_stage_compat',
    # Cache
    'spec_to_id',
    'model_exists',
    'save_model',
    'load_model',
    'log_to_manifest',
    'is_logged',
    'load_manifest',
    # Metrics
    'calc_metrics',
    'calc_tail_metrics',
    # Coefficients
    'load_model_coefficients',
    'display_dh_model_coefficients',
    'coefficients_to_dataframe',
    # Constants (paths + CV defaults)
    'TC_DATA_PATH',
    'OUTPUT_DIR',
    'MODELS_DIR',
    'RESULTS_DIR',
    'FIGURES_DIR',
    'DEFAULT_SEEDS',
    'DEFAULT_FOLDS',
]