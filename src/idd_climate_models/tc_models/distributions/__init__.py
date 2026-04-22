"""
distributions/__init__.py
Registry of all distribution modules + stage compatibility rules.

Each distribution exposes:
    fit(X, y, exposure=None, threshold=None, task=None,
        interaction=False, island_interact=False)
    predict(model, X, exposure=None, task=None,
            interaction=False, island_interact=False)

Stage compatibility
-------------------
BINARY_MODELS  : can predict P(event) in 0/1 — valid for s1, s2-binary
COUNT_MODELS   : predict counts given exposure — valid for s2-count, bulk, single
TAIL_MODELS    : valid for tail stage (extreme-value or count on exceedances)
SINGLE_ONLY    : handle zeros internally; must not be used inside a hurdle structure
ML_MODELS      : handle both binary and count via task= kwarg
"""

from . import (
    statsmodels_logistic,
    sklearn_logistic,
    statsmodels_nb,
    statsmodels_gamma,
    sklearn_lognormal,
    statsmodels_lognormal,
    sklearn_poisson,
    statsmodels_poisson,
    sklearn_tweedie,
    statsmodels_tweedie,
    statsmodels_zip,
    statsmodels_zinb,
    scipy_gpd,
    sklearn_rf,
    sklearn_xgb,
)

DISTRIBUTIONS = {
    'statsmodels_logistic':  statsmodels_logistic,
    'sklearn_logistic':      sklearn_logistic,
    'statsmodels_nb':        statsmodels_nb,
    'statsmodels_gamma':     statsmodels_gamma,
    'sklearn_lognormal':     sklearn_lognormal,
    'statsmodels_lognormal': statsmodels_lognormal,
    'sklearn_poisson':       sklearn_poisson,
    'statsmodels_poisson':   statsmodels_poisson,
    'sklearn_tweedie':       sklearn_tweedie,
    'statsmodels_tweedie':   statsmodels_tweedie,
    'statsmodels_zip':       statsmodels_zip,
    'statsmodels_zinb':      statsmodels_zinb,
    'scipy_gpd':             scipy_gpd,
    'sklearn_rf':            sklearn_rf,
    'sklearn_xgb':           sklearn_xgb,
}

BINARY_MODELS = {
    'statsmodels_logistic',
    'sklearn_logistic',
    'sklearn_rf',
    'sklearn_xgb',
}

COUNT_MODELS = {
    'statsmodels_nb',
    'statsmodels_gamma',
    'sklearn_lognormal',
    'statsmodels_lognormal',
    'sklearn_poisson',
    'statsmodels_poisson',
    'sklearn_tweedie',
    'statsmodels_tweedie',
    'statsmodels_zip',
    'statsmodels_zinb',
    'sklearn_rf',
    'sklearn_xgb',
}

TAIL_MODELS = {
    'scipy_gpd',
    'statsmodels_nb',
    'statsmodels_gamma',
    'sklearn_lognormal',
    'statsmodels_lognormal',
}

SINGLE_ONLY = {
    'statsmodels_zip',
    'statsmodels_zinb',
}

ML_MODELS = {
    'sklearn_rf',
    'sklearn_xgb',
}


def validate_stage_compat(dist_name: str, stage: str):
    """
    Raise ValueError if dist_name cannot be used for the given stage.

    stage values
    ------------
    's1'       : first binary stage (any death?)
    's2_binary': second binary stage (e.g. DH high/low classifier)
    's2_count' : second count stage (standard hurdle count model)
    'bulk'     : bulk count model (POT / DH)
    'tail'     : tail model (POT / DH)
    'single'   : single-stage model (no hurdle wrapper)
    """
    if dist_name not in DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution '{dist_name}'. "
                         f"Valid: {sorted(DISTRIBUTIONS)}")

    if stage in ('s1', 's2_binary') and dist_name not in BINARY_MODELS:
        raise ValueError(
            f"'{dist_name}' cannot predict binary outcomes. "
            f"Binary-compatible models: {sorted(BINARY_MODELS)}"
        )

    if stage in ('s2_count', 'bulk', 'single') and dist_name not in COUNT_MODELS:
        raise ValueError(
            f"'{dist_name}' is not a count model. "
            f"Count-compatible models: {sorted(COUNT_MODELS)}"
        )

    if stage == 'tail' and dist_name not in TAIL_MODELS:
        raise ValueError(
            f"'{dist_name}' is not valid for the tail stage. "
            f"Tail-compatible models: {sorted(TAIL_MODELS)}"
        )

    if stage != 'single' and dist_name in SINGLE_ONLY:
        raise ValueError(
            f"'{dist_name}' is single-stage only and cannot be used "
            f"inside a multi-stage structure (stage='{stage}')"
        )
