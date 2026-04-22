"""
spec_grid.py
Enumerate all model specs for the full TC comparison run.

Covariate sets (7, statsmodels models):
  none, wind, sdi, basin, island, wind_sdi, wind_sdi_basin_island

ML models (sklearn_rf, sklearn_xgb) always use: wind_sdi_basin_island

Distributions:
  Binary  (3): statsmodels_logistic, sklearn_rf, sklearn_xgb
  Count   (7): statsmodels_nb/gamma/lognormal/poisson/tweedie, sklearn_rf, sklearn_xgb
  Tail    (6): scipy_gpd, statsmodels_nb/gamma/lognormal, sklearn_rf, sklearn_xgb
  Single  (8): statsmodels_nb/gamma/poisson/tweedie/zip/zinb (no lognormal), sklearn_rf, sklearn_xgb

Thresholds (6): 70, 75, 80, 85, 90, 95

Total specs: ~4,560
"""

_ML_DISTS = {'sklearn_rf', 'sklearn_xgb'}

SM_COVARS = [
    'none', 'wind', 'sdi', 'basin', 'island',
    'wind_sdi', 'wind_sdi_basin_island',
]
ML_COVARS = ['wind_sdi_basin_island']

BINARY_DISTS = ['statsmodels_logistic', 'sklearn_rf', 'sklearn_xgb']
COUNT_DISTS  = [
    'statsmodels_nb', 'statsmodels_lognormal',
    'statsmodels_poisson', 'statsmodels_tweedie',
    'sklearn_rf', 'sklearn_xgb',
]
TAIL_DISTS   = [
    'scipy_gpd', 'statsmodels_nb', 'statsmodels_gamma',
    'sklearn_lognormal', 'statsmodels_lognormal',
]
SINGLE_DISTS = [
    'statsmodels_nb', 'statsmodels_poisson',
    'statsmodels_tweedie', 'statsmodels_zip', 'statsmodels_zinb',
    'sklearn_rf', 'sklearn_xgb',
]
THRESHOLDS = [70, 75, 80, 85, 90, 95]


def _allowed_covars(*dists: str):
    """If any dist is ML, only ML_COVARS; otherwise SM_COVARS."""
    if any(d in _ML_DISTS for d in dists):
        return ML_COVARS
    return SM_COVARS


def build_spec_grid():
    """
    Build the full combinatorial spec grid.
    Same covariate set is used for all stages within a model.
    ML models are constrained to wind_sdi_basin_island.

    Returns list of spec dicts.
    """
    specs = []

    # ── Single ────────────────────────────────────────────────────────────────
    # statsmodels_zinb uses StandardScaler and requires at least one covariate;
    # skip the intercept-only ('none') case which produces an empty feature matrix.
    _REQUIRES_COVARS = {'statsmodels_zinb'}
    for dist in SINGLE_DISTS:
        for covars in _allowed_covars(dist):
            if dist in _REQUIRES_COVARS and covars == 'none':
                continue
            specs.append({'structure': 'single', 'dist': dist, 'covars': covars})

    # ── Hurdle ────────────────────────────────────────────────────────────────
    for s1_dist in BINARY_DISTS:
        for s2_dist in COUNT_DISTS:
            for covars in _allowed_covars(s1_dist, s2_dist):
                specs.append({
                    'structure': 'hurdle',
                    's1_dist':   s1_dist,
                    's2_dist':   s2_dist,
                    'covars':    covars,
                })

    # ── POT ───────────────────────────────────────────────────────────────────
    for s1_dist in BINARY_DISTS:
        for bulk_dist in COUNT_DISTS:
            for tail_dist in TAIL_DISTS:
                for covars in _allowed_covars(s1_dist, bulk_dist, tail_dist):
                    for thresh in THRESHOLDS:
                        specs.append({
                            'structure':    'pot',
                            's1_dist':      s1_dist,
                            'bulk_dist':    bulk_dist,
                            'tail_dist':    tail_dist,
                            'covars':       covars,
                            'threshold_pct': thresh,
                        })

    # ── Double hurdle ─────────────────────────────────────────────────────────
    for s1_dist in BINARY_DISTS:
        for s2_dist in BINARY_DISTS:
            for bulk_dist in COUNT_DISTS:
                for tail_dist in TAIL_DISTS:
                    for covars in _allowed_covars(s1_dist, s2_dist, bulk_dist, tail_dist):
                        for thresh in THRESHOLDS:
                            specs.append({
                                'structure':    'double_hurdle',
                                's1_dist':      s1_dist,
                                's2_dist':      s2_dist,
                                'bulk_dist':    bulk_dist,
                                'tail_dist':    tail_dist,
                                'covars':       covars,
                                'threshold_pct': thresh,
                            })

    return specs
