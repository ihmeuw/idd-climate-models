"""
stage_grid.py

Enumerate all unique stage fits for the TC model comparison.

A stage is the atomic unit of fitting: one distribution on a well-defined
data subset, with a specific covariate set. All stages are independent of
each other — no stage fit depends on any other stage fit.

Stage types
-----------
  s1         : binary P(Y > 0)         — all rows
  pos_count  : count E[Y | Y>0]        — positive rows (hurdle s2 AND POT bulk)
  pos_binary : binary P(tail | Y>0)    — positive rows, target = (Y > u)  (DH s2)
  dh_bulk    : count E[Y | bulk]       — positive rows where Y <= u  (DH bulk)
  tail       : count/EVT E[Y | tail]   — positive rows where Y > u   (POT tail AND DH tail)
  single     : count model             — all rows

Each stage dict contains: stage_type, dist, covars, [threshold_pct]
"""

import json

from .distributions import (
    BINARY_MODELS, COUNT_MODELS, TAIL_MODELS, SINGLE_ONLY, ML_MODELS,
)

# Covariate sets available to every stage. Tokens are underscore-joined;
# see features.py for valid tokens.
# Note: wind/logwind and sdi/logsdi map to the same output column, so
# they are mutually exclusive within a covariate string.
ALL_COVARS = [
    'none',
    'wind',    'logwind',
    'sdi',     'logsdi',
    'wind_sdi',          'wind_logsdi',
    'logwind_sdi',       'logwind_logsdi',
    'wind_sdi_basin',    'logwind_logsdi_basin',
    'wind_sdi_island',   'logwind_logsdi_island',
    'wind_sdi_basin_island', 'logwind_logsdi_basin_island',
]

THRESHOLDS = [70, 75, 80, 85, 90, 95]

# Distributions that fail with an empty feature matrix (no covariates).
# sklearn_lognormal uses LinearRegression which requires >= 1 feature.
_REQUIRES_COVARS = {'statsmodels_zinb', 'sklearn_lognormal'} | ML_MODELS

# Distributions that require strictly positive y (y > 0).
# Cannot be used in the single stage which includes zero-death rows.
_REQUIRES_POSITIVE_Y = {'statsmodels_gamma', 'statsmodels_lognormal', 'sklearn_lognormal'}


def _covars_for(dist: str):
    """Return valid covariate sets for a distribution."""
    if dist in _REQUIRES_COVARS:
        return [c for c in ALL_COVARS if c != 'none']
    return ALL_COVARS


def build_stage_grid():
    """
    Return a list of unique stage dicts, each defining one independent model fit.
    Duplicates are impossible by construction but guarded with a seen-set.
    """
    stages = []
    seen: set = set()

    def add(d: dict):
        key = json.dumps(d, sort_keys=True)
        if key not in seen:
            seen.add(key)
            stages.append(d)

    # ── s1: binary, all rows ─────────────────────────────────────────────────
    for dist in sorted(BINARY_MODELS):
        for covars in _covars_for(dist):
            add({'stage_type': 's1', 'dist': dist, 'covars': covars})

    # ── pos_count: count on positives — hurdle s2 AND POT bulk ───────────────
    for dist in sorted(COUNT_MODELS - SINGLE_ONLY):
        for covars in _covars_for(dist):
            add({'stage_type': 'pos_count', 'dist': dist, 'covars': covars})

    # ── pos_binary: binary on positives, target = tail indicator (DH s2) ─────
    for dist in sorted(BINARY_MODELS):
        for covars in _covars_for(dist):
            for thresh in THRESHOLDS:
                add({'stage_type': 'pos_binary', 'dist': dist,
                     'covars': covars, 'threshold_pct': thresh})

    # ── dh_bulk: count on bulk positives (Y <= u) ────────────────────────────
    for dist in sorted(COUNT_MODELS - SINGLE_ONLY):
        for covars in _covars_for(dist):
            for thresh in THRESHOLDS:
                add({'stage_type': 'dh_bulk', 'dist': dist,
                     'covars': covars, 'threshold_pct': thresh})

    # ── tail: tail/EVT on exceedances (Y > u) — POT tail AND DH tail ─────────
    for dist in sorted(TAIL_MODELS):
        for covars in _covars_for(dist):
            for thresh in THRESHOLDS:
                add({'stage_type': 'tail', 'dist': dist,
                     'covars': covars, 'threshold_pct': thresh})

    # ── single: count model, all rows ────────────────────────────────────────
    # Exclude distributions requiring y > 0 — single stage includes zero-death rows.
    for dist in sorted(COUNT_MODELS - _REQUIRES_POSITIVE_Y):
        for covars in _covars_for(dist):
            add({'stage_type': 'single', 'dist': dist, 'covars': covars})

    return stages
