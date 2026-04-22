"""
features.py
Covariate matrix construction and train/test alignment.

Covariate tokens (underscore-delimited):
    'none'      - intercept only
    'wind'      - max_wind_speed (raw)
    'logwind'   - log_max_wind_speed
    'sdi'       - sdi (raw)
    'logsdi'    - log_sdi
    'basin'     - basin (one-hot dummies)
    'island'    - is_island
    'year'      - data_year

Examples:
    'wind_sdi'                  - raw wind + raw sdi
    'logwind_logsdi'            - log-transformed both
    'logwind_sdi_basin_island'  - log wind, raw sdi, + dummies
    'none'                      - intercept only

Old-style names ('base', 'island_basin', etc.) are silently remapped.
"""

import numpy as np
import pandas as pd
from typing import List

# ---------------------------------------------------------------------------
# Backward-compat alias map (old names → new token format)
# REMOVED 2026-03-31: 'basin' and 'island' aliases were breaking the 2^4 
# factorial covariate design where 'basin' should mean basin-only, not 
# 'wind_sdi_basin'. Only keep 'base' alias for truly old code.
# ---------------------------------------------------------------------------
_COVAR_ALIASES = {
    'base':              'wind_sdi',  # Legacy only - new code should use 'wind_sdi'
}

# Token → (source_column, output_name)
# Output name is used for consistent column naming across log/raw variants
_TOKEN_MAP = {
    'wind':    ('max_wind_speed',     'wind_speed_var'),
    'logwind': ('log_max_wind_speed', 'wind_speed_var'),
    'sdi':     ('sdi',                'sdi_var'),
    'logsdi':  ('log_sdi',            'sdi_var'),
    'basin':   ('basin',              'basin'),       # expanded to dummies
    'island':  ('is_island',          'is_island'),
    'year':    ('data_year',          'data_year'),
}


def build_X(
    df: pd.DataFrame,
    covars: str,
    include_log_exp: bool = False,
) -> pd.DataFrame:
    """
    Build feature matrix from a DataFrame.

    Parameters
    ----------
    df : DataFrame with raw columns
    covars : underscore-delimited covariate tokens (e.g. 'logwind_sdi_basin')
    include_log_exp : whether to append log_exp column (used by ML models)

    Returns
    -------
    DataFrame with named columns, ready for add_constant / model fitting.
    Basin is one-hot expanded; wind/sdi renamed to 'wind_speed_var'/'sdi_var'.
    """
    # Resolve alias
    covars = _COVAR_ALIASES.get(covars, covars)

    if covars == 'none':
        tokens = []
    else:
        tokens = covars.split('_')

    # Build column list and rename map
    cols = []
    rename = {}
    for token in tokens:
        if token not in _TOKEN_MAP:
            raise ValueError(f"Unknown covariate token '{token}'. "
                             f"Valid: {sorted(_TOKEN_MAP.keys())}")
        src_col, out_name = _TOKEN_MAP[token]
        cols.append(src_col)
        if src_col != out_name:
            rename[src_col] = out_name

    if include_log_exp:
        cols.append('log_exp')

    if not cols:
        return pd.DataFrame(index=df.index)

    X = df[cols].copy()

    # Rename to stable generic names
    if rename:
        X = X.rename(columns=rename)

    # Fill missing basin before dummying
    if 'basin' in cols:
        X['basin'] = X['basin'].fillna('UNKNOWN')

    # Check numeric columns for NaN
    num_cols = X.select_dtypes(include=[np.number]).columns
    if X[num_cols].isnull().any().any():
        bad = X[num_cols].isnull().sum()
        raise ValueError(f"Missing values in numeric columns: {bad[bad > 0].to_dict()}")

    # Expand basin to dummies
    if 'basin' in cols:
        dummies = pd.get_dummies(X['basin'], prefix='basin', drop_first=False, dtype=float)
        X = pd.concat([X.drop('basin', axis=1), dummies], axis=1)

    return X


def align_X(X_new: pd.DataFrame, reference_cols: List[str]) -> pd.DataFrame:
    """
    Align a feature matrix to a reference column list.
    Missing columns are filled with 0; extra columns are dropped.
    Used to make test-set features match training-set features after
    one-hot encoding (e.g. a basin present in train but not test).
    """
    X_new = X_new.copy()
    for col in reference_cols:
        if col not in X_new.columns:
            X_new[col] = 0.0
    return X_new[reference_cols]


# add_interactions lives in models/base.py — import from there, not here.