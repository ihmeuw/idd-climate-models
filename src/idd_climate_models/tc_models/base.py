"""
tc_models/base.py
Shared utilities for distribution and structure modules.
"""

import numpy as np
import pandas as pd
from statsmodels.tools import add_constant as _sm_add_constant


def sm_const(X: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame → numpy and prepend statsmodels constant."""
    return _sm_add_constant(np.asarray(X, dtype=float), has_constant='add')


def sk_const(X: pd.DataFrame) -> pd.DataFrame:
    """Prepend sklearn-compatible constant column."""
    return _sm_add_constant(X)


def add_interactions(
    X: pd.DataFrame,
    interaction: bool = False,
    island_interact: bool = False,
) -> pd.DataFrame:
    """
    Return a copy of X with interaction columns appended.
    Expects columns 'wind_speed_var', 'sdi_var', 'is_island'.
    """
    X = X.copy()
    if interaction and {'wind_speed_var', 'sdi_var'} <= set(X.columns):
        X['wind_sdi_interact'] = X['wind_speed_var'] * X['sdi_var']
    if island_interact and 'is_island' in X.columns:
        if 'wind_speed_var' in X.columns:
            X['island_wind_interact'] = X['is_island'] * X['wind_speed_var']
        if 'sdi_var' in X.columns:
            X['island_sdi_interact'] = X['is_island'] * X['sdi_var']
    return X


def safe_log_exp(exposure: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(exposure, 1e-10))
