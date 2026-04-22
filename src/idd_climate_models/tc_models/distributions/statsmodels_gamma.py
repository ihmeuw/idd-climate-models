"""
distributions/statsmodels_gamma.py
Gamma GLM with log link and log(exposure) offset.
Var(Y) ∝ E[Y]² — constant coefficient of variation.
No Jensen correction needed — estimates E[Y] directly.

Compatible stages: s2 (count), bulk, tail
Not valid for single-stage (use tweedie or nb instead).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..base import sm_const, safe_log_exp


def fit(X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray,
        threshold=None, task=None,
        interaction: bool = False, island_interact: bool = False):
    if np.any(y <= 0):
        raise ValueError("gamma requires y > 0")
    Xc = sm_const(X)
    log_exp = safe_log_exp(exposure)
    model = sm.GLM(y, Xc,
                   family=sm.families.Gamma(link=sm.families.links.Log()),
                   offset=log_exp)
    try:
        result = model.fit(disp=False, maxiter=1000)
    except Exception:
        result = model.fit(disp=False, maxiter=1000, start_params=np.zeros(Xc.shape[1]))
    result._X_cols = ['const'] + list(X.columns)
    return result


def predict(model, X: pd.DataFrame, exposure: np.ndarray,
            threshold=None, task=None,
            interaction: bool = False, island_interact: bool = False) -> np.ndarray:
    Xc = sm_const(X)
    return model.predict(Xc, offset=safe_log_exp(exposure))
