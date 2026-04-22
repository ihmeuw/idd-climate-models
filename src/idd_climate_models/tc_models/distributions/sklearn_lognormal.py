"""
distributions/sklearn_lognormal.py
Log-Normal regression via sklearn WLS on log(y/exposure).

Fitting:   WLS on log(rate) with weights = y (Poisson-like)
Predict:   exp(Xβ + log(exposure) + σ²/2)  — Jensen correction applied
σ²:        unweighted residual variance (proper scale for back-transform)

Compatible stages: s2 (count), bulk, tail
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ..base import safe_log_exp


class _LogNormalResult:
    def __init__(self, coef, intercept, sigma2, X_cols):
        self.coef_      = coef
        self.intercept_ = intercept
        self.sigma2     = sigma2
        self._X_cols    = X_cols


def fit(X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray,
        threshold=None, task=None,
        interaction: bool = False, island_interact: bool = False) -> _LogNormalResult:
    if np.any(y <= 0):
        raise ValueError("lognormal requires y > 0")
    log_rate = np.log(y) - safe_log_exp(exposure)
    X_arr = np.asarray(X, dtype=float)
    model = LinearRegression()
    model.fit(X_arr, log_rate, sample_weight=y)
    residuals = log_rate - model.predict(X_arr)
    n, p = len(y), X_arr.shape[1] + 1
    sigma2 = np.sum(residuals ** 2) / (n - p)
    return _LogNormalResult(model.coef_, model.intercept_, sigma2, list(X.columns))


def predict(model: _LogNormalResult, X: pd.DataFrame, exposure: np.ndarray,
            threshold=None, task=None,
            interaction: bool = False, island_interact: bool = False) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    log_rate = model.intercept_ + X_arr @ model.coef_
    return np.exp(log_rate + safe_log_exp(exposure) + model.sigma2 / 2)
