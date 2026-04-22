"""
distributions/statsmodels_lognormal.py
Log-Normal regression via statsmodels WLS on log(y/exposure).
Provides proper standard errors, p-values, confidence intervals.

Fitting:   WLS on log(rate) with weights = y (Poisson-like)
Predict:   exp(Xβ + log(exposure) + σ²/2)  — Jensen correction applied
σ²:        unweighted residual variance (proper scale for back-transform)

Compatible stages: s2 (count), bulk, tail
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..base import sm_const, safe_log_exp


class _LogNormalResultSM:
    def __init__(self, wls_result, sigma2, X_cols):
        self.wls_result = wls_result
        self.sigma2     = sigma2
        self._X_cols    = X_cols
        self.params     = wls_result.params
        self.bse        = wls_result.bse
        self.pvalues    = wls_result.pvalues
        self.conf_int   = wls_result.conf_int
        self.rsquared   = wls_result.rsquared
        self.nobs       = wls_result.nobs

    def summary(self):
        return self.wls_result.summary()


def fit(X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray,
        threshold=None, task=None,
        interaction: bool = False, island_interact: bool = False) -> _LogNormalResultSM:
    if np.any(y <= 0):
        raise ValueError("lognormal requires y > 0")
    log_rate = np.log(y) - safe_log_exp(exposure)
    Xc = sm_const(X)
    wls_result = sm.WLS(log_rate, Xc, weights=y).fit()
    residuals = log_rate - wls_result.fittedvalues
    n, p = len(y), Xc.shape[1]
    sigma2 = np.sum(residuals ** 2) / (n - p)
    return _LogNormalResultSM(wls_result, sigma2, ['const'] + list(X.columns))


def predict(model: _LogNormalResultSM, X: pd.DataFrame, exposure: np.ndarray,
            threshold=None, task=None,
            interaction: bool = False, island_interact: bool = False) -> np.ndarray:
    Xc = sm_const(X)
    log_rate = model.wls_result.predict(Xc)
    return np.exp(log_rate + safe_log_exp(exposure) + model.sigma2 / 2)
