"""
distributions/scipy_gpd.py
Generalised Pareto Distribution tail model via scipy.optimize.

Model:  exceedances z_i = y_i - threshold ~ GPD(scale_i, shape)
        log(scale_i) = X_i @ beta + log(exposure_i)
        E[Y | Y > u] = u + scale_i / (1 - shape)

Compatible stages: tail only.
threshold must be provided to fit().
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..base import sm_const, safe_log_exp


class GPDResult:
    def __init__(self, params: np.ndarray, X_cols: list, threshold: float):
        self.params    = params
        self._X_cols   = X_cols
        self.threshold = threshold
        self._n_beta   = len(X_cols)


def fit(X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray,
        threshold: float = None,
        task=None, interaction: bool = False, island_interact: bool = False) -> GPDResult:
    if threshold is None:
        raise ValueError("scipy_gpd requires threshold")
    z = y - threshold
    if np.any(z <= 0):
        raise ValueError("All y passed to GPD tail fit must be > threshold")

    log_exp = safe_log_exp(exposure)
    Xc = sm_const(X)
    n_beta = Xc.shape[1]

    def nll(params):
        beta = params[:n_beta]
        shape = params[n_beta]
        if shape == 0:
            return 1e10
        scale_i = np.exp(Xc @ beta + log_exp)
        z_ratio = 1.0 + shape * z / scale_i
        if np.any(z_ratio <= 0):
            return 1e10
        ll = -np.sum(np.log(scale_i)) - (1.0 / shape + 1.0) * np.sum(np.log(z_ratio))
        return -ll if np.isfinite(ll) else 1e10

    init_b0 = np.log(np.mean(z) / np.mean(exposure) + 1e-10)
    best_res, best_nll = None, np.inf
    for shape_init in (0.05, 0.1, 0.2, -0.1, 0.0001):
        init = np.concatenate([[init_b0], np.zeros(n_beta - 1), [shape_init]])
        res = minimize(nll, init, method='L-BFGS-B',
                       bounds=[(None, None)] * n_beta + [(-0.5, 0.8)],
                       options={'maxiter': 2000, 'ftol': 1e-10})
        if res.fun < best_nll:
            best_nll, best_res = res.fun, res

    return GPDResult(best_res.x, ['const'] + list(X.columns), threshold)


def predict(model: GPDResult, X: pd.DataFrame, exposure: np.ndarray,
            threshold=None, task=None,
            interaction: bool = False, island_interact: bool = False) -> np.ndarray:
    Xc = sm_const(X)
    log_exp = safe_log_exp(exposure)
    beta = model.params[:model._n_beta]
    shape = np.clip(model.params[model._n_beta], -0.49, 0.79)
    scale_i = np.exp(Xc @ beta + log_exp)
    return np.maximum(model.threshold + scale_i / (1.0 - shape), 0)
