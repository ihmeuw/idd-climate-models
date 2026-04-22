"""
distributions/statsmodels_nb.py
Negative Binomial regression (statsmodels NB2) with log(exposure) offset.
Uses Poisson start params and multiple optimisers for convergence robustness.

Compatible stages: s2 (count), bulk, tail, single
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.discrete.discrete_model import NegativeBinomial
import statsmodels.api as sm

from ..base import sm_const, add_interactions, safe_log_exp


def _poisson_start_params(Xc, y, offset):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pois = sm.GLM(y, Xc, family=sm.families.Poisson(), offset=offset)
            res = pois.fit(disp=False, maxiter=100)
            return np.append(res.params, 1.0)
    except Exception:
        return None


def fit(X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray,
        threshold=None, task=None,
        interaction: bool = False, island_interact: bool = False):
    X = add_interactions(X, interaction, island_interact)
    Xc = sm_const(X)
    offset = safe_log_exp(exposure)
    start_params = _poisson_start_params(Xc, y, offset)
    model = NegativeBinomial(endog=y, exog=Xc, offset=offset)

    best_result, best_ll = None, -np.inf
    for method in ('lbfgs', 'bfgs', 'newton', 'nm', 'powell'):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                kw = dict(disp=False, maxiter=2000, method=method)
                if start_params is not None and method in ('lbfgs', 'bfgs', 'newton'):
                    kw['start_params'] = start_params
                result = model.fit(**kw)
                if result is not None and hasattr(result, 'llf') and np.isfinite(result.llf):
                    if result.mle_retvals.get('converged', False):
                        result._X_cols = ['const'] + list(X.columns)
                        return result
                    if result.llf > best_ll:
                        best_ll = result.llf
                        best_result = result
        except Exception:
            continue

    if best_result is not None:
        best_result._X_cols = ['const'] + list(X.columns)
        return best_result
    return None


def predict(model, X: pd.DataFrame, exposure: np.ndarray,
            threshold=None, task=None,
            interaction: bool = False, island_interact: bool = False) -> np.ndarray:
    X = add_interactions(X, interaction, island_interact)
    Xc = sm_const(X)
    return model.predict(Xc, offset=safe_log_exp(exposure))
