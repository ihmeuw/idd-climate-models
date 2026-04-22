"""
distributions/statsmodels_zip.py
Zero-Inflated Poisson via statsmodels.

Compatible stages: single only (handles zeros internally).
"""

import numpy as np
import pandas as pd
from statsmodels.discrete.count_model import ZeroInflatedPoisson

from ..base import sm_const, safe_log_exp


def fit(X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray,
        threshold=None, task=None,
        interaction: bool = False, island_interact: bool = False):
    Xc = sm_const(X)
    log_exp = safe_log_exp(exposure)
    for exog_infl in (Xc, None):
        try:
            model = ZeroInflatedPoisson(
                endog=y, exog=Xc, exog_infl=exog_infl,
                offset=log_exp, inflation='logit',
            )
            result = model.fit(disp=False, maxiter=2000, method='bfgs',
                               warn_convergence=False)
            if result.mle_retvals.get('converged', True):
                result._X_cols = ['const'] + list(X.columns)
                result._exposure_mean = exposure.mean()
                return result
        except Exception:
            continue
    return None


def predict(model, X: pd.DataFrame, exposure: np.ndarray,
            threshold=None, task=None,
            interaction: bool = False, island_interact: bool = False) -> np.ndarray:
    Xc = sm_const(X)
    log_exp = safe_log_exp(exposure)
    try:
        return model.predict(exog=Xc, exog_infl=Xc, offset=log_exp)
    except Exception:
        try:
            return model.predict(Xc, offset=log_exp)
        except Exception:
            n = X.shape[1] + 1
            lp = Xc @ model.params[:n]
            pi = 1 / (1 + np.exp(-(Xc @ model.params[n:])))
            return (1 - pi) * np.exp(lp + log_exp)
