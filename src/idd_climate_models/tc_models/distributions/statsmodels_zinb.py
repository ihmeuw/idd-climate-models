"""
distributions/statsmodels_zinb.py
Zero-Inflated Negative Binomial via statsmodels.
Aggressive fallback chain: full ZINB → intercept-only inflation → NB → Poisson.

Compatible stages: single only (handles zeros internally).
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families

from ..base import sm_const, safe_log_exp


def fit(X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray,
        threshold=None, task=None,
        interaction: bool = False, island_interact: bool = False):
    X_arr = np.asarray(X, dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_arr)
    Xc = sm_const(pd.DataFrame(Xs, columns=X.columns))
    log_exp = safe_log_exp(exposure)
    methods = ['nm', 'powell', 'bfgs', 'lbfgs']

    def _ok(r):
        return (r is not None
                and np.all(np.isfinite(r.params))
                and np.max(np.abs(r.params)) < 50)

    for exog_infl in (Xc, None):
        for method in methods:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    r = ZeroInflatedNegativeBinomialP(
                        endog=y, exog=Xc, exog_infl=exog_infl,
                        offset=log_exp, inflation='logit',
                    ).fit(disp=False, maxiter=5000, method=method,
                          warn_convergence=False, cov_type='approx')
                if _ok(r):
                    r._is_full_zinb = (exog_infl is not None)
                    r._X_cols = ['const'] + list(X.columns)
                    r._scaler = scaler
                    return r
            except Exception:
                continue

    for method in methods:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                r = NegativeBinomial(endog=y, exog=Xc, offset=log_exp).fit(
                    disp=False, maxiter=5000, method=method)
            if _ok(r):
                r._is_full_zinb = False
                r._X_cols = ['const'] + list(X.columns)
                r._scaler = scaler
                return r
        except Exception:
            continue

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            r = GLM(endog=y, exog=Xc, family=families.Poisson(),
                    offset=log_exp).fit()
        r._is_full_zinb = False
        r._is_poisson_fallback = True
        r._X_cols = ['const'] + list(X.columns)
        r._scaler = scaler
        return r
    except Exception:
        return None


def predict(model, X: pd.DataFrame, exposure: np.ndarray,
            threshold=None, task=None,
            interaction: bool = False, island_interact: bool = False) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    if hasattr(model, '_scaler') and model._scaler is not None:
        X_arr = model._scaler.transform(X_arr)
    Xc = sm_const(pd.DataFrame(X_arr, columns=X.columns))
    log_exp = safe_log_exp(exposure)
    is_zinb = getattr(model, '_is_full_zinb', False)
    is_poisson = getattr(model, '_is_poisson_fallback', False)
    try:
        if is_poisson:
            lp = Xc @ model.params
            return np.clip(np.exp(np.clip(lp + log_exp, -50, 50)), 0, 1e8)
        if is_zinb:
            return np.clip(model.predict(exog=Xc, exog_infl=Xc, offset=log_exp), 0, 1e8)
        return np.clip(model.predict(Xc, offset=log_exp), 0, 1e8)
    except Exception:
        try:
            return np.clip(model.predict(Xc, offset=log_exp), 0, 1e8)
        except Exception:
            n = min(Xc.shape[1], len(model.params))
            mu = np.exp(np.clip(Xc @ model.params[:n] + log_exp, -50, 50))
            return np.clip(mu, 0, 1e8)
