"""
distributions/statsmodels_tweedie.py
Tweedie GLM (power=1.5) via statsmodels with log(exposure) offset.

Compatible stages: s2 (count), bulk, single
"""

import numpy as np
import pandas as pd

from ..base import sm_const, add_interactions, safe_log_exp


def fit(X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray,
        threshold=None, task=None,
        interaction: bool = False, island_interact: bool = False):
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod import families

    X = add_interactions(X, interaction, island_interact)
    Xc = sm_const(X)
    try:
        result = GLM(
            endog=y, exog=Xc,
            family=families.Tweedie(var_power=1.5, link=families.links.Log()),
            offset=safe_log_exp(exposure),
        ).fit(disp=False, maxiter=1000)
        result._X_cols = ['const'] + list(X.columns)
        return result
    except Exception:
        return None


def predict(model, X: pd.DataFrame, exposure: np.ndarray,
            threshold=None, task=None,
            interaction: bool = False, island_interact: bool = False) -> np.ndarray:
    X = add_interactions(X, interaction, island_interact)
    Xc = sm_const(X)
    return model.predict(Xc, offset=safe_log_exp(exposure))
