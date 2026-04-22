"""
distributions/statsmodels_poisson.py
Poisson GLM with log link and log(exposure) offset via statsmodels.

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
    result = GLM(y, Xc, family=families.Poisson(),
                 offset=safe_log_exp(exposure)).fit(disp=False, maxiter=1000)
    result._X_cols = ['const'] + list(X.columns)
    return result


def predict(model, X: pd.DataFrame, exposure: np.ndarray,
            threshold=None, task=None,
            interaction: bool = False, island_interact: bool = False) -> np.ndarray:
    X = add_interactions(X, interaction, island_interact)
    Xc = sm_const(X)
    return model.predict(Xc, offset=safe_log_exp(exposure))
