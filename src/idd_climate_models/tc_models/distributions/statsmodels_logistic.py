"""
distributions/statsmodels_logistic.py
Binary logistic regression via statsmodels GLM (Binomial family).

Compatible stages: s1, s2 (binary)
"""

import numpy as np
import pandas as pd

from ..base import sm_const, add_interactions


def fit(X: pd.DataFrame, y: np.ndarray,
        exposure=None, threshold=None, task=None,
        interaction: bool = False, island_interact: bool = False):
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod import families

    X = add_interactions(X, interaction, island_interact)
    Xc = sm_const(X)
    result = GLM(y, Xc, family=families.Binomial()).fit(disp=False)
    result._X_cols = ['const'] + list(X.columns)
    return result


def predict(model, X: pd.DataFrame,
            exposure=None, task=None,
            interaction: bool = False, island_interact: bool = False) -> np.ndarray:
    X = add_interactions(X, interaction, island_interact)
    Xc = sm_const(X)
    return model.predict(Xc)
