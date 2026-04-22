"""
distributions/sklearn_poisson.py
Poisson regression via sklearn PoissonRegressor.
Fits rates (y/exposure) weighted by exposure.

Compatible stages: s2 (count), bulk, single
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from ..base import sk_const, add_interactions
from ..features import align_X


def fit(X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray,
        threshold=None, task=None,
        interaction: bool = False, island_interact: bool = False):
    X = add_interactions(X, interaction, island_interact)
    Xc = sk_const(X)
    model = PoissonRegressor(max_iter=1000, alpha=0, solver='newton-cholesky')
    model.fit(Xc, y / exposure, sample_weight=exposure)
    model._X_cols = Xc.columns.tolist()
    return model


def predict(model, X: pd.DataFrame, exposure: np.ndarray,
            threshold=None, task=None,
            interaction: bool = False, island_interact: bool = False) -> np.ndarray:
    X = add_interactions(X, interaction, island_interact)
    Xc = align_X(sk_const(X), model._X_cols)
    return model.predict(Xc) * exposure
