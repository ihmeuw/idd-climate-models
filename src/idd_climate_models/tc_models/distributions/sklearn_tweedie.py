"""
distributions/sklearn_tweedie.py
Tweedie GLM (power=1.5) via sklearn TweedieRegressor.
Fits rates (y/exposure) weighted by exposure.

Compatible stages: s2 (count), bulk, single
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import TweedieRegressor

from ..base import sk_const, add_interactions
from ..features import align_X


def fit(X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray,
        threshold=None, task=None,
        interaction: bool = False, island_interact: bool = False):
    X = add_interactions(X, interaction, island_interact)
    Xc = sk_const(X)
    model = TweedieRegressor(power=1.5, alpha=1e-6, max_iter=1000)
    model.fit(Xc, y / exposure, sample_weight=exposure)
    model._X_cols = Xc.columns.tolist()
    return model


def predict(model, X: pd.DataFrame, exposure: np.ndarray,
            threshold=None, task=None,
            interaction: bool = False, island_interact: bool = False) -> np.ndarray:
    X = add_interactions(X, interaction, island_interact)
    Xc = align_X(sk_const(X), model._X_cols)
    return model.predict(Xc) * exposure
