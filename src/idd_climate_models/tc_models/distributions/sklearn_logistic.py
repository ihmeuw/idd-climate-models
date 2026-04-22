"""
distributions/sklearn_logistic.py
Binary logistic regression via sklearn LogisticRegression.

Compatible stages: s1, s2 (binary)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..base import sk_const, add_interactions
from ..features import align_X


def fit(X: pd.DataFrame, y: np.ndarray,
        exposure=None, threshold=None, task=None,
        interaction: bool = False, island_interact: bool = False):
    X = add_interactions(X, interaction, island_interact)
    Xc = sk_const(X)
    model = LogisticRegression(max_iter=1000, solver='newton-cholesky', penalty=None)
    model.fit(Xc, y)
    model._X_cols = Xc.columns.tolist()
    return model


def predict(model, X: pd.DataFrame,
            exposure=None, task=None,
            interaction: bool = False, island_interact: bool = False) -> np.ndarray:
    X = add_interactions(X, interaction, island_interact)
    Xc = align_X(sk_const(X), model._X_cols)
    return model.predict_proba(Xc)[:, 1]
