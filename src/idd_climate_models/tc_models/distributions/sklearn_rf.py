"""
distributions/sklearn_rf.py
Random Forest via sklearn — supports both binary (task='binary') and
count/regression (task='count') stages.

Compatible stages: s1, s2 (binary or count), bulk, single
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..features import align_X


def fit(X: pd.DataFrame, y: np.ndarray,
        exposure=None, threshold=None, task: str = 'count',
        interaction: bool = False, island_interact: bool = False):
    if task == 'binary':
        model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model


def predict(model, X: pd.DataFrame,
            exposure=None, task: str = 'count',
            interaction: bool = False, island_interact: bool = False) -> np.ndarray:
    cols = (model.feature_names_in_.tolist()
            if hasattr(model, 'feature_names_in_') else X.columns.tolist())
    X = align_X(X, cols)
    if task == 'binary':
        return model.predict_proba(X)[:, 1]
    return model.predict(X)
