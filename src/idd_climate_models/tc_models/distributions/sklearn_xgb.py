"""
distributions/sklearn_xgb.py
XGBoost — supports both binary (task='binary') and count/regression (task='count') stages.

Compatible stages: s1, s2 (binary or count), bulk, single
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from ..features import align_X


def fit(X: pd.DataFrame, y: np.ndarray,
        exposure=None, threshold=None, task: str = 'count',
        interaction: bool = False, island_interact: bool = False):
    if task == 'binary':
        model = xgb.XGBClassifier(
            n_estimators=100, random_state=42,
            eval_metric='logloss', verbosity=0,
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=100, random_state=42,
            objective='reg:squarederror', verbosity=0,
        )
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
