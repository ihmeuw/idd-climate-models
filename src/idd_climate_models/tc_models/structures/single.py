"""
structures/single.py
Single-stage model: E[Y | X] directly from one distribution.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, List

from ..distributions import DISTRIBUTIONS, validate_stage_compat


@dataclass
class SingleResult:
    model:   Any
    dist:    str
    X_cols:  List[str]


def fit(X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray,
        dist: str,
        interaction: bool = False, island_interact: bool = False) -> SingleResult:
    validate_stage_compat(dist, 'single')
    mod = DISTRIBUTIONS[dist]
    task = 'count'  # single-stage always predicts counts
    fitted = mod.fit(X, y, exposure=exposure, task=task,
                     interaction=interaction, island_interact=island_interact)
    X_cols = ['const'] + list(X.columns)
    return SingleResult(model=fitted, dist=dist, X_cols=X_cols)


def predict(result: SingleResult, X: pd.DataFrame, exposure: np.ndarray,
            interaction: bool = False, island_interact: bool = False) -> np.ndarray:
    mod = DISTRIBUTIONS[result.dist]
    task = 'count'
    preds = mod.predict(result.model, X, exposure=exposure, task=task,
                        interaction=interaction, island_interact=island_interact)
    return np.maximum(preds, 0)
