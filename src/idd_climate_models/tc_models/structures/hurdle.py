"""
structures/hurdle.py
Two-stage hurdle model.

  Stage 1 — P(Y > 0 | X)      : binary model
  Stage 2 — E[Y | Y > 0, X]   : count model fitted on positive observations only

  E[Y | X] = P(Y > 0 | X) * E[Y | Y > 0, X]
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, List

from ..distributions import DISTRIBUTIONS, validate_stage_compat


@dataclass
class HurdleResult:
    s1:       Any
    s2:       Any
    s1_dist:  str
    s2_dist:  str
    s1_X_cols: List[str]
    s2_X_cols: List[str]


def fit(X_s1: pd.DataFrame, y_all: np.ndarray,
        s1_dist: str,
        X_s2: pd.DataFrame, y_pos: np.ndarray, exposure_pos: np.ndarray,
        s2_dist: str,
        interaction: bool = False, island_interact: bool = False) -> HurdleResult:
    validate_stage_compat(s1_dist, 's1')
    validate_stage_compat(s2_dist, 's2_count')

    s1_mod = DISTRIBUTIONS[s1_dist]
    s2_mod = DISTRIBUTIONS[s2_dist]

    s1 = s1_mod.fit(X_s1, y_all, task='binary',
                    interaction=interaction, island_interact=island_interact)

    s2_task = 'count'
    s2 = s2_mod.fit(X_s2, y_pos, exposure=exposure_pos, task=s2_task,
                    interaction=interaction, island_interact=island_interact)

    return HurdleResult(
        s1=s1, s2=s2,
        s1_dist=s1_dist, s2_dist=s2_dist,
        s1_X_cols=['const'] + list(X_s1.columns),
        s2_X_cols=['const'] + list(X_s2.columns),
    )


def predict(result: HurdleResult,
            X_s1: pd.DataFrame,
            X_s2: pd.DataFrame,
            exposure: np.ndarray,
            interaction: bool = False, island_interact: bool = False) -> np.ndarray:
    s1_mod = DISTRIBUTIONS[result.s1_dist]
    s2_mod = DISTRIBUTIONS[result.s2_dist]

    p_pos = s1_mod.predict(result.s1, X_s1, task='binary',
                           interaction=interaction, island_interact=island_interact)
    e_pos = s2_mod.predict(result.s2, X_s2, exposure=exposure, task='count',
                           interaction=interaction, island_interact=island_interact)

    return np.maximum(p_pos * e_pos, 0)
