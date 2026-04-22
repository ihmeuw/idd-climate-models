"""
structures/pot.py
Peaks-Over-Threshold three-stage hurdle model.

  Stage 1 — P(Y > 0 | X)          : binary model
  Stage 2 — E[Y | 0 < Y ≤ u, X]  : bulk count model (all positives)
  Stage 3 — E[Y | Y > u, X]       : tail model (exceedances above threshold u)

  p_exceed = fraction of positive training storms above u (fixed scalar)

  E[Y | X] = P(Y > 0 | X) * [
      (1 - p_exceed) * E[Y | bulk, X]
    + p_exceed       * E[Y | tail, X]
  ]

threshold_pct is estimated on training positives — never touches test data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, List

from ..distributions import DISTRIBUTIONS, validate_stage_compat


@dataclass
class POTResult:
    s1:        Any
    bulk:      Any
    tail:      Any           # None if < 10 tail observations
    p_exceed:  float
    threshold: float
    s1_dist:   str
    bulk_dist: str
    tail_dist: str
    s1_X_cols:   List[str]
    bulk_X_cols: List[str]
    tail_X_cols: List[str]


def fit(X_s1: pd.DataFrame, y_all: np.ndarray,
        s1_dist: str,
        X_bulk: pd.DataFrame, y_pos: np.ndarray, exposure_pos: np.ndarray,
        bulk_dist: str,
        X_tail: pd.DataFrame,
        tail_dist: str,
        threshold_pct: int = 90,
        interaction: bool = False, island_interact: bool = False) -> POTResult:
    validate_stage_compat(s1_dist,   's1')
    validate_stage_compat(bulk_dist, 'bulk')
    validate_stage_compat(tail_dist, 'tail')

    # Threshold estimated on training positives only
    u = float(np.percentile(y_pos, threshold_pct))
    tail_mask = y_pos > u
    p_exceed = float(tail_mask.mean())

    s1_mod   = DISTRIBUTIONS[s1_dist]
    bulk_mod = DISTRIBUTIONS[bulk_dist]
    tail_mod = DISTRIBUTIONS[tail_dist]

    s1   = s1_mod.fit(X_s1, y_all, task='binary',
                      interaction=interaction, island_interact=island_interact)
    bulk = bulk_mod.fit(X_bulk, y_pos, exposure=exposure_pos, task='count',
                        interaction=interaction, island_interact=island_interact)

    tail = None
    if tail_mask.sum() >= 10:
        try:
            tail = tail_mod.fit(
                X_tail[tail_mask], y_pos[tail_mask],
                exposure=exposure_pos[tail_mask],
                threshold=u,
                interaction=interaction, island_interact=island_interact,
            )
        except Exception:
            tail = None

    return POTResult(
        s1=s1, bulk=bulk, tail=tail,
        p_exceed=p_exceed, threshold=u,
        s1_dist=s1_dist, bulk_dist=bulk_dist, tail_dist=tail_dist,
        s1_X_cols=['const'] + list(X_s1.columns),
        bulk_X_cols=['const'] + list(X_bulk.columns),
        tail_X_cols=['const'] + list(X_tail.columns) if tail is not None else [],
    )


def predict(result: POTResult,
            X_s1: pd.DataFrame,
            X_bulk: pd.DataFrame,
            X_tail: pd.DataFrame,
            exposure: np.ndarray,
            interaction: bool = False, island_interact: bool = False) -> np.ndarray:
    s1_mod   = DISTRIBUTIONS[result.s1_dist]
    bulk_mod = DISTRIBUTIONS[result.bulk_dist]
    tail_mod = DISTRIBUTIONS[result.tail_dist]

    p_pos  = s1_mod.predict(result.s1, X_s1, task='binary',
                            interaction=interaction, island_interact=island_interact)
    e_bulk = bulk_mod.predict(result.bulk, X_bulk, exposure=exposure, task='count',
                              interaction=interaction, island_interact=island_interact)

    if result.tail is not None:
        e_tail = tail_mod.predict(result.tail, X_tail, exposure=exposure,
                                  interaction=interaction, island_interact=island_interact)
    else:
        e_tail = e_bulk * 2.0  # fallback

    e_given_pos = (1 - result.p_exceed) * e_bulk + result.p_exceed * e_tail
    return np.maximum(p_pos * e_given_pos, 0)
