"""
structures/double_hurdle.py
Four-stage double hurdle model with covariate-dependent tail classification.

  Stage 1 — P(Y > 0 | X)          : binary model (any deaths?)
  Stage 2 — P(high | Y > 0, X)    : binary model (tail event?) — covariate-dependent
  Stage 3 — E[Y | bulk, X]        : bulk count model
  Stage 4 — E[Y | tail, X]        : tail count/EVT model

  E[Y | X] = P(Y > 0 | X) * E[Y | Y > 0, X]

  E[Y | Y > 0, X] = P(~high | Y > 0, X) * E[Y | bulk, X]
                  + P(high  | Y > 0, X) * E[Y | tail, X]

Key difference from POT: P(high | Y > 0, X) is covariate-dependent (modelled),
not a fixed scalar.  A Cat-5 hitting Haiti gets a higher tail probability than
a tropical storm brushing Japan.

threshold_pct is estimated on training positives — never touches test data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, List

from ..distributions import DISTRIBUTIONS, validate_stage_compat


@dataclass
class DHResult:
    s1:        Any
    s2:        Any
    bulk:      Any           # None if < 5 bulk observations
    tail:      Any           # None if < 10 tail observations
    threshold: float
    s1_dist:   str
    s2_dist:   str
    bulk_dist: str
    tail_dist: str
    s1_X_cols:   List[str]
    s2_X_cols:   List[str]
    bulk_X_cols: List[str]
    tail_X_cols: List[str]


def fit(X_s1: pd.DataFrame, y_all: np.ndarray,
        s1_dist: str,
        X_s2: pd.DataFrame,
        s2_dist: str,
        X_bulk: pd.DataFrame, y_pos: np.ndarray, exposure_pos: np.ndarray,
        bulk_dist: str,
        X_tail: pd.DataFrame,
        tail_dist: str,
        threshold_pct: int = 90,
        interaction: bool = False, island_interact: bool = False) -> DHResult:
    validate_stage_compat(s1_dist,   's1')
    validate_stage_compat(s2_dist,   's2_binary')
    validate_stage_compat(bulk_dist, 'bulk')
    validate_stage_compat(tail_dist, 'tail')

    u = float(np.percentile(y_pos, threshold_pct))
    tail_mask = y_pos > u
    bulk_mask = ~tail_mask
    n_tail = int(tail_mask.sum())
    n_bulk = int(bulk_mask.sum())

    s1_mod   = DISTRIBUTIONS[s1_dist]
    s2_mod   = DISTRIBUTIONS[s2_dist]
    bulk_mod = DISTRIBUTIONS[bulk_dist]
    tail_mod = DISTRIBUTIONS[tail_dist]

    s1 = s1_mod.fit(X_s1, y_all, task='binary',
                    interaction=interaction, island_interact=island_interact)
    s2 = s2_mod.fit(X_s2, tail_mask.astype(int), task='binary',
                    interaction=interaction, island_interact=island_interact)

    bulk = None
    if n_bulk >= 5:
        bulk = bulk_mod.fit(
            X_bulk[bulk_mask], y_pos[bulk_mask],
            exposure=exposure_pos[bulk_mask], task='count',
            interaction=interaction, island_interact=island_interact,
        )

    tail = None
    if n_tail >= 10:
        try:
            tail = tail_mod.fit(
                X_tail[tail_mask], y_pos[tail_mask],
                exposure=exposure_pos[tail_mask],
                threshold=u,
                interaction=interaction, island_interact=island_interact,
            )
        except Exception:
            tail = None

    return DHResult(
        s1=s1, s2=s2, bulk=bulk, tail=tail,
        threshold=u,
        s1_dist=s1_dist, s2_dist=s2_dist,
        bulk_dist=bulk_dist, tail_dist=tail_dist,
        s1_X_cols=['const'] + list(X_s1.columns),
        s2_X_cols=['const'] + list(X_s2.columns),
        bulk_X_cols=['const'] + list(X_bulk.columns),
        tail_X_cols=['const'] + list(X_tail.columns) if tail is not None else [],
    )


def predict(result: DHResult,
            X_s1: pd.DataFrame,
            X_s2: pd.DataFrame,
            X_bulk: pd.DataFrame,
            X_tail: pd.DataFrame,
            exposure: np.ndarray,
            interaction: bool = False, island_interact: bool = False) -> np.ndarray:
    n = len(X_s1)
    s1_mod   = DISTRIBUTIONS[result.s1_dist]
    s2_mod   = DISTRIBUTIONS[result.s2_dist]
    bulk_mod = DISTRIBUTIONS[result.bulk_dist]
    tail_mod = DISTRIBUTIONS[result.tail_dist]

    p_pos  = s1_mod.predict(result.s1, X_s1, task='binary',
                            interaction=interaction, island_interact=island_interact)
    p_high = s2_mod.predict(result.s2, X_s2, task='binary',
                            interaction=interaction, island_interact=island_interact)
    p_bulk = 1.0 - p_high

    if result.bulk is not None:
        e_bulk = bulk_mod.predict(result.bulk, X_bulk, exposure=exposure, task='count',
                                  interaction=interaction, island_interact=island_interact)
    else:
        e_bulk = np.full(n, result.threshold / 2)

    if result.tail is not None:
        e_tail = tail_mod.predict(result.tail, X_tail, exposure=exposure,
                                  interaction=interaction, island_interact=island_interact)
    else:
        e_tail = e_bulk * 2.0

    e_given_pos = p_bulk * e_bulk + p_high * e_tail
    return np.maximum(p_pos * e_given_pos, 0)
