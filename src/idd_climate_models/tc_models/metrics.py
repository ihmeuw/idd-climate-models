"""
metrics.py
Evaluation metrics for TC mortality models.

calc_metrics      — standard metrics used across all models
calc_tail_metrics — additional tail-specific metrics for POT evaluation
"""

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from typing import Dict, Optional


def calc_binary_metrics(
    observed:  np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, float]:
    """
    Binary classification metrics for s1 and pos_binary stages.

    Parameters
    ----------
    observed  : 0/1 array
    predicted : predicted probabilities in [0, 1]

    Returns
    -------
    dict with keys: brier_score, log_loss, roc_auc
    """
    predicted = np.clip(predicted, 1e-7, 1 - 1e-7)
    metrics = {
        'brier_score': float(brier_score_loss(observed, predicted)),
        'log_loss':    float(log_loss(observed, predicted)),
        'roc_auc':     float(roc_auc_score(observed, predicted))
                       if len(np.unique(observed)) > 1 else np.nan,
    }
    return metrics


def calc_metrics(
    observed:  np.ndarray,
    predicted: np.ndarray,
    exposed:   np.ndarray,
    baseline_rate: Optional[float] = None,
) -> Dict[str, float]:
    """
    Standard evaluation metrics.

    Parameters
    ----------
    observed  : death counts
    predicted : predicted death counts
    exposed   : exposed population
    baseline_rate : mean death rate per 100k for skill score denominator (optional)

    Returns
    -------
    dict with keys:
        mae_rate, rmse_rate, mae_log, cor_rate,
        mae_count, rmse_count, cor_count, zero_acc
        (+ skill_* keys if baseline_rate provided)
    """
    predicted = np.maximum(predicted, 0)

    obs_rate  = (observed  / exposed) * 1e5
    pred_rate = (predicted / exposed) * 1e5
    obs_log   = np.log(observed  + 1)
    pred_log  = np.log(predicted + 1)

    zero_acc = np.mean((observed == 0) == (predicted < 0.5))

    nz = observed > 0
    cor_rate  = (np.corrcoef(obs_rate[nz],  pred_rate[nz])[0, 1]
                 if nz.sum() > 2 else np.nan)
    cor_count = (np.corrcoef(observed[nz],  predicted[nz])[0, 1]
                 if nz.sum() > 2 else np.nan)

    mae_rate   = np.mean(np.abs(obs_rate  - pred_rate))
    rmse_rate  = np.sqrt(np.mean((obs_rate  - pred_rate) ** 2))
    mae_log    = np.mean(np.abs(obs_log   - pred_log))
    mae_count  = np.mean(np.abs(observed  - predicted))
    rmse_count = np.sqrt(np.mean((observed  - predicted) ** 2))

    metrics = dict(
        mae_rate=mae_rate, rmse_rate=rmse_rate, mae_log=mae_log,
        cor_rate=cor_rate, mae_count=mae_count, rmse_count=rmse_count,
        cor_count=cor_count, zero_acc=zero_acc,
    )

    if baseline_rate is not None:
        bp_rate  = np.full(len(observed), baseline_rate)
        bp_count = (baseline_rate / 1e5) * exposed

        bm_rate  = np.mean(np.abs(obs_rate - bp_rate))
        br_rate  = np.sqrt(np.mean((obs_rate - bp_rate) ** 2))
        bm_count = np.mean(np.abs(observed  - bp_count))
        br_count = np.sqrt(np.mean((observed  - bp_count) ** 2))

        metrics['skill_mae_rate']   = 1 - mae_rate   / bm_rate   if bm_rate   > 0 else np.nan
        metrics['skill_rmse_rate']  = 1 - rmse_rate  / br_rate   if br_rate   > 0 else np.nan
        metrics['skill_mae_count']  = 1 - mae_count  / bm_count  if bm_count  > 0 else np.nan
        metrics['skill_rmse_count'] = 1 - rmse_count / br_count  if br_count  > 0 else np.nan

    return metrics


def calc_tail_metrics(
    observed:  np.ndarray,
    predicted: np.ndarray,
    exposed:   np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Tail-specific metrics for POT model evaluation.

    Parameters
    ----------
    observed, predicted, exposed : full arrays (all storms)
    threshold : the death-count threshold u used to define the tail
                (storms with observed > threshold are 'tail' storms)

    Returns
    -------
    dict with keys:
        n_tail             number of tail storms in this split
        tail_mae_count     MAE on tail storms only
        tail_rmse_count    RMSE on tail storms only
        tail_cor_count     Pearson r on tail storms (counts)
        tail_mae_rate      MAE on tail storms (per 100k rate)
        tail_coverage_threshold   fraction of top-'threshold percentile' storms observed in top-'threshold percentile' predicted
        bulk_mae_count     MAE on bulk storms (observed <= threshold)
        bulk_mae_rate      MAE on bulk storms (rate)
    """
    predicted = np.maximum(predicted, 0)
    tail_mask = observed > threshold
    bulk_mask = ~tail_mask
    n_tail = int(tail_mask.sum())

    obs_rate  = (observed  / exposed) * 1e5
    pred_rate = (predicted / exposed) * 1e5

    result: Dict[str, float] = {'n_tail': n_tail}

    # Tail metrics
    if tail_mask.sum() > 1:
        result['tail_mae_count']  = np.mean(np.abs(observed[tail_mask]  - predicted[tail_mask]))
        result['tail_rmse_count'] = np.sqrt(np.mean((observed[tail_mask] - predicted[tail_mask]) ** 2))
        result['tail_mae_rate']   = np.mean(np.abs(obs_rate[tail_mask]  - pred_rate[tail_mask]))
        result['tail_cor_count']  = (
            np.corrcoef(observed[tail_mask], predicted[tail_mask])[0, 1]
            if tail_mask.sum() > 2 else np.nan
        )
    else:
        result.update(tail_mae_count=np.nan, tail_rmse_count=np.nan,
                      tail_mae_rate=np.nan,  tail_cor_count=np.nan)

    # Top-'threshold percentile' coverage: what fraction of the top-'threshold percentile' observed
    # are also in the top-'threshold percentile' predicted?
    top_obs  = set(np.argsort(observed)[-n_tail:])
    top_pred = set(np.argsort(predicted)[-n_tail:])
    result['tail_coverage_threshold'] = len(top_obs & top_pred) / n_tail

    # Bulk metrics
    if bulk_mask.sum() > 0:
        result['bulk_mae_count'] = np.mean(np.abs(observed[bulk_mask]  - predicted[bulk_mask]))
        result['bulk_mae_rate']  = np.mean(np.abs(obs_rate[bulk_mask]  - pred_rate[bulk_mask]))
    else:
        result.update(bulk_mae_count=np.nan, bulk_mae_rate=np.nan)

    return result
