"""
engine.py
Core evaluation engine.

Exported functions
------------------
  run_model_evaluation(data, model_spec, ...)
      Full in-sample + CV evaluation for one model spec.

  run_multiple_models(data, model_specs, ...)
      Convenience wrapper: runs a list of specs and returns a summary table.

  run_pot_threshold_sweep(data, covars, ...)
      Sweeps POT threshold percentiles 70–95 and returns a comparison table
      with standard + tail-specific metrics at each threshold.

model_spec fields
-----------------
  structure     : 'single' | 'hurdle' | 'pot' | 'double_hurdle'
  dist          : (single only) distribution name
  s1_dist       : (hurdle/pot/double_hurdle) stage 1 binary distribution
  s2_dist       : (hurdle/double_hurdle) stage 2 distribution
  bulk_dist     : (pot/double_hurdle) bulk count distribution
  tail_dist     : (pot/double_hurdle) tail distribution
  covars        : default covariate tokens (see features.py); fallback for stages
  interaction   : bool, add wind*sdi interaction term
  island_interact : bool, add island*wind and island*sdi interactions
  label         : optional string label (auto-generated if absent)
  threshold_pct : (pot/double_hurdle) percentile for tail threshold, default 90

Per-stage covariate overrides (all fall back to 'covars' if absent)
--------------------------------------------------------------------
  s1_covars     : stage 1 — binary any-death
  s2_covars     : stage 2 — binary high/low (double_hurdle) or count (hurdle)
  bulk_covars   : bulk count model (pot, double_hurdle)
  tail_covars   : tail model (pot, double_hurdle)
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.utils.parallel import Parallel, delayed

from .constants import DEFAULT_SEEDS, DEFAULT_FOLDS
from .features  import build_X
from .metrics   import calc_metrics, calc_tail_metrics
from .structures import STRUCTURES
from .distributions import DISTRIBUTIONS, validate_stage_compat, ML_MODELS
from .cache import spec_to_id, save_model, is_logged, log_to_manifest


# ============================================================================
# Internal: build feature matrices for a given spec
# ============================================================================

def _get_stage_covars(spec: dict, stage: str) -> str:
    """Resolve covariate set for a stage, falling back to spec['covars']."""
    return spec.get(f'{stage}_covars', spec.get('covars', 'wind_sdi'))


def _build_Xs(data: pd.DataFrame, spec: dict,
              stage: str, include_log_exp: bool = False) -> pd.DataFrame:
    """Build X for a given stage, using per-stage covariate overrides if present."""
    return build_X(
        data,
        covars=_get_stage_covars(spec, stage),
        include_log_exp=include_log_exp,
    )


def _is_ml_dist(dist_name: str) -> bool:
    """Check if distribution is an ML model requiring log_exp feature."""
    return dist_name in ML_MODELS


# ============================================================================
# Internal: fit + predict dispatching to structures/
# ============================================================================

def _fit(train: pd.DataFrame, spec: dict):
    """
    Fit a model by dispatching to the appropriate structure module.
    Returns a structure-specific result object (SingleResult, HurdleResult, etc.)
    """
    structure = spec['structure']
    struct_mod = STRUCTURES[structure]
    interact = spec.get('interaction', False)
    island = spec.get('island_interact', False)

    if structure == 'single':
        dist = spec['dist']
        X = _build_Xs(train, spec, 'single', include_log_exp=_is_ml_dist(dist))
        y = train['total_deaths'].values
        exposure = train['exposed_population'].values
        return struct_mod.fit(X, y, exposure, dist=dist,
                              interaction=interact, island_interact=island)

    elif structure == 'hurdle':
        s1_dist = spec['s1_dist']
        s2_dist = spec['s2_dist']
        train_nz = train[train['death_y_n'] == 1].copy()

        X_s1 = _build_Xs(train, spec, 's1', include_log_exp=False)
        y_all = train['death_y_n'].values
        X_s2 = _build_Xs(train_nz, spec, 's2', include_log_exp=_is_ml_dist(s2_dist))
        y_pos = train_nz['total_deaths'].values
        exposure_pos = train_nz['exposed_population'].values

        return struct_mod.fit(
            X_s1=X_s1, y_all=y_all, s1_dist=s1_dist,
            X_s2=X_s2, y_pos=y_pos, exposure_pos=exposure_pos, s2_dist=s2_dist,
            interaction=interact, island_interact=island,
        )

    elif structure == 'pot':
        s1_dist = spec['s1_dist']
        bulk_dist = spec['bulk_dist']
        tail_dist = spec['tail_dist']
        threshold_pct = spec.get('threshold_pct', 90)
        train_nz = train[train['death_y_n'] == 1].copy()

        X_s1 = _build_Xs(train, spec, 's1', include_log_exp=False)
        y_all = train['death_y_n'].values
        X_bulk = _build_Xs(train_nz, spec, 'bulk', include_log_exp=_is_ml_dist(bulk_dist))
        y_pos = train_nz['total_deaths'].values
        exposure_pos = train_nz['exposed_population'].values
        X_tail = _build_Xs(train_nz, spec, 'tail', include_log_exp=_is_ml_dist(tail_dist))

        return struct_mod.fit(
            X_s1=X_s1, y_all=y_all, s1_dist=s1_dist,
            X_bulk=X_bulk, y_pos=y_pos, exposure_pos=exposure_pos, bulk_dist=bulk_dist,
            X_tail=X_tail, tail_dist=tail_dist,
            threshold_pct=threshold_pct,
            interaction=interact, island_interact=island,
        )

    elif structure == 'double_hurdle':
        s1_dist = spec['s1_dist']
        s2_dist = spec['s2_dist']
        bulk_dist = spec['bulk_dist']
        tail_dist = spec['tail_dist']
        threshold_pct = spec.get('threshold_pct', 90)
        train_nz = train[train['death_y_n'] == 1].copy()

        X_s1 = _build_Xs(train, spec, 's1', include_log_exp=False)
        y_all = train['death_y_n'].values
        X_s2 = _build_Xs(train_nz, spec, 's2', include_log_exp=False)
        X_bulk = _build_Xs(train_nz, spec, 'bulk', include_log_exp=_is_ml_dist(bulk_dist))
        y_pos = train_nz['total_deaths'].values
        exposure_pos = train_nz['exposed_population'].values
        X_tail = _build_Xs(train_nz, spec, 'tail', include_log_exp=_is_ml_dist(tail_dist))

        return struct_mod.fit(
            X_s1=X_s1, y_all=y_all, s1_dist=s1_dist,
            X_s2=X_s2, s2_dist=s2_dist,
            X_bulk=X_bulk, y_pos=y_pos, exposure_pos=exposure_pos, bulk_dist=bulk_dist,
            X_tail=X_tail, tail_dist=tail_dist,
            threshold_pct=threshold_pct,
            interaction=interact, island_interact=island,
        )

    else:
        raise ValueError(f"Unknown structure '{structure}'")


def _predict(fitted, test: pd.DataFrame, spec: dict) -> np.ndarray:
    """
    Predict from a fitted model by dispatching to the appropriate structure module.
    """
    structure = spec['structure']
    struct_mod = STRUCTURES[structure]
    interact = spec.get('interaction', False)
    island = spec.get('island_interact', False)
    exposure = test['exposed_population'].values

    if structure == 'single':
        dist = spec['dist']
        X = _build_Xs(test, spec, 'single', include_log_exp=_is_ml_dist(dist))
        return struct_mod.predict(fitted, X, exposure,
                                  interaction=interact, island_interact=island)

    elif structure == 'hurdle':
        s2_dist = spec['s2_dist']
        X_s1 = _build_Xs(test, spec, 's1', include_log_exp=False)
        X_s2 = _build_Xs(test, spec, 's2', include_log_exp=_is_ml_dist(s2_dist))
        return struct_mod.predict(fitted, X_s1=X_s1, X_s2=X_s2, exposure=exposure,
                                  interaction=interact, island_interact=island)

    elif structure == 'pot':
        bulk_dist = spec['bulk_dist']
        tail_dist = spec['tail_dist']
        X_s1 = _build_Xs(test, spec, 's1', include_log_exp=False)
        X_bulk = _build_Xs(test, spec, 'bulk', include_log_exp=_is_ml_dist(bulk_dist))
        X_tail = _build_Xs(test, spec, 'tail', include_log_exp=_is_ml_dist(tail_dist))
        return struct_mod.predict(fitted, X_s1=X_s1, X_bulk=X_bulk, X_tail=X_tail,
                                  exposure=exposure,
                                  interaction=interact, island_interact=island)

    elif structure == 'double_hurdle':
        bulk_dist = spec['bulk_dist']
        tail_dist = spec['tail_dist']
        X_s1 = _build_Xs(test, spec, 's1', include_log_exp=False)
        X_s2 = _build_Xs(test, spec, 's2', include_log_exp=False)
        X_bulk = _build_Xs(test, spec, 'bulk', include_log_exp=_is_ml_dist(bulk_dist))
        X_tail = _build_Xs(test, spec, 'tail', include_log_exp=_is_ml_dist(tail_dist))
        return struct_mod.predict(fitted, X_s1=X_s1, X_s2=X_s2,
                                  X_bulk=X_bulk, X_tail=X_tail, exposure=exposure,
                                  interaction=interact, island_interact=island)

    else:
        raise ValueError(f"Unknown structure '{structure}'")


# ============================================================================
# Internal: validate spec
# ============================================================================

def _validate_spec(spec: dict):
    """Validate spec has required fields and compatible distributions."""
    structure = spec.get('structure', '')

    if structure not in STRUCTURES:
        raise ValueError(f"Unknown structure '{structure}'. "
                         f"Valid: {sorted(STRUCTURES.keys())}")

    # Validate distributions per structure
    if structure == 'single':
        if 'dist' not in spec:
            raise ValueError("single structure requires 'dist' field")
        validate_stage_compat(spec['dist'], 'single')

    elif structure == 'hurdle':
        for field in ('s1_dist', 's2_dist'):
            if field not in spec:
                raise ValueError(f"hurdle structure requires '{field}' field")
        validate_stage_compat(spec['s1_dist'], 's1')
        validate_stage_compat(spec['s2_dist'], 's2_count')

    elif structure == 'pot':
        for field in ('s1_dist', 'bulk_dist', 'tail_dist'):
            if field not in spec:
                raise ValueError(f"pot structure requires '{field}' field")
        validate_stage_compat(spec['s1_dist'], 's1')
        validate_stage_compat(spec['bulk_dist'], 'bulk')
        validate_stage_compat(spec['tail_dist'], 'tail')

    elif structure == 'double_hurdle':
        for field in ('s1_dist', 's2_dist', 'bulk_dist', 'tail_dist'):
            if field not in spec:
                raise ValueError(f"double_hurdle structure requires '{field}' field")
        validate_stage_compat(spec['s1_dist'], 's1')
        validate_stage_compat(spec['s2_dist'], 's2_binary')
        validate_stage_compat(spec['bulk_dist'], 'bulk')
        validate_stage_compat(spec['tail_dist'], 'tail')


def _auto_label(spec: dict) -> str:
    """Generate a deterministic label from spec fields."""
    structure = spec['structure']
    parts = [structure]

    if structure == 'single':
        parts.append(spec['dist'])
    elif structure == 'hurdle':
        parts.extend([spec['s1_dist'], spec['s2_dist']])
    elif structure == 'pot':
        parts.extend([spec['s1_dist'], spec['bulk_dist'], spec['tail_dist']])
        parts.append(f"pct{spec.get('threshold_pct', 90)}")
    elif structure == 'double_hurdle':
        parts.extend([spec['s1_dist'], spec['s2_dist'],
                      spec['bulk_dist'], spec['tail_dist']])
        parts.append(f"pct{spec.get('threshold_pct', 90)}")

    # Add covariate info
    parts.append(spec.get('covars', 'wind_sdi'))

    # Add per-stage covars if different from default
    default_covars = spec.get('covars', 'wind_sdi')
    for stage in ('s1', 's2', 'bulk', 'tail'):
        stage_covars = spec.get(f'{stage}_covars')
        if stage_covars and stage_covars != default_covars:
            parts.append(f"{stage}={stage_covars}")

    # Add interaction flags if set
    if spec.get('interaction'):
        parts.append('interact')
    if spec.get('island_interact'):
        parts.append('island_int')

    return '_'.join(parts)


# ============================================================================
# Internal: run one train/test split
# ============================================================================

def _run_one_split(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    spec:  dict,
    return_predictions: bool = False,
    verbose: bool = True,
) -> Optional[Union[dict, Tuple]]:
    """
    Fit on train, predict on test, return metrics (and optionally raw arrays).
    Returns None on any exception.
    """
    try:
        fitted = _fit(train, spec)
        if fitted is None:
            return None
        preds = _predict(fitted, test, spec)

        observed = test['total_deaths'].values
        exposure = test['exposed_population'].values
        preds    = np.maximum(preds, 0)

        metrics = calc_metrics(observed, preds, exposure)

        if return_predictions:
            return metrics, observed, preds, exposure
        return metrics

    except Exception as e:
        if verbose:
            print(f"  Error [{spec.get('label','?')}]: {e}")
        return None


# ============================================================================
# Public: run_model_evaluation
# ============================================================================

def run_model_evaluation(
    data:       pd.DataFrame,
    model_spec: Dict,
    seeds:      List[int] = DEFAULT_SEEDS,
    k_folds:    int  = DEFAULT_FOLDS,
    verbose:    bool = True,
    n_jobs:     int  = 1,
    return_predictions: bool = False,
) -> Dict:
    """
    Full in-sample + CV evaluation for one model spec.

    Parameters
    ----------
    data       : full dataset
    model_spec : dict — see module docstring for fields
    seeds      : random seeds for CV splits
    k_folds    : folds per seed (0 = in-sample only)
    verbose    : print progress
    n_jobs     : parallel CV workers (-1 = all cores)
    return_predictions : attach in-sample predictions to result dict

    Returns
    -------
    dict with keys: spec, insample, oos_raw, oos_summary
    (+ predictions, observed, exposure, total_predicted,
       total_observed, ratio_pred_obs  if return_predictions=True)
    """
    spec = dict(model_spec)          # don't mutate caller's dict
    _validate_spec(spec)
    if 'label' not in spec or spec['label'] is None:
        spec['label'] = _auto_label(spec)

    _NAN_METRICS = {k: np.nan for k in [
        'mae_rate', 'rmse_rate', 'mae_log', 'cor_rate',
        'mae_count', 'rmse_count', 'cor_count', 'zero_acc',
        'skill_mae_rate', 'skill_rmse_rate',
        'skill_mae_count', 'skill_rmse_count',
    ]}

    if verbose:
        print(f"Evaluating: {spec['label']}")

    # ── In-sample ────────────────────────────────────────────────────────────
    if verbose:
        print("  In-sample...", end='', flush=True)

    insample_result = _run_one_split(data, data, spec,
                                     return_predictions=True, verbose=verbose)
    if insample_result is None:
        insample_metrics = _NAN_METRICS.copy()
        predictions_data = None
    else:
        _, obs, pred, exp = insample_result
        nz = obs > 0
        baseline = float((obs[nz] / exp[nz] * 1e5).mean()) if nz.sum() > 0 else None
        insample_metrics = calc_metrics(obs, pred, exp, baseline_rate=baseline)
        predictions_data = dict(
            observed=obs, predicted=pred, exposure=exp,
            total_predicted=pred.sum(), total_observed=obs.sum(),
            ratio_pred_obs=(pred.sum() / obs.sum() if obs.sum() > 0 else np.nan),
        )

    if verbose:
        print(" done")

    # ── In-sample only (k_folds=0) ───────────────────────────────────────────
    if k_folds == 0:
        if verbose:
            print("  Skipping OOS (k_folds=0)")
        result = dict(
            spec=spec,
            insample=insample_metrics,
            oos_raw=pd.DataFrame(),
            oos_summary=dict(
                n_tests=0,
                **{k: insample_metrics.get(k, np.nan)
                   for k in ['mae_rate', 'rmse_rate', 'mae_log', 'cor_rate',
                              'mae_count', 'rmse_count', 'cor_count', 'zero_acc']},
                sd_mae_rate=0.0, sd_mae_count=0.0,
            ),
        )
        if return_predictions and predictions_data:
            result.update(
                predictions=predictions_data['predicted'],
                observed=predictions_data['observed'],
                exposure=predictions_data['exposure'],
                total_predicted=predictions_data['total_predicted'],
                total_observed=predictions_data['total_observed'],
                ratio_pred_obs=predictions_data['ratio_pred_obs'],
            )
        return result

    # ── CV ───────────────────────────────────────────────────────────────────
    if verbose:
        print("  OOS CV...", end='', flush=True)

    spec_id = spec_to_id(spec)

    def _one_fold(seed, fold):
        rng = np.random.default_rng(seed)
        fold_ids = rng.integers(0, k_folds, size=len(data))
        tr = data[fold_ids != fold].copy()
        te = data[fold_ids == fold].copy()

        try:
            fitted = _fit(tr, spec)
            if fitted is None:
                return None
            preds = _predict(fitted, te, spec)
            observed = te['total_deaths'].values
            exposure = te['exposed_population'].values
            preds = np.maximum(preds, 0)
            metrics = calc_metrics(observed, preds, exposure)
            metrics.update(seed=seed, fold=fold)
            save_model(spec_id, 'oos', fitted, seed=seed, fold=fold)
            log_to_manifest(spec_id, spec, 'oos', metrics, seed=seed, fold=fold)
            return dict(metrics=metrics,
                        predictions=dict(observed=observed, predicted=preds,
                                         exposure=exposure, seed=seed, fold=fold))
        except Exception as e:
            if verbose:
                print(f"  Error [{spec.get('label','?')} seed={seed} fold={fold}]: {e}")
            return None

    fold_jobs = [(s, f) for s in seeds for f in range(k_folds)]

    if n_jobs != 1:
        raw_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
            delayed(_one_fold)(s, f) for s, f in fold_jobs
        )
    else:
        raw_results = []
        for s, f in fold_jobs:
            if verbose:
                print(f"  seed={s} fold={f}", end='\r', flush=True)
            raw_results.append(_one_fold(s, f))

    if verbose:
        print()

    fold_metrics = [r['metrics']      for r in raw_results if r is not None]
    fold_preds   = [r['predictions']  for r in raw_results if r is not None]

    oos_df = pd.DataFrame(fold_metrics)

    if not fold_preds:
        if verbose:
            print("  WARNING: all CV folds failed")
        oos_summary = dict(n_tests=0, **{k: np.nan for k in [
            'mae_rate', 'sd_mae_rate', 'rmse_rate', 'mae_log', 'cor_rate',
            'mae_count', 'sd_mae_count', 'rmse_count', 'cor_count', 'zero_acc',
        ]})
    else:
        all_obs  = np.concatenate([p['observed']  for p in fold_preds])
        all_pred = np.concatenate([p['predicted'] for p in fold_preds])
        all_exp  = np.concatenate([p['exposure']  for p in fold_preds])

        nz = all_obs > 0
        baseline = float((all_obs[nz] / all_exp[nz] * 1e5).mean()) if nz.sum() > 0 else None
        om = calc_metrics(all_obs, all_pred, all_exp, baseline_rate=baseline)

        oos_summary = dict(
            n_tests=len(oos_df),
            mae_rate=om['mae_rate'],
            sd_mae_rate=float(oos_df['mae_rate'].std()),
            rmse_rate=om['rmse_rate'],
            mae_log=om['mae_log'],
            cor_rate=om['cor_rate'],
            mae_count=om['mae_count'],
            sd_mae_count=float(oos_df['mae_count'].std()),
            rmse_count=om['rmse_count'],
            cor_count=om['cor_count'],
            zero_acc=om['zero_acc'],
        )
        for k in ('skill_mae_rate', 'skill_rmse_rate',
                  'skill_mae_count', 'skill_rmse_count'):
            if k in om:
                oos_summary[k] = om[k]

    result = dict(spec=spec, insample=insample_metrics,
                  oos_raw=oos_df, oos_summary=oos_summary)

    if return_predictions and predictions_data:
        result.update(
            predictions=predictions_data['predicted'],
            observed=predictions_data['observed'],
            exposure=predictions_data['exposure'],
            total_predicted=predictions_data['total_predicted'],
            total_observed=predictions_data['total_observed'],
            ratio_pred_obs=predictions_data['ratio_pred_obs'],
        )
    return result


# ============================================================================
# Public: run_multiple_models
# ============================================================================

def run_multiple_models(
    data:        pd.DataFrame,
    model_specs: List[Dict],
    seeds:       List[int] = DEFAULT_SEEDS,
    k_folds:     int  = DEFAULT_FOLDS,
    verbose:     bool = True,
    return_predictions: bool = False,
) -> Dict:
    """
    Run a list of model specs and return results + summary table.

    Returns
    -------
    dict with:
      'results' : {label: result_dict, ...}
      'summary' : DataFrame, one row per model
    """
    results = {}
    for spec in model_specs:
        r = run_model_evaluation(
            data, spec, seeds=seeds, k_folds=k_folds,
            verbose=verbose, return_predictions=return_predictions,
        )
        results[r['spec']['label']] = r

    rows = []
    for label, r in results.items():
        spec = r['spec']
        structure = spec['structure']
        row = dict(
            label=label,
            structure=structure,
            covars=spec.get('covars', 'wind_sdi'),
        )

        # Add distribution names based on structure
        if structure == 'single':
            row['dist'] = spec['dist']
            row['short_label'] = f"single_{spec['dist']}"
        elif structure == 'hurdle':
            row['s1_dist'] = spec['s1_dist']
            row['s2_dist'] = spec['s2_dist']
            row['short_label'] = f"hurdle_{spec['s2_dist']}"
        elif structure == 'pot':
            row['s1_dist'] = spec['s1_dist']
            row['bulk_dist'] = spec['bulk_dist']
            row['tail_dist'] = spec['tail_dist']
            row['threshold_pct'] = spec.get('threshold_pct', 90)
            row['short_label'] = f"pot@{row['threshold_pct']}"
        elif structure == 'double_hurdle':
            row['s1_dist'] = spec['s1_dist']
            row['s2_dist'] = spec['s2_dist']
            row['bulk_dist'] = spec['bulk_dist']
            row['tail_dist'] = spec['tail_dist']
            row['threshold_pct'] = spec.get('threshold_pct', 90)
            row['short_label'] = f"dh({spec['bulk_dist']}/{spec['tail_dist']})@{row['threshold_pct']}"

        row.update(r['oos_summary'])
        if return_predictions:
            row['total_predicted'] = r.get('total_predicted', np.nan)
            row['total_observed']  = r.get('total_observed',  np.nan)
            row['ratio_pred_obs']  = r.get('ratio_pred_obs',  np.nan)
        rows.append(row)

    return dict(results=results, summary=pd.DataFrame(rows))


# ============================================================================
# Public: run_pot_threshold_sweep
# ============================================================================

def run_pot_threshold_sweep(
    data:             pd.DataFrame,
    covars:           str,
    tail_covars:      str       = 'none',
    threshold_pcts:   List[int] = (70, 75, 80, 85, 90, 95),
    seeds:            List[int] = DEFAULT_SEEDS,
    k_folds:          int       = DEFAULT_FOLDS,
    verbose:          bool      = True,
) -> pd.DataFrame:
    """
    Sweep POT threshold percentiles and compare standard + tail metrics.

    For each threshold_pct in threshold_pcts, fits POT and a baseline
    hurdle NB with the same covariates.  Returns a DataFrame comparing:
      - Standard metrics  (mae_rate, rmse_rate, cor_rate, etc.)
      - Tail metrics      (tail_mae_count, tail_coverage_10, etc.)
      - n_tail            (number of tail storms at each threshold)

    The baseline hurdle_nb row is included once (threshold_pct = NA).

    Parameters
    ----------
    data           : full dataset
    covars         : covariate tokens for logistic + NB components
    tail_covars    : covariate tokens for GPD tail ('none' recommended)
    threshold_pcts : list of percentiles to evaluate
    seeds, k_folds : CV parameters
    verbose        : print progress

    Returns
    -------
    DataFrame with one row per model/threshold, sorted by tail_mae_count.
    """
    rows = []

    # ── Baseline: hurdle NB ──────────────────────────────────────────────────
    baseline_spec = dict(
        structure='hurdle',
        s1_dist='statsmodels_logistic',
        s2_dist='statsmodels_nb',
        covars=covars,
    )
    if verbose:
        print(f"\n{'='*60}")
        print("Baseline: hurdle NB")
    nb_result = run_model_evaluation(data, baseline_spec, seeds=seeds,
                                     k_folds=k_folds, verbose=verbose,
                                     return_predictions=True)

    # Tail metrics for baseline (using median of positives as threshold reference)
    y_pos = data.loc[data['total_deaths'] > 0, 'total_deaths'].values
    for pct in threshold_pcts:
        u = float(np.percentile(y_pos, pct))
        if 'predictions' in nb_result:
            tm = calc_tail_metrics(
                nb_result['observed'], nb_result['predictions'],
                nb_result['exposure'], threshold=u,
            )
        else:
            tm = {}
        row = dict(model='hurdle_nb', threshold_pct=pct, threshold_u=u)
        row.update(nb_result['oos_summary'])
        row.update(tm)
        rows.append(row)

    # ── POT sweep ────────────────────────────────────────────────────────────
    for pct in threshold_pcts:
        spec = dict(
            structure='pot',
            s1_dist='statsmodels_logistic',
            bulk_dist='statsmodels_nb',
            tail_dist='scipy_gpd',
            covars=covars,
            tail_covars=tail_covars,
            threshold_pct=pct,
        )
        if verbose:
            print(f"\n{'='*60}")
            print(f"POT threshold = {pct}th percentile")

        r = run_model_evaluation(data, spec, seeds=seeds, k_folds=k_folds,
                                 verbose=verbose, return_predictions=True)

        u = float(np.percentile(y_pos, pct))
        if 'predictions' in r:
            tm = calc_tail_metrics(
                r['observed'], r['predictions'], r['exposure'], threshold=u,
            )
        else:
            tm = {}

        row = dict(model='pot', threshold_pct=pct, threshold_u=u)
        row.update(r['oos_summary'])
        row.update(tm)
        rows.append(row)

    sweep_df = pd.DataFrame(rows)

    # Sort by tail_mae_count so best tail performance is at top
    if 'tail_mae_count' in sweep_df.columns:
        sweep_df = sweep_df.sort_values('tail_mae_count').reset_index(drop=True)

    return sweep_df