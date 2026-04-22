"""
run_one_stage.py
Worker script: fit one stage (insample + OOS CV) given a stage index.

Called by jobmon with:
    python run_one_stage.py --stage_index N --grid_path /path/to/stage_grid.json

Each stage fit is independent. Skips fits already logged in the per-stage result file.
"""

import argparse
import json
import sys

import numpy as np


def _fit_and_eval(stage: dict, train, test):
    """
    Fit a stage on train data, evaluate on test data.
    Returns (fitted_model, metrics_dict).

    For insample pass train=test=full_data.
    For OOS pass the appropriate fold splits.

    Threshold u for threshold-dependent stages is always computed from
    training positives, then applied to select test rows.
    """
    from idd_climate_models.tc_models.distributions import DISTRIBUTIONS, ML_MODELS
    from idd_climate_models.tc_models.features import build_X, align_X
    from idd_climate_models.tc_models.metrics import calc_metrics, calc_binary_metrics

    stage_type    = stage['stage_type']
    dist          = stage['dist']
    covars        = stage['covars']
    threshold_pct = stage.get('threshold_pct')
    is_ml         = dist in ML_MODELS
    dist_mod      = DISTRIBUTIONS[dist]

    tr_cols = None  # set after first build_X on train; used for align_X on test

    if stage_type == 's1':
        X_tr = build_X(train, covars, include_log_exp=False)
        y_tr = train['death_y_n'].values
        fitted = dist_mod.fit(X_tr, y_tr, task='binary')

        X_te = align_X(build_X(test, covars, include_log_exp=False), list(X_tr.columns))
        y_te = test['death_y_n'].values
        preds = dist_mod.predict(fitted, X_te, task='binary')
        metrics = calc_binary_metrics(y_te, preds)
        metrics['n_obs'] = int(len(y_tr))

    elif stage_type == 'pos_count':
        train_pos = train[train['death_y_n'] == 1]
        if len(train_pos) == 0:
            raise ValueError("No positive training observations")
        X_tr   = build_X(train_pos, covars, include_log_exp=is_ml)
        y_tr   = train_pos['total_deaths'].values
        exp_tr = train_pos['exposed_population'].values
        fitted = dist_mod.fit(X_tr, y_tr, exposure=exp_tr, task='count')

        test_pos = test[test['death_y_n'] == 1]
        if len(test_pos) == 0:
            raise ValueError("No positive test observations")
        X_te   = align_X(build_X(test_pos, covars, include_log_exp=is_ml), list(X_tr.columns))
        exp_te = test_pos['exposed_population'].values
        preds  = dist_mod.predict(fitted, X_te, exposure=exp_te, task='count')
        metrics = calc_metrics(test_pos['total_deaths'].values,
                               np.maximum(preds, 0), exp_te)
        metrics['n_obs'] = int(len(train_pos))

    elif stage_type == 'pos_binary':
        train_pos = train[train['death_y_n'] == 1]
        if len(train_pos) == 0:
            raise ValueError("No positive training observations")
        train_rates = train_pos['total_deaths'].values / train_pos['exposed_population'].values
        u      = float(np.percentile(train_rates, threshold_pct))
        y_tr   = (train_rates > u).astype(int)
        X_tr   = build_X(train_pos, covars, include_log_exp=False)
        fitted = dist_mod.fit(X_tr, y_tr, task='binary')

        test_pos = test[test['death_y_n'] == 1]
        if len(test_pos) == 0:
            raise ValueError("No positive test observations")
        X_te      = align_X(build_X(test_pos, covars, include_log_exp=False), list(X_tr.columns))
        test_rates = test_pos['total_deaths'].values / test_pos['exposed_population'].values
        y_te      = (test_rates > u).astype(int)  # training u (rate)
        preds     = dist_mod.predict(fitted, X_te, task='binary')
        metrics = calc_binary_metrics(y_te, preds)
        metrics['n_obs']        = int(len(train_pos))
        metrics['n_tail_train'] = int(y_tr.sum())
        metrics['threshold']    = u

    elif stage_type == 'dh_bulk':
        train_pos = train[train['death_y_n'] == 1]
        if len(train_pos) == 0:
            raise ValueError("No positive training observations")
        train_rates = train_pos['total_deaths'].values / train_pos['exposed_population'].values
        u           = float(np.percentile(train_rates, threshold_pct))
        train_bulk  = train_pos[train_rates <= u]
        if len(train_bulk) < 5:
            raise ValueError(f"Too few bulk training obs: {len(train_bulk)}")
        X_tr   = build_X(train_bulk, covars, include_log_exp=is_ml)
        y_tr   = train_bulk['total_deaths'].values
        exp_tr = train_bulk['exposed_population'].values
        fitted = dist_mod.fit(X_tr, y_tr, exposure=exp_tr, task='count')

        test_pos   = test[test['death_y_n'] == 1]
        test_rates = test_pos['total_deaths'].values / test_pos['exposed_population'].values
        test_bulk  = test_pos[test_rates <= u]  # training u (rate)
        if len(test_bulk) == 0:
            raise ValueError("No bulk test observations")
        X_te   = align_X(build_X(test_bulk, covars, include_log_exp=is_ml), list(X_tr.columns))
        exp_te = test_bulk['exposed_population'].values
        preds  = dist_mod.predict(fitted, X_te, exposure=exp_te, task='count')
        metrics = calc_metrics(test_bulk['total_deaths'].values,
                               np.maximum(preds, 0), exp_te)
        metrics['n_obs']     = int(len(train_bulk))
        metrics['threshold'] = u

    elif stage_type == 'tail':
        train_pos = train[train['death_y_n'] == 1]
        if len(train_pos) == 0:
            raise ValueError("No positive training observations")
        train_rates = train_pos['total_deaths'].values / train_pos['exposed_population'].values
        u           = float(np.percentile(train_rates, threshold_pct))
        train_tail  = train_pos[train_rates > u]
        if len(train_tail) < 10:
            raise ValueError(f"Too few tail training obs: {len(train_tail)}")
        X_tr   = build_X(train_tail, covars, include_log_exp=False)
        y_tr   = train_tail['total_deaths'].values
        exp_tr = train_tail['exposed_population'].values
        fitted = dist_mod.fit(X_tr, y_tr, exposure=exp_tr, threshold=u)

        test_pos   = test[test['death_y_n'] == 1]
        test_rates = test_pos['total_deaths'].values / test_pos['exposed_population'].values
        test_tail  = test_pos[test_rates > u]  # training u (rate)
        if len(test_tail) == 0:
            raise ValueError("No tail test observations")
        X_te   = align_X(build_X(test_tail, covars, include_log_exp=False), list(X_tr.columns))
        exp_te = test_tail['exposed_population'].values
        preds  = dist_mod.predict(fitted, X_te, exposure=exp_te)
        metrics = calc_metrics(test_tail['total_deaths'].values,
                               np.maximum(preds, 0), exp_te)
        metrics['n_obs']     = int(len(train_tail))
        metrics['threshold'] = u

    elif stage_type == 'single':
        X_tr   = build_X(train, covars, include_log_exp=is_ml)
        y_tr   = train['total_deaths'].values
        exp_tr = train['exposed_population'].values
        fitted = dist_mod.fit(X_tr, y_tr, exposure=exp_tr, task='count')

        X_te   = align_X(build_X(test, covars, include_log_exp=is_ml), list(X_tr.columns))
        exp_te = test['exposed_population'].values
        preds  = dist_mod.predict(fitted, X_te, exposure=exp_te, task='count')
        metrics = calc_metrics(test['total_deaths'].values,
                               np.maximum(preds, 0), exp_te)
        metrics['n_obs'] = int(len(train))

    else:
        raise ValueError(f"Unknown stage_type: {stage_type!r}")

    return fitted, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage_index', type=int, required=True)
    parser.add_argument('--grid_path',   type=str, required=True)
    parser.add_argument('--seeds',   type=str, default='123,456,789,101112,131415')
    parser.add_argument('--k_folds', type=int, default=5)
    args = parser.parse_args()

    with open(args.grid_path) as f:
        grid = json.load(f)

    stage   = grid[args.stage_index]
    seeds   = [int(s) for s in args.seeds.split(',')]
    k_folds = args.k_folds

    from idd_climate_models.tc_models.data  import load_tc_data
    from idd_climate_models.tc_models.cache import (
        stage_to_id, save_stage_model, is_stage_logged, log_stage_result,
    )

    stage_id = stage_to_id(stage)
    print(f"stage_id:   {stage_id}")
    print(f"stage_type: {stage['stage_type']}")
    print(f"dist:       {stage['dist']}")
    print(f"covars:     {stage['covars']}")
    if 'threshold_pct' in stage:
        print(f"threshold:  {stage['threshold_pct']}")

    data = load_tc_data()
    n    = len(data)
    print(f"data rows:  {n}")

    errors = []

    # ── Insample ──────────────────────────────────────────────────────────────
    if not is_stage_logged(stage_id, 'insample'):
        print("Fitting insample...")
        try:
            fitted, metrics = _fit_and_eval(stage, data, data)
            save_stage_model(stage_id, 'insample', fitted)
            log_stage_result(stage_id, stage, 'insample', metrics)
            first_metric = next(iter(metrics.items()))
            print(f"  insample done — {first_metric[0]}={first_metric[1]:.4f}")
        except Exception as e:
            msg = f"insample FAILED: {e}"
            print(msg, file=sys.stderr)
            errors.append(msg)
    else:
        print("insample already logged, skipping")

    # ── OOS per seed × fold ───────────────────────────────────────────────────
    for seed in seeds:
        for fold in range(k_folds):
            if is_stage_logged(stage_id, 'oos', seed=seed, fold=fold):
                continue
            try:
                rng      = np.random.default_rng(seed)
                fold_ids = rng.integers(0, k_folds, size=n)
                train    = data[fold_ids != fold].copy()
                test     = data[fold_ids == fold].copy()

                fitted, metrics = _fit_and_eval(stage, train, test)
                metrics.update(seed=seed, fold=fold)

                save_stage_model(stage_id, 'oos', fitted, seed=seed, fold=fold)
                log_stage_result(stage_id, stage, 'oos', metrics, seed=seed, fold=fold)
                print(f"  seed={seed} fold={fold} done")
            except Exception as e:
                msg = f"seed={seed} fold={fold} FAILED: {e}"
                print(msg, file=sys.stderr)
                errors.append(msg)

    if errors:
        print(f"\n{len(errors)} error(s) encountered:")
        for err in errors:
            print(f"  {err}")

    print("Done.")


if __name__ == '__main__':
    main()
