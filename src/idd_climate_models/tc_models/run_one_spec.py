"""
run_one_spec.py
Worker script: fit one spec (insample + OOS CV) given a spec index.

Called by jobmon with:
    python run_one_spec.py --spec_index N --grid_path /path/to/grid.json

Skips fits already logged in the per-spec result file.
"""

import argparse
import json
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_index', type=int, required=True)
    parser.add_argument('--grid_path',  type=str, required=True)
    parser.add_argument('--seeds',   type=str, default='123,456,789,101112,131415')
    parser.add_argument('--k_folds', type=int, default=5)
    args = parser.parse_args()

    with open(args.grid_path) as f:
        grid = json.load(f)

    spec   = grid[args.spec_index]
    seeds  = [int(s) for s in args.seeds.split(',')]
    k_folds = args.k_folds

    # ── Imports ───────────────────────────────────────────────────────────────
    from idd_climate_models.tc_models.data    import load_tc_data
    from idd_climate_models.tc_models.engine  import _fit, _predict, _validate_spec, _auto_label
    from idd_climate_models.tc_models.metrics import calc_metrics
    from idd_climate_models.tc_models.cache   import (
        spec_to_id, save_model, is_logged, log_to_manifest,
    )

    # ── Prepare spec ──────────────────────────────────────────────────────────
    _validate_spec(spec)
    spec['label'] = _auto_label(spec)
    spec_id = spec_to_id(spec)

    print(f"spec_id: {spec_id}")
    print(f"label:   {spec['label']}")

    data = load_tc_data()
    n    = len(data)
    print(f"data rows: {n}")

    errors = []

    # ── Insample ──────────────────────────────────────────────────────────────
    if not is_logged(spec_id, 'insample'):
        print("Fitting insample...")
        try:
            fitted  = _fit(data, spec)
            preds   = _predict(fitted, data, spec)
            obs     = data['total_deaths'].values
            exp     = data['exposed_population'].values
            preds   = np.maximum(preds, 0)
            nz      = obs > 0
            baseline = float((obs[nz] / exp[nz] * 1e5).mean()) if nz.sum() > 0 else None
            metrics = calc_metrics(obs, preds, exp, baseline_rate=baseline)
            save_model(spec_id, 'insample', fitted)
            log_to_manifest(spec_id, spec, 'insample', metrics)
            print(f"  insample done — mae_rate={metrics.get('mae_rate', float('nan')):.4f}")
        except Exception as e:
            msg = f"insample FAILED: {e}"
            print(msg, file=sys.stderr)
            errors.append(msg)
    else:
        print("insample already logged, skipping")

    # ── OOS per seed × fold ───────────────────────────────────────────────────
    for seed in seeds:
        for fold in range(k_folds):
            if is_logged(spec_id, 'oos', seed=seed, fold=fold):
                continue
            try:
                rng      = np.random.default_rng(seed)
                fold_ids = rng.integers(0, k_folds, size=n)
                train    = data[fold_ids != fold].copy()
                test     = data[fold_ids == fold].copy()

                fitted  = _fit(train, spec)
                preds   = _predict(fitted, test, spec)
                obs     = test['total_deaths'].values
                exp     = test['exposed_population'].values
                preds   = np.maximum(preds, 0)
                metrics = calc_metrics(obs, preds, exp)
                metrics.update(seed=seed, fold=fold)

                save_model(spec_id, 'oos', fitted, seed=seed, fold=fold)
                log_to_manifest(spec_id, spec, 'oos', metrics, seed=seed, fold=fold)
                print(f"  seed={seed} fold={fold} done")
            except Exception as e:
                msg = f"seed={seed} fold={fold} FAILED: {e}"
                print(msg, file=sys.stderr)
                errors.append(msg)

    if errors:
        print(f"\n{len(errors)} error(s) encountered:")
        for err in errors:
            print(f"  {err}")
        # Exit 1 only if insample failed (OOS partial failures are acceptable)
        if any('insample' in e for e in errors):
            sys.exit(1)

    print("Done.")


if __name__ == '__main__':
    main()
