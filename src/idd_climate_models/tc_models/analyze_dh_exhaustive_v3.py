"""
analyze_dh_exhaustive_v3.py

Exhaustive enumeration of v3 DH model combinations.
Rate-based threshold: bulk/tail split on death_rate = total_deaths / exposed_population.

Key differences from analyze_dh_exhaustive.py (v2):
- Points to stage_results_dh_v3/ — never mixes with v2 artifacts
- Threshold is rate-based (death_rate percentile) throughout
- Only assembles same-covariate DH models (s1/s2/bulk/tail all share the same covar set)
- GLM-only: NB, gamma, lognormal, Poisson (ML and GPD removed)
- Predictions constrained in rate space: bulk in (0, u], tail in [u, inf)
- Bulk/tail predict with exposure=1 (rate), clipped, then multiplied by exposure for counts

Usage:
    python src/idd_climate_models/tc_models/analyze_dh_exhaustive_v3.py
"""

import sys
import pickle
import warnings
import os
import time
from pathlib import Path
from itertools import product
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

_DH_ROOT = Path('/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/stage_results_dh_v3')
os.environ['STAGE_MODELS_DIR'] = str(_DH_ROOT / 'models')
os.environ['STAGE_RESULTS_DIR'] = str(_DH_ROOT / 'stage_logs')

sys.path.insert(0, str(Path(__file__).parents[4] / 'src'))

from idd_climate_models.tc_models.cache import load_stage_manifest, STAGE_MODELS_DIR
from idd_climate_models.tc_models.constants import STAGE_RESULTS_DIR
from idd_climate_models.tc_models.data import load_tc_data
from idd_climate_models.tc_models.features import build_X, align_X
from idd_climate_models.tc_models.metrics import calc_metrics
from idd_climate_models.tc_models.distributions import DISTRIBUTIONS

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS   = [123, 456, 789, 101112, 131415]
K_FOLDS = 5
THREADS = 8

COVAR_SETS = [
    'none',
    'wind_sdi',
    'wind_sdi_basin_island',
]

S1_DISTS   = ['statsmodels_logistic']
S2_DISTS   = ['statsmodels_logistic']
BULK_DISTS = ['statsmodels_nb', 'statsmodels_gamma', 'statsmodels_lognormal',
              'statsmodels_poisson']
TAIL_DISTS = ['statsmodels_gamma', 'statsmodels_lognormal', 'statsmodels_poisson']

THRESHOLDS = [70, 75, 80, 85, 90, 95]


def _model_path(stage_id, seed, fold):
    return STAGE_MODELS_DIR / f'{stage_id}_seed{seed}_fold{fold}.pkl'


def _insample_path(stage_id):
    return STAGE_MODELS_DIR / f'{stage_id}_insample.pkl'


# Minimum rate prediction for bulk (must be positive, never zero)
_BULK_FLOOR = 1e-9


def _rate_threshold(pos_data, thr_pct):
    """Compute rate-based threshold from positive-death training rows."""
    rates = pos_data['total_deaths'].values / pos_data['exposed_population'].values
    return float(np.percentile(rates, thr_pct))


def _predict_stage(args):
    """Load a stage pkl and generate predictions on the correct test subset."""
    stage_id, seed, fold, data, meta = args
    try:
        stype   = meta['stage_type']
        dist    = meta['dist']
        covars  = meta['covars']
        thr_pct = meta.get('threshold_pct')

        pkl_path = _model_path(stage_id, seed, fold)
        if not pkl_path.exists():
            return (stage_id, seed, fold, None)

        with open(pkl_path, 'rb') as f:
            fitted = pickle.load(f)

        dist_mod = DISTRIBUTIONS[dist]

        rng       = np.random.default_rng(seed)
        fold_ids  = rng.integers(0, K_FOLDS, size=len(data))
        train_idx = np.where(fold_ids != fold)[0]
        test_idx  = np.where(fold_ids == fold)[0]
        train = data.iloc[train_idx]
        test  = data.iloc[test_idx]

        if stype == 's1':
            test_sub = test
            sub_idx  = test_idx
            preds    = dist_mod.predict(fitted, build_X(test_sub, covars, include_log_exp=False),
                                        task='binary')

        elif stype == 'pos_binary':
            test_sub = test[test['death_y_n'] == 1]
            sub_idx  = test_idx[test['death_y_n'].values == 1]
            X_tr = build_X(train[train['death_y_n'] == 1], covars, include_log_exp=False)
            X_te = align_X(build_X(test_sub, covars, include_log_exp=False), list(X_tr.columns))
            preds = dist_mod.predict(fitted, X_te, task='binary')

        elif stype == 'dh_bulk':
            train_pos = train[train['death_y_n'] == 1]
            u         = _rate_threshold(train_pos, thr_pct)
            test_pos  = test[test['death_y_n'] == 1]
            test_rates = test_pos['total_deaths'].values / test_pos['exposed_population'].values
            test_sub  = test_pos[test_rates <= u]
            sub_idx   = test_idx[
                (test['death_y_n'].values == 1) &
                (test['total_deaths'].values / test['exposed_population'].values <= u)
            ]
            train_bulk = train_pos[
                train_pos['total_deaths'].values / train_pos['exposed_population'].values <= u
            ]
            X_tr   = build_X(train_bulk, covars, include_log_exp=False)
            X_te   = align_X(build_X(test_sub, covars, include_log_exp=False), list(X_tr.columns))
            exp_te = test_sub['exposed_population'].values
            # Predict in rate space (exposure=1), clip to (0, u], convert to count
            rate_preds = np.asarray(
                dist_mod.predict(fitted, X_te, exposure=np.ones(len(test_sub)), task='count'),
                dtype=float)
            rate_preds = np.clip(rate_preds, _BULK_FLOOR, u)
            preds = rate_preds * exp_te

        elif stype == 'tail':
            train_pos = train[train['death_y_n'] == 1]
            u         = _rate_threshold(train_pos, thr_pct)
            test_pos  = test[test['death_y_n'] == 1]
            test_rates = test_pos['total_deaths'].values / test_pos['exposed_population'].values
            test_sub  = test_pos[test_rates > u]
            sub_idx   = test_idx[
                (test['death_y_n'].values == 1) &
                (test['total_deaths'].values / test['exposed_population'].values > u)
            ]
            train_tail = train_pos[
                train_pos['total_deaths'].values / train_pos['exposed_population'].values > u
            ]
            X_tr   = build_X(train_tail, covars, include_log_exp=False)
            X_te   = align_X(build_X(test_sub, covars, include_log_exp=False), list(X_tr.columns))
            exp_te = test_sub['exposed_population'].values
            # Predict in rate space (exposure=1), clip to [u, inf), convert to count
            rate_preds = np.asarray(
                dist_mod.predict(fitted, X_te, exposure=np.ones(len(test_sub))),
                dtype=float)
            rate_preds = np.maximum(rate_preds, u)
            preds = rate_preds * exp_te

        else:
            return (stage_id, seed, fold, None)

        if len(test_sub) == 0:
            return (stage_id, seed, fold, None)

        preds = np.asarray(preds, dtype=float)
        return (stage_id, seed, fold, pd.Series(preds, index=sub_idx))

    except Exception as e:
        return (stage_id, seed, fold, None, str(e))


def _predict_insample_stage(args):
    """Load insample pkl and generate predictions on the full dataset."""
    stage_id, data, meta = args
    try:
        stype   = meta['stage_type']
        dist    = meta['dist']
        covars  = meta['covars']
        thr_pct = meta.get('threshold_pct')

        pkl_path = _insample_path(stage_id)
        if not pkl_path.exists():
            return (stage_id, None)

        with open(pkl_path, 'rb') as f:
            fitted = pickle.load(f)

        dist_mod = DISTRIBUTIONS[dist]
        n        = len(data)
        pos_data = data[data['death_y_n'] == 1]

        if stype == 's1':
            X     = build_X(data, covars, include_log_exp=False)
            preds = np.asarray(dist_mod.predict(fitted, X, task='binary'), dtype=float)
            return (stage_id, pd.Series(preds, index=range(n)))

        elif stype == 'pos_binary':
            pos_idx = np.where(data['death_y_n'].values == 1)[0]
            if len(pos_idx) == 0:
                return (stage_id, None)
            X     = build_X(data.iloc[pos_idx], covars, include_log_exp=False)
            preds = np.asarray(dist_mod.predict(fitted, X, task='binary'), dtype=float)
            return (stage_id, pd.Series(preds, index=pos_idx))

        elif stype == 'dh_bulk':
            u         = _rate_threshold(pos_data, thr_pct)
            all_rates = data['total_deaths'].values / data['exposed_population'].values
            bulk_idx  = np.where((data['death_y_n'].values == 1) & (all_rates <= u))[0]
            if len(bulk_idx) == 0:
                return (stage_id, None)
            bulk_data = data.iloc[bulk_idx]
            X   = build_X(bulk_data, covars, include_log_exp=False)
            exp = bulk_data['exposed_population'].values
            # Rate space: predict with exposure=1, clip to (0, u], convert to count
            rate_preds = np.asarray(
                dist_mod.predict(fitted, X, exposure=np.ones(len(bulk_idx)), task='count'),
                dtype=float)
            rate_preds = np.clip(rate_preds, _BULK_FLOOR, u)
            return (stage_id, pd.Series(rate_preds * exp, index=bulk_idx))

        elif stype == 'tail':
            u         = _rate_threshold(pos_data, thr_pct)
            all_rates = data['total_deaths'].values / data['exposed_population'].values
            tail_idx  = np.where((data['death_y_n'].values == 1) & (all_rates > u))[0]
            if len(tail_idx) == 0:
                return (stage_id, None)
            tail_data = data.iloc[tail_idx]
            X   = build_X(tail_data, covars, include_log_exp=False)
            exp = tail_data['exposed_population'].values
            # Rate space: predict with exposure=1, clip to [u, inf), convert to count
            rate_preds = np.asarray(
                dist_mod.predict(fitted, X, exposure=np.ones(len(tail_idx))),
                dtype=float)
            rate_preds = np.maximum(rate_preds, u)
            return (stage_id, pd.Series(rate_preds * exp, index=tail_idx))

        return (stage_id, None)

    except Exception as e:
        return (stage_id, None, str(e))


def precompute_predictions(stage_ids, stage_meta_lookup, data):
    jobs = [
        (sid, seed, fold, data, stage_meta_lookup[sid])
        for sid in stage_ids
        for seed in SEEDS
        for fold in range(K_FOLDS)
        if _model_path(sid, seed, fold).exists()
    ]
    print(f"  Loading {len(jobs)} stage×fold predictions ({THREADS} threads)...")
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        results = list(ex.map(_predict_stage, jobs))

    cache = {}
    n_ok = n_fail = 0
    errors = []
    for result in results:
        if len(result) == 4:
            sid, seed, fold, preds = result
        else:
            sid, seed, fold, preds, err = result
            errors.append(err)
        if preds is not None:
            cache[(sid, seed, fold)] = preds
            n_ok += 1
        else:
            n_fail += 1
    print(f"  Loaded {n_ok}, failed/missing {n_fail}")
    if errors:
        print(f"  Sample errors: {errors[:3]}")
    return cache


def precompute_insample_predictions(stage_ids, stage_meta_lookup, data):
    jobs = [
        (sid, data, stage_meta_lookup[sid])
        for sid in stage_ids
        if _insample_path(sid).exists()
    ]
    print(f"  Loading {len(jobs)} insample stage predictions ({THREADS} threads)...")
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        results = list(ex.map(_predict_insample_stage, jobs))

    cache = {}
    n_ok = n_fail = 0
    for result in results:
        sid, preds = result[0], result[1]
        if preds is not None:
            cache[sid] = preds
            n_ok += 1
        else:
            n_fail += 1
    print(f"  Loaded {n_ok}, failed/missing {n_fail}")
    return cache


def _coverage_at(obs, pred, frac):
    k = max(1, int(round(len(obs) * frac)))
    top_obs  = set(np.argsort(obs)[-k:])
    top_pred = set(np.argsort(pred)[-k:])
    return len(top_obs & top_pred) / k


def build_aligned_cache(cache, data, seeds, k_folds):
    obs_parts = []
    exp_parts = []
    fold_slices = {}

    idx = 0
    for seed in seeds:
        for fold in range(k_folds):
            rng = np.random.default_rng(seed)
            fold_ids = rng.integers(0, k_folds, size=len(data))
            test_idx = np.where(fold_ids == fold)[0]

            obs_parts.append(data.iloc[test_idx]['total_deaths'].values)
            exp_parts.append(data.iloc[test_idx]['exposed_population'].values)

            fold_slices[(seed, fold)] = (idx, idx + len(test_idx), test_idx)
            idx += len(test_idx)

    obs_concat = np.concatenate(obs_parts)
    exp_concat = np.concatenate(exp_parts)

    stage_folds = {}
    for (sid, seed, fold), preds in cache.items():
        if preds is None:
            continue
        if sid not in stage_folds:
            stage_folds[sid] = []
        stage_folds[sid].append((seed, fold, preds))

    stage_preds = {}
    for sid, fold_data in stage_folds.items():
        if len(fold_data) != len(seeds) * k_folds:
            continue
        fold_data.sort(key=lambda x: (seeds.index(x[0]), x[1]))
        parts = []
        for seed, fold, preds_series in fold_data:
            start, end, test_idx = fold_slices[(seed, fold)]
            aligned = preds_series.reindex(test_idx).fillna(0.0).values
            parts.append(aligned)
        stage_preds[sid] = np.concatenate(parts)

    return obs_concat, exp_concat, stage_preds


def assemble_dh_fast(s1_sid, pb_sid, bulk_sid, tail_sid, obs, exp, stage_preds):
    """E[Y] = p_pos * ((1 - p_high) * e_bulk + p_high * e_tail)"""
    if not all(sid in stage_preds for sid in [s1_sid, pb_sid, bulk_sid, tail_sid]):
        return None

    pred = (stage_preds[s1_sid] *
            ((1 - stage_preds[pb_sid]) * stage_preds[bulk_sid] +
             stage_preds[pb_sid] * stage_preds[tail_sid]))

    m = calc_metrics(obs, pred, exp)
    for frac, label in [(0.01, 'cov_1'), (0.05, 'cov_5'), (0.10, 'cov_10'), (0.20, 'cov_20')]:
        m[label] = _coverage_at(obs, pred, frac)
    m['pred_obs_ratio'] = pred.sum() / max(obs.sum(), 1)
    m['total_pred'] = pred.sum()
    m['total_obs']  = obs.sum()
    return m


def assemble_dh_insample(s1_sid, pb_sid, bulk_sid, tail_sid, is_cache, data):
    if not all(sid in is_cache for sid in [s1_sid, pb_sid, bulk_sid, tail_sid]):
        return None

    n = len(data)
    p_pos  = is_cache[s1_sid].reindex(range(n)).fillna(0.5).values
    p_high = np.zeros(n)
    e_bulk = np.zeros(n)
    e_tail = np.zeros(n)

    for i in is_cache[pb_sid].index:   p_high[i] = is_cache[pb_sid][i]
    for i in is_cache[bulk_sid].index: e_bulk[i] = is_cache[bulk_sid][i]
    for i in is_cache[tail_sid].index: e_tail[i] = is_cache[tail_sid][i]

    obs  = data['total_deaths'].values
    exp  = data['exposed_population'].values
    pred = np.maximum(p_pos * ((1 - p_high) * e_bulk + p_high * e_tail), 0)

    ro = (obs / exp) * 1e5
    rp = (pred / exp) * 1e5
    nz = obs > 0

    return {
        'mae_rate_is':      float(np.mean(np.abs(ro - rp))),
        'rmse_rate_is':     float(np.sqrt(np.mean((ro - rp) ** 2))),
        'mae_count_is':     float(np.mean(np.abs(obs - pred))),
        'rmse_count_is':    float(np.sqrt(np.mean((obs - pred) ** 2))),
        'cor_rate_is':      float(np.corrcoef(ro[nz], rp[nz])[0, 1]) if nz.sum() > 2 else np.nan,
        'cor_count_is':     float(np.corrcoef(obs[nz], pred[nz])[0, 1]) if nz.sum() > 2 else np.nan,
        'pred_obs_ratio_is': float(pred.sum() / obs.sum()) if obs.sum() > 0 else np.nan,
    }


def main():
    print("=" * 72)
    print("EXHAUSTIVE DH ENUMERATION — v3 (rate-based threshold)")
    print("=" * 72)
    print(f"Covariate sets: {COVAR_SETS}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Bulk dists: {BULK_DISTS}")
    print(f"Tail dists: {TAIL_DISTS}")
    print(f"Constraint: same covariate set across all 4 stages")

    n_dist_combos  = len(S1_DISTS) * len(S2_DISTS) * len(BULK_DISTS) * len(TAIL_DISTS)
    n_total        = n_dist_combos * len(COVAR_SETS) * len(THRESHOLDS)
    print(f"\nExpected: {n_dist_combos} dist × {len(COVAR_SETS)} covar × {len(THRESHOLDS)} thr = {n_total} models")

    print("\nLoading stage manifest...")
    manifest = load_stage_manifest()
    oos = manifest[manifest['fit_type'] == 'oos'].copy()
    print(f"  {len(oos)} OOS rows")

    stage_lookup = {}
    stage_meta   = {}
    for _, row in oos.drop_duplicates('stage_id').iterrows():
        sid     = row['stage_id']
        thr     = row.get('threshold_pct')
        thr_key = None if pd.isna(thr) else float(thr)
        key = (row['stage_type'], row['dist'], row['covars'], thr_key)
        stage_lookup[key] = sid
        stage_meta[sid] = {
            'stage_type':   row['stage_type'],
            'dist':         row['dist'],
            'covars':       row['covars'],
            'threshold_pct': thr_key,
        }

    needed_stages = set()
    for thr in THRESHOLDS:
        for cov in COVAR_SETS:
            for dist in S1_DISTS:
                k = ('s1', dist, cov, None)
                if k in stage_lookup: needed_stages.add(stage_lookup[k])
            for dist in S2_DISTS:
                k = ('pos_binary', dist, cov, float(thr))
                if k in stage_lookup: needed_stages.add(stage_lookup[k])
            for dist in BULK_DISTS:
                k = ('dh_bulk', dist, cov, float(thr))
                if k in stage_lookup: needed_stages.add(stage_lookup[k])
            for dist in TAIL_DISTS:
                k = ('tail', dist, cov, float(thr))
                if k in stage_lookup: needed_stages.add(stage_lookup[k])

    print(f"\nUnique stages needed: {len(needed_stages)}")

    print("\nLoading TC data...")
    data = load_tc_data()
    print(f"  {len(data)} rows")

    print("\nPrecomputing OOS predictions...")
    cache = precompute_predictions(list(needed_stages), stage_meta, data)

    print("\nBuilding pre-concatenated OOS arrays...")
    obs_concat, exp_concat, stage_preds = build_aligned_cache(cache, data, SEEDS, K_FOLDS)
    print(f"  {len(stage_preds)} stages with complete OOS predictions")

    print("\nPrecomputing IS predictions...")
    is_cache = precompute_insample_predictions(list(needed_stages), stage_meta, data)
    print(f"  {len(is_cache)} stages with IS predictions")

    print(f"\nAssembling up to {n_total:,} DH models...")
    results = []
    n_done = n_skip = 0
    t_start = t_last = time.time()

    for thr in THRESHOLDS:
        for cov in COVAR_SETS:
            # Same covariate set across all stages
            for s1_dist, s2_dist, bulk_dist, tail_dist in product(
                    S1_DISTS, S2_DISTS, BULK_DISTS, TAIL_DISTS):

                s1_key   = ('s1',          s1_dist,   cov, None)
                s2_key   = ('pos_binary',  s2_dist,   cov, float(thr))
                bulk_key = ('dh_bulk',     bulk_dist, cov, float(thr))
                tail_key = ('tail',        tail_dist, cov, float(thr))

                if not all(k in stage_lookup for k in [s1_key, s2_key, bulk_key, tail_key]):
                    n_skip += 1
                    continue

                s1_sid   = stage_lookup[s1_key]
                s2_sid   = stage_lookup[s2_key]
                bulk_sid = stage_lookup[bulk_key]
                tail_sid = stage_lookup[tail_key]

                m_oos = assemble_dh_fast(s1_sid, s2_sid, bulk_sid, tail_sid,
                                         obs_concat, exp_concat, stage_preds)
                if m_oos is None:
                    n_skip += 1
                    continue

                m_is = assemble_dh_insample(s1_sid, s2_sid, bulk_sid, tail_sid, is_cache, data)

                row = {
                    'threshold_pct': thr,
                    's1_dist':   s1_dist,   's1_cov':   cov, 's1_sid':   s1_sid,
                    's2_dist':   s2_dist,   's2_cov':   cov, 's2_sid':   s2_sid,
                    'bulk_dist': bulk_dist, 'bulk_cov': cov, 'bulk_sid': bulk_sid,
                    'tail_dist': tail_dist, 'tail_cov': cov, 'tail_sid': tail_sid,
                    **m_oos,
                }
                if m_is is not None:
                    row.update(m_is)

                results.append(row)
                n_done += 1

    print(f"\nAssembled {n_done:,} models, skipped {n_skip:,}")

    if not results:
        print("No results — are the stage pkl files present?")
        return

    df = pd.DataFrame(results)

    out_path = STAGE_RESULTS_DIR / 'dh_exhaustive_v3.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} models to {out_path}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"\nTotal models: {len(df)}")
    print(f"\nBy threshold:")
    print(df.groupby('threshold_pct')['mae_rate'].describe()[['count', 'mean', 'min']].to_string())
    print(f"\nOOS pred/obs ratio:")
    print(df['pred_obs_ratio'].describe())


if __name__ == '__main__':
    main()
