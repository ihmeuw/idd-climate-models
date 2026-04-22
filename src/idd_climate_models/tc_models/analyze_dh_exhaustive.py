"""
analyze_dh_exhaustive.py

Exhaustive enumeration of all non-ML DH model combinations.
Computes BOTH OOS (cross-validated) and IS (in-sample) metrics in one pass.

Constraints:
- Non-ML distributions only (no sklearn_rf, sklearn_xgb)
- 16 covariate sets: full 2^4 factorial of wind, sdi, basin, island
- Single threshold (70%)
- No poisson/tweedie in bulk (poor performers)
- 1×1×3×4 distribution combinations × 16^4 covariate combinations = 786,432 models

Output: dh_exhaustive_expanded.csv with both OOS and IS metrics

Usage:
    python src/idd_climate_models/tc_models/analyze_dh_exhaustive.py
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

# Set paths to stage_results_dh_v2 BEFORE importing from cache/constants
_DH_ROOT = Path('/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/stage_results_dh_v2')
os.environ['STAGE_MODELS_DIR'] = str(_DH_ROOT / 'models')
os.environ['STAGE_RESULTS_DIR'] = str(_DH_ROOT / 'stage_logs')

sys.path.insert(0, str(Path(__file__).parents[4] / 'src'))

from idd_climate_models.tc_models.cache import load_stage_manifest, STAGE_MODELS_DIR
from idd_climate_models.tc_models.constants import STAGE_RESULTS_DIR
from idd_climate_models.tc_models.data import load_tc_data
from idd_climate_models.tc_models.features import build_X, align_X
from idd_climate_models.tc_models.metrics import calc_metrics
from idd_climate_models.tc_models.distributions import DISTRIBUTIONS, ML_MODELS

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS   = [123, 456, 789, 101112, 131415]
K_FOLDS = 5
THREADS = 8

# Full 2^4 factorial of covariates (no log transforms)
COVAR_SETS = [
    'none',
    'wind', 'sdi', 'basin', 'island',
    'wind_sdi', 'wind_basin', 'wind_island', 'sdi_basin', 'sdi_island', 'basin_island',
    'wind_sdi_basin', 'wind_sdi_island', 'wind_basin_island', 'sdi_basin_island',
    'wind_sdi_basin_island',
]

# Non-ML distributions per stage
S1_DISTS   = ['statsmodels_logistic']
S2_DISTS   = ['statsmodels_logistic']
BULK_DISTS = ['statsmodels_nb', 'statsmodels_gamma', 'statsmodels_lognormal']  # No pois/tweedie
TAIL_DISTS = ['scipy_gpd', 'statsmodels_nb', 'statsmodels_gamma', 'statsmodels_lognormal']

# Single threshold - 70% was best in TOPSIS analysis
THRESHOLDS = [70]

# Abbreviate dist names for display
_ABBREV = {
    'statsmodels_logistic': 'sm_logit',
    'statsmodels_nb':       'sm_nb',
    'statsmodels_poisson':  'sm_pois',
    'statsmodels_gamma':    'sm_gamma',
    'statsmodels_lognormal':'sm_lnorm',
    'statsmodels_tweedie':  'sm_tweedie',
    'scipy_gpd':            'scipy_gpd',
}
def abbrev(dist):
    return _ABBREV.get(dist, dist)


def _model_path(stage_id, seed, fold):
    return STAGE_MODELS_DIR / f'{stage_id}_seed{seed}_fold{fold}.pkl'


def _predict_stage(args):
    """Load a stage pkl and generate predictions on test subset."""
    stage_id, seed, fold, data, meta = args
    try:
        stype    = meta['stage_type']
        dist     = meta['dist']
        covars   = meta['covars']
        thr_pct  = meta.get('threshold_pct')

        pkl_path = _model_path(stage_id, seed, fold)
        if not pkl_path.exists():
            return (stage_id, seed, fold, None)

        with open(pkl_path, 'rb') as f:
            fitted = pickle.load(f)

        dist_mod = DISTRIBUTIONS[dist]
        is_ml    = dist in ML_MODELS

        rng      = np.random.default_rng(seed)
        fold_ids = rng.integers(0, K_FOLDS, size=len(data))
        train_idx = np.where(fold_ids != fold)[0]
        test_idx  = np.where(fold_ids == fold)[0]
        train = data.iloc[train_idx]
        test  = data.iloc[test_idx]

        if stype == 's1':
            train_sub = train
            test_sub  = test
            sub_idx   = test_idx
            task      = 'binary'
        elif stype == 'pos_binary':
            u = float(np.percentile(train[train['death_y_n'] == 1]['total_deaths'].values, thr_pct))
            train_sub = train[train['death_y_n'] == 1]
            test_sub  = test[test['death_y_n'] == 1]
            sub_idx   = test_idx[test['death_y_n'].values == 1]
            task      = 'binary'
        elif stype == 'dh_bulk':
            u = float(np.percentile(train[train['death_y_n'] == 1]['total_deaths'].values, thr_pct))
            train_sub = train[(train['death_y_n'] == 1) & (train['total_deaths'] <= u)]
            test_sub  = test[(test['death_y_n'] == 1) & (test['total_deaths'] <= u)]
            sub_idx   = test_idx[
                (test['death_y_n'].values == 1) &
                (test['total_deaths'].values <= u)
            ]
            task = 'count'
        elif stype == 'tail':
            u = float(np.percentile(train[train['death_y_n'] == 1]['total_deaths'].values, thr_pct))
            train_sub = train[(train['death_y_n'] == 1) & (train['total_deaths'] > u)]
            test_pos  = test[test['death_y_n'] == 1]
            test_sub  = test_pos[test_pos['total_deaths'] > u]
            sub_idx   = test_idx[
                (test['death_y_n'].values == 1) &
                (test['total_deaths'].values > u)
            ]
            task = 'count'
        else:
            return (stage_id, seed, fold, None)

        if len(test_sub) == 0:
            return (stage_id, seed, fold, None)

        X_tr = build_X(train_sub if len(train_sub) > 0 else train, covars, include_log_exp=False)
        X_te = align_X(build_X(test_sub, covars, include_log_exp=False), list(X_tr.columns))

        if task == 'count':
            exp_te = test_sub['exposed_population'].values
            preds  = dist_mod.predict(fitted, X_te, exposure=exp_te, task='count')
        else:
            preds  = dist_mod.predict(fitted, X_te, task='binary')

        preds = np.maximum(np.asarray(preds, dtype=float), 0)
        return (stage_id, seed, fold, pd.Series(preds, index=sub_idx))

    except Exception as e:
        import traceback
        return (stage_id, seed, fold, None, str(e))


def precompute_predictions(stage_ids, stage_meta_lookup, data):
    """Load predictions for all requested stages."""
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
            err = None
        else:
            sid, seed, fold, preds, err = result
        if preds is not None:
            cache[(sid, seed, fold)] = preds
            n_ok += 1
        else:
            n_fail += 1
            if err:
                errors.append(err)
    print(f"  Loaded {n_ok}, failed/missing {n_fail}")
    if errors:
        print(f"  Sample errors: {errors[:3]}")
    return cache


# ── In-sample prediction functions ─────────────────────────────────────────────

def _insample_path(stage_id):
    return STAGE_MODELS_DIR / f'{stage_id}_insample.pkl'


def _predict_insample_stage(args):
    """Load insample pkl and generate predictions on full data."""
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
        is_ml    = dist in ML_MODELS
        n        = len(data)

        if stype == 's1':
            X     = build_X(data, covars, include_log_exp=False)
            preds = dist_mod.predict(fitted, X, task='binary')
            preds = np.maximum(np.asarray(preds, dtype=float), 0)
            return (stage_id, pd.Series(preds, index=range(n)))

        elif stype == 'pos_binary':
            u = float(np.percentile(data[data['death_y_n']==1]['total_deaths'].values, thr_pct))
            pos_idx  = np.where(data['death_y_n'].values == 1)[0]
            pos_data = data.iloc[pos_idx]
            if len(pos_data) == 0:
                return (stage_id, None)
            X     = build_X(pos_data, covars, include_log_exp=False)
            preds = dist_mod.predict(fitted, X, task='binary')
            preds = np.maximum(np.asarray(preds, dtype=float), 0)
            return (stage_id, pd.Series(preds, index=pos_idx))

        elif stype == 'dh_bulk':
            u = float(np.percentile(data[data['death_y_n']==1]['total_deaths'].values, thr_pct))
            bulk_idx  = np.where((data['death_y_n'].values==1) & (data['total_deaths'].values<=u))[0]
            bulk_data = data.iloc[bulk_idx]
            if len(bulk_data) == 0:
                return (stage_id, None)
            X   = build_X(bulk_data, covars, include_log_exp=is_ml)
            exp = bulk_data['exposed_population'].values
            preds = dist_mod.predict(fitted, X, exposure=exp, task='count')
            preds = np.maximum(np.asarray(preds, dtype=float), 0)
            return (stage_id, pd.Series(preds, index=bulk_idx))

        elif stype == 'tail':
            u = float(np.percentile(data[data['death_y_n']==1]['total_deaths'].values, thr_pct))
            tail_idx  = np.where((data['death_y_n'].values==1) & (data['total_deaths'].values>u))[0]
            tail_data = data.iloc[tail_idx]
            if len(tail_data) == 0:
                return (stage_id, None)
            X   = build_X(tail_data, covars, include_log_exp=False)
            exp = tail_data['exposed_population'].values
            preds = dist_mod.predict(fitted, X, exposure=exp, task='count')
            preds = np.maximum(np.asarray(preds, dtype=float), 0)
            return (stage_id, pd.Series(preds, index=tail_idx))

        return (stage_id, None)

    except Exception as e:
        return (stage_id, None, str(e))


def precompute_insample_predictions(stage_ids, stage_meta_lookup, data):
    """Load insample predictions for all requested stages."""
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
        if len(result) == 2:
            sid, preds = result
        else:
            sid, preds, err = result
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
    """
    Convert prediction cache to aligned numpy arrays for fast assembly.
    
    Returns:
        obs_concat: concatenated observations across all folds
        exp_concat: concatenated exposures across all folds  
        stage_preds: dict stage_id -> concatenated predictions across all folds
    """
    # First pass: build fold structure and concatenate obs/exp
    obs_parts = []
    exp_parts = []
    fold_slices = {}  # (seed, fold) -> (start_idx, end_idx) in concatenated array
    
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
    
    # Second pass: concatenate predictions per stage
    # Group by stage_id first
    stage_folds = {}  # stage_id -> list of (seed, fold)
    for (sid, seed, fold), preds in cache.items():
        if preds is None:
            continue
        if sid not in stage_folds:
            stage_folds[sid] = []
        stage_folds[sid].append((seed, fold, preds))
    
    stage_preds = {}
    for sid, fold_data in stage_folds.items():
        # Check if we have all 25 folds
        if len(fold_data) != len(seeds) * k_folds:
            continue  # Skip incomplete stages
        
        # Sort by seed, fold to match obs_concat order
        fold_data.sort(key=lambda x: (seeds.index(x[0]), x[1]))
        
        parts = []
        for seed, fold, preds_series in fold_data:
            start, end, test_idx = fold_slices[(seed, fold)]
            # Align predictions to test_idx
            aligned = preds_series.reindex(test_idx).fillna(0.0).values
            parts.append(aligned)
        
        stage_preds[sid] = np.concatenate(parts)
    
    return obs_concat, exp_concat, stage_preds


def assemble_dh_fast(s1_sid, pb_sid, bulk_sid, tail_sid, obs, exp, stage_preds):
    """
    Fast DH assembly using pre-concatenated arrays.
    E[Y] = p_pos * ((1 - p_high) * e_bulk + p_high * e_tail)
    
    All inputs are already concatenated across all 25 folds.
    """
    if not all(sid in stage_preds for sid in [s1_sid, pb_sid, bulk_sid, tail_sid]):
        return None
    
    p_pos = stage_preds[s1_sid]
    p_high = stage_preds[pb_sid]
    e_bulk = stage_preds[bulk_sid]
    e_tail = stage_preds[tail_sid]
    
    # One vectorized operation across all ~7000 test observations
    pred = p_pos * ((1 - p_high) * e_bulk + p_high * e_tail)
    
    m = calc_metrics(obs, pred, exp)
    
    # Coverage metrics
    for frac, label in [(0.01, 'cov_1'), (0.05, 'cov_5'), (0.10, 'cov_10'), (0.20, 'cov_20')]:
        m[label] = _coverage_at(obs, pred, frac)
    
    m['pred_obs_ratio'] = pred.sum() / max(obs.sum(), 1)
    m['total_pred'] = pred.sum()
    m['total_obs'] = obs.sum()
    
    return m


def assemble_dh_insample(s1_sid, pb_sid, bulk_sid, tail_sid, is_cache, data):
    """
    Assemble in-sample DH predictions and compute IS metrics.
    E[Y] = p_pos * ((1 - p_high) * e_bulk + p_high * e_tail)
    """
    if not all(sid in is_cache for sid in [s1_sid, pb_sid, bulk_sid, tail_sid]):
        return None
    
    s1_p = is_cache[s1_sid]
    s2_p = is_cache[pb_sid]
    bk_p = is_cache[bulk_sid]
    tl_p = is_cache[tail_sid]
    
    n = len(data)
    p_pos  = s1_p.reindex(range(n)).fillna(0.5).values
    p_high = np.zeros(n)
    e_bulk = np.zeros(n)
    e_tail = np.zeros(n)
    
    for i in s2_p.index:
        p_high[i] = s2_p[i]
    for i in bk_p.index:
        e_bulk[i] = bk_p[i]
    for i in tl_p.index:
        e_tail[i] = tl_p[i]
    
    obs = data['total_deaths'].values
    exp = data['exposed_population'].values
    pred = np.maximum(p_pos * ((1 - p_high) * e_bulk + p_high * e_tail), 0)
    
    # Compute metrics (with _is suffix)
    ro = (obs / exp) * 1e5
    rp = (pred / exp) * 1e5
    nz = obs > 0
    
    return {
        'mae_rate_is': float(np.mean(np.abs(ro - rp))),
        'rmse_rate_is': float(np.sqrt(np.mean((ro - rp) ** 2))),
        'mae_count_is': float(np.mean(np.abs(obs - pred))),
        'rmse_count_is': float(np.sqrt(np.mean((obs - pred) ** 2))),
        'cor_rate_is': float(np.corrcoef(ro[nz], rp[nz])[0, 1]) if nz.sum() > 2 else np.nan,
        'cor_count_is': float(np.corrcoef(obs[nz], pred[nz])[0, 1]) if nz.sum() > 2 else np.nan,
        'pred_obs_ratio_is': float(np.sum(pred) / np.sum(obs)) if np.sum(obs) > 0 else np.nan,
    }


def main():
    print("="*72)
    print("EXHAUSTIVE DH ENUMERATION")
    print("="*72)
    print(f"Covariate sets: {COVAR_SETS}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"s1 dists: {S1_DISTS}")
    print(f"s2 dists: {S2_DISTS}")
    print(f"bulk dists: {BULK_DISTS}")
    print(f"tail dists: {TAIL_DISTS}")
    
    # Count expected combos
    n_dist_combos = len(S1_DISTS) * len(S2_DISTS) * len(BULK_DISTS) * len(TAIL_DISTS)
    n_covar_combos = len(COVAR_SETS) ** 4
    n_total = n_dist_combos * n_covar_combos * len(THRESHOLDS)
    print(f"\nExpected: {n_dist_combos} dist × {n_covar_combos} covar × {len(THRESHOLDS)} thr = {n_total} models")

    # Load manifest
    print("\nLoading stage manifest...")
    manifest = load_stage_manifest()
    oos = manifest[manifest['fit_type'] == 'oos'].copy()
    print(f"  {len(oos)} OOS rows")

    # Build stage lookup: (stage_type, dist, covars, threshold_pct) → stage_id
    # Use None for NaN thresholds to make lookups work
    stage_lookup = {}
    stage_meta = {}
    for _, row in oos.drop_duplicates('stage_id').iterrows():
        sid = row['stage_id']
        thr = row.get('threshold_pct')
        thr_key = None if pd.isna(thr) else float(thr)
        key = (row['stage_type'], row['dist'], row['covars'], thr_key)
        stage_lookup[key] = sid
        stage_meta[sid] = {
            'stage_type': row['stage_type'],
            'dist': row['dist'],
            'covars': row['covars'],
            'threshold_pct': thr_key,
        }

    # Collect all stage_ids we need
    needed_stages = set()
    for thr in THRESHOLDS:
        for s1_dist in S1_DISTS:
            for s1_cov in COVAR_SETS:
                key = ('s1', s1_dist, s1_cov, None)
                if key in stage_lookup:
                    needed_stages.add(stage_lookup[key])
        for s2_dist in S2_DISTS:
            for s2_cov in COVAR_SETS:
                key = ('pos_binary', s2_dist, s2_cov, float(thr))
                if key in stage_lookup:
                    needed_stages.add(stage_lookup[key])
        for bulk_dist in BULK_DISTS:
            for bulk_cov in COVAR_SETS:
                key = ('dh_bulk', bulk_dist, bulk_cov, float(thr))
                if key in stage_lookup:
                    needed_stages.add(stage_lookup[key])
        for tail_dist in TAIL_DISTS:
            for tail_cov in COVAR_SETS:
                key = ('tail', tail_dist, tail_cov, float(thr))
                if key in stage_lookup:
                    needed_stages.add(stage_lookup[key])

    print(f"\nUnique stages needed: {len(needed_stages)}")

    # Load data
    print("\nLoading TC data...")
    data = load_tc_data()
    print(f"  {len(data)} rows")

    # Precompute OOS predictions (cross-validated folds)
    print("\nPrecomputing OOS predictions...")
    cache = precompute_predictions(list(needed_stages), stage_meta, data)
    
    # Build aligned arrays for fast OOS assembly
    print("\nBuilding pre-concatenated OOS prediction arrays...")
    obs_concat, exp_concat, stage_preds = build_aligned_cache(cache, data, SEEDS, K_FOLDS)
    print(f"  {len(stage_preds)} stages with complete OOS predictions")
    print(f"  {len(obs_concat)} total test observations across all folds")

    # Precompute IS predictions (in-sample)
    print("\nPrecomputing IS predictions...")
    is_cache = precompute_insample_predictions(list(needed_stages), stage_meta, data)
    print(f"  {len(is_cache)} stages with IS predictions")

    # Enumerate all DH combinations
    n_expected = (len(THRESHOLDS) * len(S1_DISTS) * len(S2_DISTS) * 
                  len(BULK_DISTS) * len(TAIL_DISTS) * len(COVAR_SETS)**4)
    print(f"\nAssembling up to {n_expected:,} DH models (OOS + IS metrics)...")
    results = []
    n_done = 0
    n_skip = 0
    t_start = time.time()
    t_last = t_start

    for thr in THRESHOLDS:
        for s1_dist, s2_dist, bulk_dist, tail_dist in product(S1_DISTS, S2_DISTS, BULK_DISTS, TAIL_DISTS):
            for s1_cov, s2_cov, bulk_cov, tail_cov in product(COVAR_SETS, repeat=4):
                # Look up stage_ids
                s1_key   = ('s1', s1_dist, s1_cov, None)
                s2_key   = ('pos_binary', s2_dist, s2_cov, float(thr))
                bulk_key = ('dh_bulk', bulk_dist, bulk_cov, float(thr))
                tail_key = ('tail', tail_dist, tail_cov, float(thr))

                if not all(k in stage_lookup for k in [s1_key, s2_key, bulk_key, tail_key]):
                    n_skip += 1
                    continue

                s1_sid   = stage_lookup[s1_key]
                s2_sid   = stage_lookup[s2_key]
                bulk_sid = stage_lookup[bulk_key]
                tail_sid = stage_lookup[tail_key]

                # OOS metrics
                m_oos = assemble_dh_fast(s1_sid, s2_sid, bulk_sid, tail_sid, obs_concat, exp_concat, stage_preds)
                if m_oos is None:
                    n_skip += 1
                    continue

                # IS metrics
                m_is = assemble_dh_insample(s1_sid, s2_sid, bulk_sid, tail_sid, is_cache, data)
                
                row = {
                    'threshold_pct': thr,
                    's1_dist': s1_dist, 's1_cov': s1_cov, 's1_sid': s1_sid,
                    's2_dist': s2_dist, 's2_cov': s2_cov, 's2_sid': s2_sid,
                    'bulk_dist': bulk_dist, 'bulk_cov': bulk_cov, 'bulk_sid': bulk_sid,
                    'tail_dist': tail_dist, 'tail_cov': tail_cov, 'tail_sid': tail_sid,
                    **m_oos,
                }
                if m_is is not None:
                    row.update(m_is)
                
                results.append(row)
                n_done += 1
                if n_done % 10000 == 0:
                    t_now = time.time()
                    elapsed = t_now - t_last
                    total_elapsed = t_now - t_start
                    rate = n_done / total_elapsed
                    eta = (n_expected - n_done) / rate if rate > 0 else 0
                    print(f"At model {n_done:,} of {n_expected:,} | batch: {elapsed:.1f}s | total: {total_elapsed:.1f}s | ETA: {eta:.0f}s", flush=True)
                    t_last = t_now

    print(f"\nAssembled {n_done:,} models, skipped {n_skip:,}")

    if not results:
        print("No results!")
        return

    df = pd.DataFrame(results)

    # Save full results
    out_path = STAGE_RESULTS_DIR / 'dh_exhaustive_expanded.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved all {len(df):,} models to {out_path}")

    # Summary stats
    print("\n" + "="*72)
    print("SUMMARY")
    print("="*72)
    print(f"\nTotal models: {len(df)}")
    
    print(f"\nOOS pred/obs ratio distribution:")
    print(df['pred_obs_ratio'].describe())
    
    if 'pred_obs_ratio_is' in df.columns:
        print(f"\nIS pred/obs ratio distribution:")
        print(df['pred_obs_ratio_is'].describe())
        
        print(f"\nIS/OOS MAE gap (mae_rate - mae_rate_is):")
        gap = df['mae_rate'] - df['mae_rate_is']
        print(f"  Mean: {gap.mean():.4f}")
        print(f"  Std:  {gap.std():.4f}")
        print(f"  Min:  {gap.min():.4f}")
        print(f"  Max:  {gap.max():.4f}")

    print(f"\nModels with pred_obs_ratio in [0.5, 2.0]: {((df['pred_obs_ratio'] >= 0.5) & (df['pred_obs_ratio'] <= 2.0)).sum()}")
    print(f"Models with pred_obs_ratio in [0.8, 1.2]: {((df['pred_obs_ratio'] >= 0.8) & (df['pred_obs_ratio'] <= 1.2)).sum()}")

    # Best by mae_rate
    print("\n── Top 20 by mae_rate ──")
    cols = ['threshold_pct', 's1_cov', 's2_cov', 'bulk_dist', 'bulk_cov', 'tail_dist', 'tail_cov',
            'mae_rate', 'pred_obs_ratio', 'cov_10']
    print(df.nsmallest(20, 'mae_rate')[cols].to_string(index=False))

    # Best by pred_obs_ratio closeness to 1
    df['calib_err'] = (df['pred_obs_ratio'] - 1).abs()
    print("\n── Top 20 by calibration (pred_obs ≈ 1) ──")
    print(df.nsmallest(20, 'calib_err')[cols + ['calib_err']].to_string(index=False))


if __name__ == '__main__':
    main()
