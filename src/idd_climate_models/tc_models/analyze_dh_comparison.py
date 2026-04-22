"""
analyze_dh_comparison.py

Exhaustive DH model comparison using stage_results_dh output.
Only includes stages with ALL 26 folds complete (0 failures).

Usage:
    python src/idd_climate_models/tc_models/analyze_dh_comparison.py
"""

import json
import pickle
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path('/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/stage_results_dh')
MODELS_DIR  = OUTPUT_DIR / 'models'
GRID_PATH   = OUTPUT_DIR / 'stage_grid.json'

SEEDS   = [123, 456, 789, 101112, 131415]
K_FOLDS = 5

# ── Stage ID ──────────────────────────────────────────────────────────────────
def stage_to_id(stage: dict) -> str:
    import hashlib
    key = json.dumps(stage, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def is_complete(stage_id: str) -> bool:
    """True iff insample + all 25 OOS pkl files exist."""
    if not (MODELS_DIR / f'{stage_id}_insample.pkl').exists():
        return False
    for seed in SEEDS:
        for fold in range(K_FOLDS):
            if not (MODELS_DIR / f'{stage_id}_seed{seed}_fold{fold}.pkl').exists():
                return False
    return True


# ── Load data ─────────────────────────────────────────────────────────────────
def load_tc_data():
    from idd_climate_models.tc_models.data import load_tc_data as _load
    return _load()


def load_grid():
    with open(GRID_PATH) as f:
        return json.load(f)


# ── Build stage lookup ────────────────────────────────────────────────────────
def build_stage_lookup(grid):
    """
    Returns:
        lookup: (stage_type, dist, covars, threshold_pct) -> stage_id
        meta:   stage_id -> stage dict
        complete_ids: set of stage_ids that are 100% complete
    """
    lookup = {}
    meta = {}
    complete_ids = set()
    
    for stage in grid:
        sid = stage_to_id(stage)
        thr = stage.get('threshold_pct')
        key = (stage['stage_type'], stage['dist'], stage['covars'], thr)
        lookup[key] = sid
        meta[sid] = stage
        
        if is_complete(sid):
            complete_ids.add(sid)
    
    return lookup, meta, complete_ids


# ── Prediction cache ──────────────────────────────────────────────────────────
def load_insample_predictions(stage_id, meta, data):
    """
    Load IN-SAMPLE predictions for a stage (fit on all data, predict on all data).
    Returns: pd.Series indexed by row positions for the relevant subset.
    """
    from idd_climate_models.tc_models.features import build_X, align_X
    from idd_climate_models.tc_models.distributions import DISTRIBUTIONS
    
    stage = meta[stage_id]
    stype = stage['stage_type']
    dist = stage['dist']
    covars = stage['covars']
    thr_pct = stage.get('threshold_pct')
    dist_mod = DISTRIBUTIONS[dist]
    
    pkl_path = MODELS_DIR / f'{stage_id}_insample.pkl'
    if not pkl_path.exists():
        return None
    
    with open(pkl_path, 'rb') as f:
        fitted = pickle.load(f)
    
    if stype == 's1':
        X = build_X(data, covars, include_log_exp=False)
        pred_vals = dist_mod.predict(fitted, X, task='binary')
        return pd.Series(pred_vals, index=range(len(data)))
        
    elif stype == 'pos_binary':
        # Threshold from full data
        u = float(np.percentile(data[data['death_y_n'] == 1]['total_deaths'].values, thr_pct))
        subset = data[data['death_y_n'] == 1]
        sub_idx = np.where(data['death_y_n'].values == 1)[0]
        X = build_X(subset, covars, include_log_exp=False)
        pred_vals = dist_mod.predict(fitted, X, task='binary')
        return pd.Series(pred_vals, index=sub_idx)
        
    elif stype == 'dh_bulk':
        u = float(np.percentile(data[data['death_y_n'] == 1]['total_deaths'].values, thr_pct))
        mask = (data['death_y_n'].values == 1) & (data['total_deaths'].values <= u)
        subset = data[mask]
        sub_idx = np.where(mask)[0]
        X = build_X(subset, covars, include_log_exp=False)
        exp = subset['exposed_population'].values
        pred_vals = dist_mod.predict(fitted, X, exposure=exp, task='count')
        return pd.Series(pred_vals, index=sub_idx)
        
    elif stype == 'tail':
        u = float(np.percentile(data[data['death_y_n'] == 1]['total_deaths'].values, thr_pct))
        mask = (data['death_y_n'].values == 1) & (data['total_deaths'].values > u)
        subset = data[mask]
        sub_idx = np.where(mask)[0]
        X = build_X(subset, covars, include_log_exp=False)
        exp = subset['exposed_population'].values
        pred_vals = dist_mod.predict(fitted, X, exposure=exp)
        return pd.Series(pred_vals, index=sub_idx)
    
    return None


def load_predictions(stage_id, meta, data):
    """
    Load OOS predictions for a complete stage.
    Returns dict: (seed, fold) -> pd.Series indexed by test row positions.
    """
    from idd_climate_models.tc_models.features import build_X, align_X
    from idd_climate_models.tc_models.distributions import DISTRIBUTIONS
    
    stage = meta[stage_id]
    stype = stage['stage_type']
    dist = stage['dist']
    covars = stage['covars']
    thr_pct = stage.get('threshold_pct')
    dist_mod = DISTRIBUTIONS[dist]
    
    preds = {}
    
    for seed in SEEDS:
        for fold in range(K_FOLDS):
            pkl_path = MODELS_DIR / f'{stage_id}_seed{seed}_fold{fold}.pkl'
            with open(pkl_path, 'rb') as f:
                fitted = pickle.load(f)
            
            rng = np.random.default_rng(seed)
            fold_ids = rng.integers(0, K_FOLDS, size=len(data))
            train_idx = np.where(fold_ids != fold)[0]
            test_idx = np.where(fold_ids == fold)[0]
            train = data.iloc[train_idx]
            test = data.iloc[test_idx]
            
            if stype == 's1':
                test_sub = test
                sub_idx = test_idx
                X_tr = build_X(train, covars, include_log_exp=False)
                X_te = align_X(build_X(test_sub, covars, include_log_exp=False), list(X_tr.columns))
                pred_vals = dist_mod.predict(fitted, X_te, task='binary')
                
            elif stype == 'pos_binary':
                u = float(np.percentile(train[train['death_y_n'] == 1]['total_deaths'].values, thr_pct))
                test_sub = test[test['death_y_n'] == 1]
                sub_idx = test_idx[test['death_y_n'].values == 1]
                X_tr = build_X(train[train['death_y_n'] == 1], covars, include_log_exp=False)
                X_te = align_X(build_X(test_sub, covars, include_log_exp=False), list(X_tr.columns))
                pred_vals = dist_mod.predict(fitted, X_te, task='binary')
                
            elif stype == 'dh_bulk':
                u = float(np.percentile(train[train['death_y_n'] == 1]['total_deaths'].values, thr_pct))
                test_sub = test[(test['death_y_n'] == 1) & (test['total_deaths'] <= u)]
                sub_idx = test_idx[(test['death_y_n'].values == 1) & (test['total_deaths'].values <= u)]
                train_bulk = train[(train['death_y_n'] == 1) & (train['total_deaths'] <= u)]
                X_tr = build_X(train_bulk, covars, include_log_exp=False)
                X_te = align_X(build_X(test_sub, covars, include_log_exp=False), list(X_tr.columns))
                exp_te = test_sub['exposed_population'].values
                pred_vals = dist_mod.predict(fitted, X_te, exposure=exp_te, task='count')
                
            elif stype == 'tail':
                u = float(np.percentile(train[train['death_y_n'] == 1]['total_deaths'].values, thr_pct))
                test_sub = test[(test['death_y_n'] == 1) & (test['total_deaths'] > u)]
                sub_idx = test_idx[(test['death_y_n'].values == 1) & (test['total_deaths'].values > u)]
                train_tail = train[(train['death_y_n'] == 1) & (train['total_deaths'] > u)]
                X_tr = build_X(train_tail, covars, include_log_exp=False)
                X_te = align_X(build_X(test_sub, covars, include_log_exp=False), list(X_tr.columns))
                exp_te = test_sub['exposed_population'].values
                pred_vals = dist_mod.predict(fitted, X_te, exposure=exp_te)
            
            else:
                continue
            
            if len(test_sub) > 0:
                pred_vals = np.maximum(np.asarray(pred_vals, dtype=float), 0)
                preds[(seed, fold)] = pd.Series(pred_vals, index=sub_idx)
    
    return preds


# ── Assemble DH ───────────────────────────────────────────────────────────────
def assemble_dh(s1_preds, s2_preds, bulk_preds, tail_preds, data):
    """
    Compute OOS predictions for a DH model.
    E[Y] = p_pos * ((1 - p_high) * e_bulk + p_high * e_tail)
    """
    obs_all = []
    pred_all = []
    exp_all = []
    
    for seed in SEEDS:
        for fold in range(K_FOLDS):
            key = (seed, fold)
            if key not in s1_preds or key not in s2_preds:
                continue
            if key not in bulk_preds or key not in tail_preds:
                continue
            
            rng = np.random.default_rng(seed)
            fold_ids = rng.integers(0, K_FOLDS, size=len(data))
            test_idx = np.where(fold_ids == fold)[0]
            n_test = len(test_idx)
            
            p_pos = s1_preds[key].reindex(test_idx).fillna(0.5).values
            p_high = np.zeros(n_test)
            e_bulk = np.zeros(n_test)
            e_tail = np.zeros(n_test)
            
            for i, gi in enumerate(test_idx):
                if gi in s2_preds[key].index:
                    p_high[i] = s2_preds[key][gi]
                if gi in bulk_preds[key].index:
                    e_bulk[i] = bulk_preds[key][gi]
                if gi in tail_preds[key].index:
                    e_tail[i] = tail_preds[key][gi]
            
            preds = p_pos * ((1 - p_high) * e_bulk + p_high * e_tail)
            obs = data.iloc[test_idx]['total_deaths'].values
            exp = data.iloc[test_idx]['exposed_population'].values
            
            obs_all.append(obs)
            pred_all.append(preds)
            exp_all.append(exp)
    
    if not obs_all:
        return None
    
    obs_c = np.concatenate(obs_all)
    pred_c = np.concatenate(pred_all)
    exp_c = np.concatenate(exp_all)
    
    # Compute metrics
    from idd_climate_models.tc_models.metrics import calc_metrics
    m = calc_metrics(obs_c, pred_c, exp_c)
    
    # Add calibration
    m['pred_obs_ratio'] = pred_c.sum() / max(obs_c.sum(), 1)
    m['total_pred'] = pred_c.sum()
    m['total_obs'] = obs_c.sum()
    
    # Add coverage
    for frac, label in [(0.05, 'cov_5'), (0.10, 'cov_10'), (0.20, 'cov_20')]:
        k = max(1, int(round(len(obs_c) * frac)))
        top_obs = set(np.argsort(obs_c)[-k:])
        top_pred = set(np.argsort(pred_c)[-k:])
        m[label] = len(top_obs & top_pred) / k
    
    return m


# ── Assemble DH IN-SAMPLE ─────────────────────────────────────────────────────
def assemble_dh_insample(s1_preds, s2_preds, bulk_preds, tail_preds, data):
    """
    Compute IN-SAMPLE predictions for a DH model.
    E[Y] = p_pos * ((1 - p_high) * e_bulk + p_high * e_tail)
    """
    if s1_preds is None or s2_preds is None or bulk_preds is None or tail_preds is None:
        return None
    
    n = len(data)
    
    # s1 predicts for all rows
    p_pos = s1_preds.reindex(range(n)).fillna(0.5).values
    
    # s2, bulk, tail only predict for subsets - default to 0 where missing
    p_high = np.zeros(n)
    e_bulk = np.zeros(n)
    e_tail = np.zeros(n)
    
    for gi in range(n):
        if gi in s2_preds.index:
            p_high[gi] = s2_preds[gi]
        if gi in bulk_preds.index:
            e_bulk[gi] = bulk_preds[gi]
        if gi in tail_preds.index:
            e_tail[gi] = tail_preds[gi]
    
    pred_c = p_pos * ((1 - p_high) * e_bulk + p_high * e_tail)
    pred_c = np.maximum(pred_c, 0)
    obs_c = data['total_deaths'].values
    exp_c = data['exposed_population'].values
    
    # Compute metrics
    from idd_climate_models.tc_models.metrics import calc_metrics
    m = calc_metrics(obs_c, pred_c, exp_c)
    
    # Add calibration
    m['pred_obs_ratio'] = pred_c.sum() / max(obs_c.sum(), 1)
    m['total_pred'] = pred_c.sum()
    m['total_obs'] = obs_c.sum()
    
    # Add coverage
    for frac, label in [(0.05, 'cov_5'), (0.10, 'cov_10'), (0.20, 'cov_20')]:
        k = max(1, int(round(len(obs_c) * frac)))
        top_obs = set(np.argsort(obs_c)[-k:])
        top_pred = set(np.argsort(pred_c)[-k:])
        m[label] = len(top_obs & top_pred) / k
    
    return m


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parents[3]))
    
    print("="*72)
    print("DH MODEL COMPARISON")
    print("="*72)
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Only including stages with ALL 26 folds complete (0 failures)")
    
    # Load grid
    print("\nLoading grid...")
    grid = load_grid()
    print(f"  Total stages in grid: {len(grid)}")
    
    # Build lookup
    lookup, meta, complete_ids = build_stage_lookup(grid)
    print(f"  Complete stages: {len(complete_ids)}")
    print(f"  Incomplete/failed: {len(grid) - len(complete_ids)}")
    
    # Load data
    print("\nLoading TC data...")
    data = load_tc_data()
    print(f"  {len(data)} rows")
    
    # Enumerate combinations
    COVAR_SETS = ['none', 'wind_sdi', 'wind_sdi_basin_island']
    THRESHOLDS = [70, 75, 80, 85, 90, 95]
    S1_DISTS = ['statsmodels_logistic']
    S2_DISTS = ['statsmodels_logistic']
    BULK_DISTS = ['statsmodels_nb', 'statsmodels_gamma', 'statsmodels_lognormal',
                  'statsmodels_poisson', 'statsmodels_tweedie']
    TAIL_DISTS = ['scipy_gpd', 'statsmodels_nb', 'statsmodels_gamma',
                  'statsmodels_lognormal', 'statsmodels_poisson']
    
    # Preload all complete stage predictions (OOS and IS)
    print("\nLoading predictions for complete stages...")
    pred_cache = {}
    is_pred_cache = {}
    for sid in complete_ids:
        pred_cache[sid] = load_predictions(sid, meta, data)
        is_pred_cache[sid] = load_insample_predictions(sid, meta, data)
    print(f"  Loaded {len(pred_cache)} stages (OOS + IS)")
    
    # Enumerate DH combinations
    print("\nAssembling DH models...")
    results = []
    n_skip_incomplete = 0
    n_done = 0
    
    for thr in THRESHOLDS:
        for s1_dist, s2_dist, bulk_dist, tail_dist in product(S1_DISTS, S2_DISTS, BULK_DISTS, TAIL_DISTS):
            for s1_cov, s2_cov, bulk_cov, tail_cov in product(COVAR_SETS, repeat=4):
                # Look up stage IDs
                s1_key = ('s1', s1_dist, s1_cov, None)
                s2_key = ('pos_binary', s2_dist, s2_cov, thr)
                bulk_key = ('dh_bulk', bulk_dist, bulk_cov, thr)
                tail_key = ('tail', tail_dist, tail_cov, thr)
                
                # Check all exist in lookup
                if not all(k in lookup for k in [s1_key, s2_key, bulk_key, tail_key]):
                    n_skip_incomplete += 1
                    continue
                
                s1_sid = lookup[s1_key]
                s2_sid = lookup[s2_key]
                bulk_sid = lookup[bulk_key]
                tail_sid = lookup[tail_key]
                
                # Check all complete
                if not all(sid in complete_ids for sid in [s1_sid, s2_sid, bulk_sid, tail_sid]):
                    n_skip_incomplete += 1
                    continue
                
                # Assemble OOS
                m_oos = assemble_dh(
                    pred_cache[s1_sid],
                    pred_cache[s2_sid],
                    pred_cache[bulk_sid],
                    pred_cache[tail_sid],
                    data
                )
                
                # Assemble IS
                m_is = assemble_dh_insample(
                    is_pred_cache[s1_sid],
                    is_pred_cache[s2_sid],
                    is_pred_cache[bulk_sid],
                    is_pred_cache[tail_sid],
                    data
                )
                
                if m_oos is None:
                    n_skip_incomplete += 1
                    continue
                
                row = {
                    'threshold_pct': thr,
                    's1_dist': s1_dist, 's1_cov': s1_cov,
                    's2_dist': s2_dist, 's2_cov': s2_cov,
                    'bulk_dist': bulk_dist, 'bulk_cov': bulk_cov,
                    'tail_dist': tail_dist, 'tail_cov': tail_cov,
                }
                # Add OOS metrics with prefix
                for k, v in m_oos.items():
                    row[f'oos_{k}'] = v
                # Add IS metrics with prefix (if available)
                if m_is is not None:
                    for k, v in m_is.items():
                        row[f'is_{k}'] = v
                
                results.append(row)
                n_done += 1
                
                if n_done % 500 == 0:
                    print(f"  {n_done} models assembled...")
    
    print(f"\nAssembled {n_done} models")
    print(f"Skipped {n_skip_incomplete} (incomplete stages)")
    
    if not results:
        print("No results!")
        return
    
    df = pd.DataFrame(results)
    
    # Save
    out_path = OUTPUT_DIR / 'dh_comparison_results.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    
    # Summary
    print("\n" + "="*72)
    print("SUMMARY")
    print("="*72)
    print(f"\nTotal models: {len(df)}")
    
    print(f"\nOOS Pred/obs ratio distribution:")
    print(df['oos_pred_obs_ratio'].describe())
    
    if 'is_pred_obs_ratio' in df.columns:
        print(f"\nIS Pred/obs ratio distribution:")
        print(df['is_pred_obs_ratio'].describe())
    
    calibrated = df[(df['oos_pred_obs_ratio'] >= 0.5) & (df['oos_pred_obs_ratio'] <= 2.0)]
    print(f"\nModels with OOS pred_obs_ratio in [0.5, 2.0]: {len(calibrated)} ({100*len(calibrated)/len(df):.1f}%)")
    
    print("\n── Top 20 by OOS mae_rate ──")
    cols = ['threshold_pct', 'bulk_dist', 'bulk_cov', 'tail_dist', 'tail_cov',
            'is_mae_rate', 'oos_mae_rate', 'is_pred_obs_ratio', 'oos_pred_obs_ratio', 'oos_cov_10']
    cols = [c for c in cols if c in df.columns]
    print(df.nsmallest(20, 'oos_mae_rate')[cols].to_string(index=False))
    
    print("\n── Top 20 by OOS calibration (pred_obs ≈ 1) ──")
    df['calib_err'] = (df['oos_pred_obs_ratio'] - 1).abs()
    print(df.nsmallest(20, 'calib_err')[cols].to_string(index=False))


if __name__ == '__main__':
    main()
