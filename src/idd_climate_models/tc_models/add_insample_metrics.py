"""
add_insample_metrics.py

Add in-sample (IS) metrics to the exhaustive DH model results.

================================================================================
PIPELINE OVERVIEW — How we got here:
================================================================================

1. STAGE FITTING (workflow 559214):
   - Script: orchestrate_dh_expanded.py calls run_one_stage.py via jobmon
   - Input: stage_grid_dh_expanded.py defines 144 stages:
       * 16 covariate sets (2^4 factorial: wind, sdi, basin, island)
       * 1 threshold (70%)
       * Distributions: s1/s2=logistic, bulk=NB/gamma/lognormal, tail=GPD/NB/gamma/lognormal
   - Output: /mnt/team/rapidresponse/pub/tropical-storms/direct_risk/stage_results_dh/models/
       * {stage_id}_insample.pkl — fitted model on full data
       * {stage_id}_seed{N}_fold{K}.pkl — fitted model on training fold
   - Result: 129/144 stages complete (15 gamma failures due to numerical instability)

2. OOS METRIC COMPUTATION (analyze_dh_exhaustive.py):
   - Loads OOS fold predictions for all stages
   - Assembles all 16^4 × 12 = 786,432 model combinations
   - Computes OOS metrics: mae_rate, rmse_rate, cor_rate, coverage, pred_obs_ratio
   - Output: dh_exhaustive_expanded.csv (587,776 models after excluding failed stages)

3. THIS SCRIPT (add_insample_metrics.py):
   - Loads insample pkl files ({stage_id}_insample.pkl)
   - Computes IS predictions for each unique stage (once per stage)
   - Assembles IS predictions for each model row in exhaustive results
   - Computes IS metrics: mae_rate_is, rmse_rate_is, cor_rate_is, pred_obs_ratio_is
   - Joins IS metrics to existing results
   - Output: dh_exhaustive_with_is.csv

4. NEXT STEP — MODEL SELECTION (dh_model_selection_topsis.ipynb):
   - Loads dh_exhaustive_with_is.csv
   - Filters by calibration (pred_obs_ratio) and IS/OOS gap
   - Runs TOPSIS multi-criteria ranking
   - Selects final model

================================================================================
USAGE:
================================================================================

    python src/idd_climate_models/tc_models/add_insample_metrics.py

Expected runtime: ~5 minutes (129 stage predictions + 587k model assemblies)

================================================================================
"""

import sys
import pickle
import warnings
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Set paths to stage_results_dh BEFORE importing from cache/constants
_DH_ROOT = Path('/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/stage_results_dh')
os.environ['STAGE_MODELS_DIR'] = str(_DH_ROOT / 'models')
os.environ['STAGE_RESULTS_DIR'] = str(_DH_ROOT / 'results')

sys.path.insert(0, str(Path(__file__).parents[3]))

from idd_climate_models.tc_models.cache import STAGE_MODELS_DIR
from idd_climate_models.tc_models.constants import STAGE_RESULTS_DIR
from idd_climate_models.tc_models.data import load_tc_data
from idd_climate_models.tc_models.features import build_X
from idd_climate_models.tc_models.distributions import DISTRIBUTIONS, ML_MODELS

warnings.filterwarnings('ignore')


def load_insample_pkl(stage_id):
    """Load insample fitted model."""
    path = STAGE_MODELS_DIR / f'{stage_id}_insample.pkl'
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def predict_insample_stage(stage_id, stage_meta, data, is_pred_cache):
    """
    Compute insample predictions for a stage.
    Returns Series indexed by data row indices.
    """
    if stage_id in is_pred_cache:
        return is_pred_cache[stage_id]
    
    fitted = load_insample_pkl(stage_id)
    if fitted is None:
        is_pred_cache[stage_id] = None
        return None
    
    meta = stage_meta.get(stage_id)
    if meta is None:
        is_pred_cache[stage_id] = None
        return None
    
    dist = meta['dist']
    covars = meta['covars']
    stype = meta['stage_type']
    thr_pct = meta.get('threshold_pct')
    is_ml = dist in ML_MODELS
    dm = DISTRIBUTIONS[dist]
    
    result = None
    try:
        if stype == 's1':
            X = build_X(data, covars, include_log_exp=False)
            p = dm.predict(fitted, X, task='binary')
            result = pd.Series(np.maximum(p, 0).astype(float), index=range(len(data)))
            
        elif stype == 'pos_binary':
            pos_idx = np.where(data['death_y_n'].values == 1)[0]
            pos_data = data.iloc[pos_idx]
            if len(pos_data) > 0:
                X = build_X(pos_data, covars, include_log_exp=False)
                p = dm.predict(fitted, X, task='binary')
                result = pd.Series(np.maximum(p, 0).astype(float), index=pos_idx)
                
        elif stype == 'dh_bulk':
            u = float(np.percentile(data[data['death_y_n']==1]['total_deaths'].values, thr_pct))
            bulk_idx = np.where((data['death_y_n'].values==1) & (data['total_deaths'].values<=u))[0]
            bulk_data = data.iloc[bulk_idx]
            if len(bulk_data) > 0:
                X = build_X(bulk_data, covars, include_log_exp=is_ml)
                p = dm.predict(fitted, X, exposure=bulk_data['exposed_population'].values, task='count')
                result = pd.Series(np.maximum(p, 0).astype(float), index=bulk_idx)
                
        elif stype == 'tail':
            u = float(np.percentile(data[data['death_y_n']==1]['total_deaths'].values, thr_pct))
            tail_idx = np.where((data['death_y_n'].values==1) & (data['total_deaths'].values>u))[0]
            tail_data = data.iloc[tail_idx]
            if len(tail_data) > 0:
                X = build_X(tail_data, covars, include_log_exp=False)
                p = dm.predict(fitted, X, exposure=tail_data['exposed_population'].values)
                result = pd.Series(np.maximum(p, 0).astype(float), index=tail_idx)
    except Exception as e:
        print(f"  Warning: prediction failed for {stage_id}: {e}")
        result = None
    
    is_pred_cache[stage_id] = result
    return result


def assemble_dh_insample(s1_id, s2_id, bulk_id, tail_id, is_pred_cache, data):
    """
    Assemble insample predictions for a DH model.
    Returns dict of IS metrics or None if any stage failed.
    """
    s1_p = is_pred_cache.get(s1_id)
    s2_p = is_pred_cache.get(s2_id)
    bk_p = is_pred_cache.get(bulk_id)
    tl_p = is_pred_cache.get(tail_id)
    
    if any(x is None for x in [s1_p, s2_p, bk_p, tl_p]):
        return None
    
    n = len(data)
    p_pos = s1_p.reindex(range(n)).fillna(0.5).values
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
    
    # Compute metrics
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
        'mae_log_is': float(np.mean(np.abs(np.log1p(obs) - np.log1p(pred)))),
        'pred_obs_ratio_is': float(np.sum(pred) / np.sum(obs)) if np.sum(obs) > 0 else np.nan,
    }


def main():
    print("=" * 72)
    print("ADD INSAMPLE METRICS TO EXHAUSTIVE DH RESULTS")
    print("=" * 72)
    
    # ── Load existing OOS results ──────────────────────────────────────────────
    oos_path = STAGE_RESULTS_DIR / 'dh_exhaustive_expanded.csv'
    print(f"\nLoading OOS results from {oos_path}...")
    df = pd.read_csv(oos_path)
    print(f"  {len(df):,} model rows")
    
    # ── Load TC data ───────────────────────────────────────────────────────────
    print("\nLoading TC data...")
    data = load_tc_data()
    print(f"  {len(data)} rows")
    
    # ── Build stage metadata lookup ────────────────────────────────────────────
    # Extract unique stages from the results
    stage_ids = set()
    for col in ['s1_sid', 's2_sid', 'bulk_sid', 'tail_sid']:
        stage_ids.update(df[col].unique())
    print(f"\nUnique stages to predict: {len(stage_ids)}")
    
    # Build metadata from the dataframe itself
    stage_meta = {}
    
    # s1 stages
    s1_rows = df[['s1_sid', 's1_dist', 's1_cov']].drop_duplicates()
    for _, r in s1_rows.iterrows():
        stage_meta[r['s1_sid']] = {
            'stage_type': 's1',
            'dist': r['s1_dist'],
            'covars': r['s1_cov'],
            'threshold_pct': None,
        }
    
    # s2 stages
    s2_rows = df[['s2_sid', 's2_dist', 's2_cov', 'threshold_pct']].drop_duplicates()
    for _, r in s2_rows.iterrows():
        stage_meta[r['s2_sid']] = {
            'stage_type': 'pos_binary',
            'dist': r['s2_dist'],
            'covars': r['s2_cov'],
            'threshold_pct': r['threshold_pct'],
        }
    
    # bulk stages
    bulk_rows = df[['bulk_sid', 'bulk_dist', 'bulk_cov', 'threshold_pct']].drop_duplicates()
    for _, r in bulk_rows.iterrows():
        stage_meta[r['bulk_sid']] = {
            'stage_type': 'dh_bulk',
            'dist': r['bulk_dist'],
            'covars': r['bulk_cov'],
            'threshold_pct': r['threshold_pct'],
        }
    
    # tail stages
    tail_rows = df[['tail_sid', 'tail_dist', 'tail_cov', 'threshold_pct']].drop_duplicates()
    for _, r in tail_rows.iterrows():
        stage_meta[r['tail_sid']] = {
            'stage_type': 'tail',
            'dist': r['tail_dist'],
            'covars': r['tail_cov'],
            'threshold_pct': r['threshold_pct'],
        }
    
    print(f"  Built metadata for {len(stage_meta)} stages")
    
    # ── Precompute insample predictions for all stages ─────────────────────────
    print("\nPrecomputing insample predictions...")
    is_pred_cache = {}
    t0 = time.time()
    
    for i, sid in enumerate(sorted(stage_ids)):
        predict_insample_stage(sid, stage_meta, data, is_pred_cache)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1} / {len(stage_ids)} stages predicted...")
    
    n_ok = sum(1 for v in is_pred_cache.values() if v is not None)
    print(f"  Predicted {n_ok} / {len(stage_ids)} stages in {time.time() - t0:.1f}s")
    
    # ── Compute IS metrics for each model row ──────────────────────────────────
    print(f"\nComputing IS metrics for {len(df):,} models...")
    t0 = time.time()
    
    is_metrics_list = []
    n_ok = 0
    
    for i, (_, row) in enumerate(df.iterrows()):
        m = assemble_dh_insample(
            row['s1_sid'], row['s2_sid'], row['bulk_sid'], row['tail_sid'],
            is_pred_cache, data
        )
        is_metrics_list.append(m if m else {})
        if m:
            n_ok += 1
        
        if (i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(df) - i - 1) / rate
            print(f"  At model {i + 1:,} of {len(df):,} | {elapsed:.1f}s elapsed | ETA: {eta:.0f}s")
    
    print(f"\nComputed IS metrics for {n_ok:,} / {len(df):,} models in {time.time() - t0:.1f}s")
    
    # ── Join IS metrics to dataframe ───────────────────────────────────────────
    is_df = pd.DataFrame(is_metrics_list)
    df_combined = pd.concat([df.reset_index(drop=True), is_df.reset_index(drop=True)], axis=1)
    
    # ── Save results ───────────────────────────────────────────────────────────
    out_path = STAGE_RESULTS_DIR / 'dh_exhaustive_with_is.csv'
    df_combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_combined):,} models to {out_path}")
    
    # ── Summary stats ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    
    if 'mae_rate_is' in df_combined.columns:
        print("\nIS/OOS metric comparison (mean):")
        for metric in ['mae_rate', 'rmse_rate', 'cor_rate', 'pred_obs_ratio']:
            oos_col = metric
            is_col = f'{metric}_is'
            if oos_col in df_combined.columns and is_col in df_combined.columns:
                oos_mean = df_combined[oos_col].mean()
                is_mean = df_combined[is_col].mean()
                gap = abs(oos_mean - is_mean) / oos_mean * 100 if oos_mean != 0 else np.nan
                print(f"  {metric:20s}: OOS={oos_mean:.4f}, IS={is_mean:.4f}, gap={gap:.1f}%")
    
    print(f"\nOutput file: {out_path}")
    print("Next step: Run dh_model_selection_topsis.ipynb with this file")


if __name__ == '__main__':
    main()
