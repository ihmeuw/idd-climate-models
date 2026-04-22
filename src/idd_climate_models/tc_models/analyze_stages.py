"""
analyze_stages.py

Two-phase analysis:
  Phase 1 — stage-level OOS metric summary, best component per stage type
  Phase 2 — assemble top-K stages into full models (single/hurdle/POT/DH),
             compute full OOS mae_rate, rank top 50

Usage:
    python src/idd_climate_models/tc_models/analyze_stages.py

Outputs printed to stdout; CSV written to STAGE_RESULTS_DIR/top_models.csv
"""

import sys
import pickle
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[4] / 'src'))

from idd_climate_models.tc_models.cache import load_stage_manifest, STAGE_MODELS_DIR
from idd_climate_models.tc_models.data import load_tc_data
from idd_climate_models.tc_models.features import build_X, align_X
from idd_climate_models.tc_models.metrics import calc_metrics
from idd_climate_models.tc_models.distributions import DISTRIBUTIONS, ML_MODELS

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS   = [123, 456, 789, 101112, 131415]
K_FOLDS = 5
TOP_K   = 5   # top stages per type fed into assembly (per threshold bucket for threshold-dependent types)
THREADS = 8

# Abbreviate dist names for display
_ABBREV = {
    'statsmodels_logistic':        'sm_logit',
    'statsmodels_nb':              'sm_nb',
    'statsmodels_poisson':         'sm_pois',
    'statsmodels_zinb':            'sm_zinb',
    'statsmodels_gamma':           'sm_gamma',
    'statsmodels_lognormal':       'sm_lnorm',
    'statsmodels_gpd':             'sm_gpd',
    'sklearn_lognormal':           'sk_lnorm',
    'sklearn_rf':                  'sk_rf',
    'sklearn_gb':                  'sk_gb',
}
def abbrev(dist):
    return _ABBREV.get(dist, dist)


# ── Phase 1: stage-level summary ─────────────────────────────────────────────

def load_oos(manifest):
    """OOS rows only."""
    return manifest[manifest['fit_type'] == 'oos'].copy()


def agg_stage_oos(oos):
    """Mean OOS metrics per stage, require at least 20 of 25 folds."""
    group = ['stage_id', 'stage_type', 'dist', 'covars', 'threshold_pct']
    agg = (
        oos.groupby(group, dropna=False)
        .agg(
            n_folds    = ('fold', 'count'),
            roc_auc    = ('roc_auc',     'mean'),
            brier      = ('brier_score', 'mean'),
            logloss    = ('log_loss',    'mean'),
            mae_rate   = ('mae_rate',    'mean'),
            mae_log    = ('mae_log',     'mean'),
            cor_rate   = ('cor_rate',    'mean'),
        )
        .reset_index()
    )
    return agg[agg['n_folds'] >= 20]


def print_best_stages(agg):
    """Print top-5 per stage type."""
    binary_types = ['s1', 'pos_binary']
    count_types  = ['pos_count', 'dh_bulk', 'tail', 'single']

    print("\n" + "="*72)
    print("STAGE-LEVEL OOS RANKINGS")
    print("="*72)

    for stype in ['s1', 'pos_count', 'pos_binary', 'dh_bulk', 'tail', 'single']:
        sub = agg[agg['stage_type'] == stype]
        if sub.empty:
            continue

        if stype in binary_types:
            metric, ascending = 'roc_auc', False
        else:
            metric, ascending = 'mae_rate', True

        sub = sub.dropna(subset=[metric])
        ranked = sub.sort_values(metric, ascending=ascending).head(10)
        print(f"\n── {stype} (n={len(sub)}, metric={metric}) ──")
        for _, r in ranked.iterrows():
            thr = f" thr={int(r.threshold_pct)}" if pd.notna(r.threshold_pct) else ''
            extra = (f"  roc_auc={r.roc_auc:.3f}" if stype in binary_types
                     else f"  mae_rate={r.mae_rate:.2f}  cor_rate={r.cor_rate:.3f}")
            print(f"  {abbrev(r.dist):12s}  {r.covars:30s}{thr}{extra}  n_folds={r.n_folds:.0f}")


# ── Phase 2: full-model assembly ──────────────────────────────────────────────

def _ml_log(row):
    """True if this stage is an ML model with log-transformed covariates (redundant)."""
    return row['dist'] in ML_MODELS and 'log' in str(row['covars'])


def select_top(agg, stage_type, metric, ascending, k):
    """Select top-k stages for a given type, excluding redundant ML+log covariate combos."""
    sub = (agg[(agg['stage_type'] == stage_type)]
           .dropna(subset=[metric])
           [~agg.apply(_ml_log, axis=1)])
    return list(sub.sort_values(metric, ascending=ascending).head(k)['stage_id'])


def select_top_by_threshold(agg, stage_type, metric, ascending, k):
    """Select top-k stages per threshold_pct bucket, excluding redundant ML+log combos."""
    sub = (agg[(agg['stage_type'] == stage_type)]
           .dropna(subset=[metric])
           [~agg.apply(_ml_log, axis=1)])
    result = {}
    for thr, grp in sub.groupby('threshold_pct'):
        result[int(thr)] = list(grp.sort_values(metric, ascending=ascending).head(k)['stage_id'])
    return result


def _model_path(stage_id, seed, fold):
    return STAGE_MODELS_DIR / f'{stage_id}_seed{seed}_fold{fold}.pkl'


def _predict_stage(args):
    """
    Load a stage pkl model and generate predictions on the test subset.
    Returns (stage_id, seed, fold, preds_indexed) where preds_indexed is a
    pd.Series indexed by test-row integer positions in full data.
    """
    stage_id, seed, fold, data, stage_meta = args
    path = _model_path(stage_id, seed, fold)
    if not path.exists():
        return (stage_id, seed, fold, None)

    try:
        with open(path, 'rb') as fh:
            fitted = pickle.load(fh)
    except Exception:
        return (stage_id, seed, fold, None)

    rng      = np.random.default_rng(seed)
    fold_ids = rng.integers(0, K_FOLDS, size=len(data))
    train    = data[fold_ids != fold]
    test     = data[fold_ids == fold]
    test_idx = np.where(fold_ids == fold)[0]

    dist     = stage_meta['dist']
    covars   = stage_meta['covars']
    stype    = stage_meta['stage_type']
    thr_pct  = stage_meta.get('threshold_pct')
    is_ml    = dist in ML_MODELS
    dist_mod = DISTRIBUTIONS[dist]

    try:
        # Determine data subset and task for train (to get column names for align_X)
        include_log = is_ml and stype in ('pos_count', 'dh_bulk', 'single')

        if stype == 's1':
            train_sub = train
            test_sub  = test
            sub_idx   = test_idx
            task      = 'binary'
            include_log = False
        elif stype == 'pos_count':
            train_sub = train[train['death_y_n'] == 1]
            test_sub  = test[test['death_y_n'] == 1]
            sub_idx   = test_idx[test['death_y_n'].values == 1]
            task      = 'count'
            include_log = is_ml
        elif stype == 'pos_binary':
            train_sub = train[train['death_y_n'] == 1]
            test_sub  = test[test['death_y_n'] == 1]
            sub_idx   = test_idx[test['death_y_n'].values == 1]
            task      = 'binary'
            include_log = False
        elif stype == 'dh_bulk':
            u          = float(np.percentile(train[train['death_y_n'] == 1]['total_deaths'].values, thr_pct))
            train_sub  = train[(train['death_y_n'] == 1) & (train['total_deaths'] <= u)]
            test_pos   = test[test['death_y_n'] == 1]
            test_sub   = test_pos[test_pos['total_deaths'] <= u]
            sub_idx    = test_idx[
                (test['death_y_n'].values == 1) &
                (test['total_deaths'].values <= u)
            ]
            task       = 'count'
            include_log = is_ml
        elif stype == 'tail':
            u          = float(np.percentile(train[train['death_y_n'] == 1]['total_deaths'].values, thr_pct))
            train_sub  = train[(train['death_y_n'] == 1) & (train['total_deaths'] > u)]
            test_pos   = test[test['death_y_n'] == 1]
            test_sub   = test_pos[test_pos['total_deaths'] > u]
            sub_idx    = test_idx[
                (test['death_y_n'].values == 1) &
                (test['total_deaths'].values > u)
            ]
            task       = 'count'
            include_log = False
        elif stype == 'single':
            train_sub  = train
            test_sub   = test
            sub_idx    = test_idx
            task       = 'count'
            include_log = is_ml
        else:
            return (stage_id, seed, fold, None)

        if len(train_sub) == 0 or len(test_sub) == 0:
            return (stage_id, seed, fold, None)

        X_tr = build_X(train_sub, covars, include_log_exp=include_log)
        X_te = align_X(build_X(test_sub, covars, include_log_exp=include_log), list(X_tr.columns))

        if stype == 'tail':
            exp_te = test_sub['exposed_population'].values
            preds  = dist_mod.predict(fitted, X_te, exposure=exp_te)
        elif stype in ('pos_count', 'dh_bulk', 'single'):
            exp_te = test_sub['exposed_population'].values
            preds  = dist_mod.predict(fitted, X_te, exposure=exp_te, task='count')
        else:
            preds  = dist_mod.predict(fitted, X_te, task='binary')

        preds = np.maximum(np.asarray(preds, dtype=float), 0)
        return (stage_id, seed, fold, pd.Series(preds, index=sub_idx))

    except Exception as e:
        return (stage_id, seed, fold, None)


def precompute_predictions(stage_ids, stage_meta_lookup, data):
    """
    For each (stage_id, seed, fold), load pkl and compute predictions.
    Returns dict: (stage_id, seed, fold) → pd.Series indexed by test-row positions.
    """
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
    for (sid, seed, fold, preds) in results:
        if preds is not None:
            cache[(sid, seed, fold)] = preds
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


def _add_coverage(metrics, obs, pred):
    for frac, label in [(0.01,'cov_1'), (0.05,'cov_5'), (0.10,'cov_10'), (0.20,'cov_20')]:
        metrics[label] = _coverage_at(obs, pred, frac)
    return metrics


def assemble_single(s_sid, pred_cache, data, seeds, k_folds):
    """OOS metrics for a single-stage model."""
    obs_all = []; pred_all = []; exp_all = []
    for seed in seeds:
        for fold in range(k_folds):
            key = (s_sid, seed, fold)
            if key not in pred_cache:
                continue
            rng      = np.random.default_rng(seed)
            fold_ids = rng.integers(0, k_folds, size=len(data))
            test_idx = np.where(fold_ids == fold)[0]
            preds_s  = pred_cache[key].reindex(test_idx).fillna(0).values
            obs  = data.iloc[test_idx]['total_deaths'].values
            exp  = data.iloc[test_idx]['exposed_population'].values
            obs_all.append(obs); pred_all.append(preds_s); exp_all.append(exp)
    if not obs_all:
        return None
    obs_c = np.concatenate(obs_all); pred_c = np.concatenate(pred_all)
    m = calc_metrics(obs_c, pred_c, np.concatenate(exp_all))
    return _add_coverage(m, obs_c, pred_c)


def assemble_hurdle(s1_sid, pc_sid, pred_cache, data, seeds, k_folds):
    """OOS metrics for hurdle model: p_pos * e_count."""
    obs_all = []; pred_all = []; exp_all = []
    for seed in seeds:
        for fold in range(k_folds):
            k1 = (s1_sid, seed, fold); k2 = (pc_sid, seed, fold)
            if k1 not in pred_cache or k2 not in pred_cache:
                continue
            rng      = np.random.default_rng(seed)
            fold_ids = rng.integers(0, k_folds, size=len(data))
            test_idx = np.where(fold_ids == fold)[0]
            n_test   = len(test_idx)

            p_pos    = pred_cache[k1].reindex(test_idx).fillna(0.5).values
            e_count  = np.zeros(n_test)
            pos_map  = {pos: i for i, pos in enumerate(test_idx)}
            for global_i, local_i in pos_map.items():
                if global_i in pred_cache[k2].index:
                    e_count[local_i] = pred_cache[k2][global_i]

            preds = p_pos * e_count
            obs   = data.iloc[test_idx]['total_deaths'].values
            exp   = data.iloc[test_idx]['exposed_population'].values
            obs_all.append(obs); pred_all.append(preds); exp_all.append(exp)
    if not obs_all:
        return None
    obs_c = np.concatenate(obs_all); pred_c = np.concatenate(pred_all)
    m = calc_metrics(obs_c, pred_c, np.concatenate(exp_all))
    return _add_coverage(m, obs_c, pred_c)


def assemble_pot(s1_sid, pc_sid, tail_sid, thr_pct, pred_cache, data, seeds, k_folds):
    """
    OOS metrics for POT model.
    E[Y] = p_pos * ((1 - p_exceed) * e_bulk + p_exceed * e_tail)
    p_exceed is estimated from training positives for each fold.
    """
    obs_all = []; pred_all = []; exp_all = []
    for seed in seeds:
        for fold in range(k_folds):
            k1 = (s1_sid, seed, fold); k2 = (pc_sid, seed, fold); k3 = (tail_sid, seed, fold)
            if k1 not in pred_cache or k2 not in pred_cache or k3 not in pred_cache:
                continue
            rng      = np.random.default_rng(seed)
            fold_ids = rng.integers(0, k_folds, size=len(data))
            train    = data[fold_ids != fold]
            test_idx = np.where(fold_ids == fold)[0]
            n_test   = len(test_idx)

            # p_exceed from training positives
            train_pos = train[train['death_y_n'] == 1]['total_deaths'].values
            u         = float(np.percentile(train_pos, thr_pct)) if len(train_pos) > 0 else 0.0
            p_exceed  = float((train_pos > u).mean()) if len(train_pos) > 0 else 0.0

            p_pos    = pred_cache[k1].reindex(test_idx).fillna(0.5).values
            e_bulk   = np.zeros(n_test)
            e_tail   = np.zeros(n_test)
            pos_idx  = np.where(fold_ids == fold)[0]

            for i, gi in enumerate(test_idx):
                if gi in pred_cache[k2].index:
                    e_bulk[i] = pred_cache[k2][gi]
                if gi in pred_cache[k3].index:
                    e_tail[i] = pred_cache[k3][gi]

            preds = p_pos * ((1 - p_exceed) * e_bulk + p_exceed * e_tail)
            obs   = data.iloc[test_idx]['total_deaths'].values
            exp   = data.iloc[test_idx]['exposed_population'].values
            obs_all.append(obs); pred_all.append(preds); exp_all.append(exp)
    if not obs_all:
        return None
    obs_c = np.concatenate(obs_all); pred_c = np.concatenate(pred_all)
    m = calc_metrics(obs_c, pred_c, np.concatenate(exp_all))
    return _add_coverage(m, obs_c, pred_c)


def assemble_dh(s1_sid, pb_sid, bulk_sid, tail_sid, pred_cache, data, seeds, k_folds):
    """
    OOS metrics for double-hurdle model.
    E[Y] = p_pos * ((1 - p_high) * e_bulk + p_high * e_tail)
    p_high comes from pos_binary stage predictions.
    """
    obs_all = []; pred_all = []; exp_all = []
    for seed in seeds:
        for fold in range(k_folds):
            k1 = (s1_sid, seed, fold); k2 = (pb_sid, seed, fold)
            k3 = (bulk_sid, seed, fold); k4 = (tail_sid, seed, fold)
            if not all(k in pred_cache for k in (k1, k2, k3, k4)):
                continue
            rng      = np.random.default_rng(seed)
            fold_ids = rng.integers(0, k_folds, size=len(data))
            test_idx = np.where(fold_ids == fold)[0]
            n_test   = len(test_idx)

            p_pos   = pred_cache[k1].reindex(test_idx).fillna(0.5).values
            p_high  = np.zeros(n_test)
            e_bulk  = np.zeros(n_test)
            e_tail  = np.zeros(n_test)

            for i, gi in enumerate(test_idx):
                if gi in pred_cache[k2].index:
                    p_high[i] = pred_cache[k2][gi]
                if gi in pred_cache[k3].index:
                    e_bulk[i] = pred_cache[k3][gi]
                if gi in pred_cache[k4].index:
                    e_tail[i] = pred_cache[k4][gi]

            preds = p_pos * ((1 - p_high) * e_bulk + p_high * e_tail)
            obs   = data.iloc[test_idx]['total_deaths'].values
            exp   = data.iloc[test_idx]['exposed_population'].values
            obs_all.append(obs); pred_all.append(preds); exp_all.append(exp)
    if not obs_all:
        return None
    obs_c = np.concatenate(obs_all); pred_c = np.concatenate(pred_all)
    m = calc_metrics(obs_c, pred_c, np.concatenate(exp_all))
    return _add_coverage(m, obs_c, pred_c)


def stage_label(agg, stage_id):
    """Short label: 'sm_nb/logwind_logsdi' etc."""
    row = agg[agg['stage_id'] == stage_id]
    if row.empty:
        return stage_id[:8]
    r = row.iloc[0]
    thr = f"@{int(r.threshold_pct)}" if pd.notna(r.threshold_pct) else ''
    return f"{abbrev(r.dist)}/{r.covars}{thr}"


def find_no_covars_analog(agg, stage_id):
    """
    Find the same dist + stage_type + threshold_pct but with covars='none'.
    Returns stage_id or None.
    """
    row = agg[agg['stage_id'] == stage_id]
    if row.empty:
        return None
    r = row.iloc[0]
    match = agg[
        (agg['stage_type'] == r.stage_type) &
        (agg['dist']       == r.dist) &
        (agg['covars']     == 'none') &
        (agg['threshold_pct'].isna() if pd.isna(r.threshold_pct)
         else agg['threshold_pct'] == r.threshold_pct)
    ]
    return match.iloc[0]['stage_id'] if not match.empty else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading stage manifest...")
    manifest = load_stage_manifest()
    print(f"  Total records: {len(manifest)}")

    oos = load_oos(manifest)
    print(f"  OOS records: {len(oos)}")

    # Stage-level meta (any record per stage is fine for meta columns)
    stage_meta_df = manifest.drop_duplicates('stage_id')[
        ['stage_id', 'stage_type', 'dist', 'covars', 'threshold_pct']
    ].set_index('stage_id')
    stage_meta_lookup = stage_meta_df.to_dict('index')

    agg = agg_stage_oos(oos)
    print_best_stages(agg)

    # ── Phase 2: assembly ────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("ASSEMBLING FULL MODELS")
    print("="*72)

    # Select candidate stages
    s1_ids    = select_top(agg, 's1',        'roc_auc',  False, TOP_K)
    pc_ids    = select_top(agg, 'pos_count', 'mae_rate', True,  TOP_K)
    sg_ids    = select_top(agg, 'single',    'mae_rate', True,  TOP_K)
    pb_by_thr = select_top_by_threshold(agg, 'pos_binary', 'roc_auc',  False, TOP_K)
    bk_by_thr = select_top_by_threshold(agg, 'dh_bulk',    'mae_rate', True,  TOP_K)
    tl_by_thr = select_top_by_threshold(agg, 'tail',       'mae_rate', True,  TOP_K)

    thresholds = sorted(set(pb_by_thr) & set(bk_by_thr) & set(tl_by_thr))

    all_stage_ids = set(s1_ids) | set(pc_ids) | set(sg_ids)
    for thr in thresholds:
        all_stage_ids |= set(pb_by_thr.get(thr, []))
        all_stage_ids |= set(bk_by_thr.get(thr, []))
        all_stage_ids |= set(tl_by_thr.get(thr, []))
    # Also collect no-covariate analogs
    no_cov_ids = set()
    for sid in all_stage_ids:
        nc = find_no_covars_analog(agg, sid)
        if nc:
            no_cov_ids.add(nc)
    all_stage_ids |= no_cov_ids

    print(f"\nCandidate stages: {len(all_stage_ids)}")
    print("Loading data...")
    data = load_tc_data()
    print(f"  {len(data)} rows")

    print("\nPre-computing predictions...")
    cache = precompute_predictions(list(all_stage_ids), stage_meta_lookup, data)

    # ── Enumerate and assemble ────────────────────────────────────────────────
    results = []

    # Single
    print("\nAssembling single models...")
    for sid in sg_ids:
        m = assemble_single(sid, cache, data, SEEDS, K_FOLDS)
        if m:
            results.append({'model_type': 'single', 's1': None, 'pc': None,
                            'pb': None, 'bk': None, 'tl': None,
                            'sg': sid, 'threshold_pct': None, **m,
                            'label': f"single | {stage_label(agg, sid)}"})

    # Hurdle
    print("Assembling hurdle models...")
    for s1 in s1_ids:
        for pc in pc_ids:
            m = assemble_hurdle(s1, pc, cache, data, SEEDS, K_FOLDS)
            if m:
                results.append({'model_type': 'hurdle', 's1': s1, 'pc': pc,
                                'pb': None, 'bk': None, 'tl': None,
                                'sg': None, 'threshold_pct': None, **m,
                                'label': (f"hurdle | s1={stage_label(agg,s1)} | "
                                          f"count={stage_label(agg,pc)}")})

    # POT
    print("Assembling POT models...")
    for thr in thresholds:
        for s1 in s1_ids:
            for pc in pc_ids:
                for tl in tl_by_thr.get(thr, []):
                    m = assemble_pot(s1, pc, tl, thr, cache, data, SEEDS, K_FOLDS)
                    if m:
                        results.append({'model_type': 'pot', 's1': s1, 'pc': pc,
                                        'pb': None, 'bk': tl, 'tl': tl,
                                        'sg': None, 'threshold_pct': thr, **m,
                                        'label': (f"pot@{thr} | s1={stage_label(agg,s1)} | "
                                                  f"bulk={stage_label(agg,pc)} | "
                                                  f"tail={stage_label(agg,tl)}")})

    # DH
    print("Assembling DH models...")
    for thr in thresholds:
        for s1 in s1_ids:
            for pb in pb_by_thr.get(thr, []):
                for bk in bk_by_thr.get(thr, []):
                    for tl in tl_by_thr.get(thr, []):
                        m = assemble_dh(s1, pb, bk, tl, cache, data, SEEDS, K_FOLDS)
                        if m:
                            results.append({'model_type': 'dh', 's1': s1, 'pc': None,
                                            'pb': pb, 'bk': bk, 'tl': tl,
                                            'sg': None, 'threshold_pct': thr, **m,
                                            'label': (f"dh@{thr} | s1={stage_label(agg,s1)} | "
                                                      f"pb={stage_label(agg,pb)} | "
                                                      f"bulk={stage_label(agg,bk)} | "
                                                      f"tail={stage_label(agg,tl)}")})

    if not results:
        print("No assembled model results — check pred_cache coverage.")
        return

    df = pd.DataFrame(results)

    # ── Multi-metric top-~50 ──────────────────────────────────────────────────
    # Take top N per metric (ascending=lower is better for errors, descending for cor)
    # so that the union is ~50 unique models.
    METRICS_INFO = [
        ('mae_rate',   True,  'MAE rate'),
        ('rmse_rate',  True,  'RMSE rate'),
        ('mae_count',  True,  'MAE count'),
        ('rmse_count', True,  'RMSE count'),
        ('cor_rate',   False, 'Cor rate'),
        ('cor_count',  False, 'Cor count'),
        ('cov_1',      False, 'Coverage @1%'),
        ('cov_5',      False, 'Coverage @5%'),
        ('cov_10',     False, 'Coverage @10%'),
        ('cov_20',     False, 'Coverage @20%'),
    ]
    N_PER_METRIC = 6  # 6 × 10 = 60, union ≈ 50 after overlaps

    selected_idx = set()
    for col, asc, label in METRICS_INFO:
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col])
        selected_idx.update(sub.sort_values(col, ascending=asc).head(N_PER_METRIC).index)

    top_df = df.loc[sorted(selected_idx)].sort_values('mae_rate')

    print("\n" + "="*120)
    print(f"TOP MODELS — multi-metric union (top {N_PER_METRIC}/metric × {len(METRICS_INFO)} metrics, {len(top_df)} unique models)")
    print(f"  {'#':>3}  {'type':>6}  {'thr':>4}  {'mae_r':>6}  {'rmse_r':>7}  {'cor_r':>6}  "
          f"{'cov@1%':>6}  {'cov@5%':>6}  {'cov@10%':>7}  {'cov@20%':>7}  label")
    print("  " + "-"*130)
    for rank, (_, r) in enumerate(top_df.iterrows(), 1):
        thr = f"{int(r.threshold_pct)}" if pd.notna(r.get('threshold_pct')) else '—'
        print(f"  {rank:3d}  {r.model_type:>6}  {thr:>4}  {r.mae_rate:6.3f}  {r.rmse_rate:7.1f}  "
              f"{r.cor_rate:6.3f}  {r.cov_1:6.3f}  {r.cov_5:6.3f}  {r.cov_10:7.3f}  "
              f"{r.cov_20:7.3f}  {r.label}")

    # ── Best per type ─────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("BEST PER MODEL TYPE (by mae_rate)")
    print("="*72)
    for mtype in ['single', 'hurdle', 'pot', 'dh']:
        sub = df[df['model_type'] == mtype].sort_values('mae_rate')
        if sub.empty:
            print(f"  {mtype}: no results")
            continue
        best = sub.iloc[0]
        print(f"\n  {mtype.upper()}: mae_rate={best.mae_rate:.2f}  rmse_rate={best.rmse_rate:.1f}  "
              f"cor_rate={best.cor_rate:.3f}  mae_log={best.mae_log:.3f}")
        print(f"    {best.label}")

    # ── No-covariate baselines ────────────────────────────────────────────────
    # Best model of each type where ALL component stages use covars='none'.
    # Because ML models (sk_rf, xgb) can't use 'none', this selects from
    # statsmodels/sklearn Poisson/logistic families that support intercept-only.
    print("\n" + "="*72)
    print("NO-COVARIATE BASELINES (best all-none-covariate model per type)")
    print("="*72)

    none_agg = agg[agg['covars'] == 'none']

    def best_none(stage_type, metric, ascending, threshold_pct=None):
        """Best stage with covars='none' for given type and optional threshold."""
        sub = none_agg[none_agg['stage_type'] == stage_type].dropna(subset=[metric])
        if threshold_pct is not None:
            sub = sub[sub['threshold_pct'] == threshold_pct]
        if sub.empty:
            return None
        return sub.sort_values(metric, ascending=ascending).iloc[0]['stage_id']

    # Collect none-covariate stage IDs
    nc_sg  = best_none('single',    'mae_rate', True)
    nc_s1  = best_none('s1',        'roc_auc',  False)
    nc_pc  = best_none('pos_count', 'mae_rate', True)

    nc_stages_by_thr = {}
    for thr in thresholds:
        nc_stages_by_thr[thr] = {
            'pb': best_none('pos_binary', 'roc_auc',  False, thr),
            'bk': best_none('dh_bulk',    'mae_rate', True,  thr),
            'tl': best_none('tail',       'mae_rate', True,  thr),
        }

    # Load predictions for any missing none-covariate stages
    nc_all_ids = {nc_sg, nc_s1, nc_pc} - {None}
    for thr in thresholds:
        nc_all_ids |= {v for v in nc_stages_by_thr[thr].values() if v}
    missing_nc = nc_all_ids - {k[0] for k in cache}
    if missing_nc:
        print(f"  Loading {len(missing_nc)} additional none-covariate stages...")
        nc_extra = precompute_predictions(list(missing_nc), stage_meta_lookup, data)
        cache.update(nc_extra)

    # Single
    if nc_sg:
        m = assemble_single(nc_sg, cache, data, SEEDS, K_FOLDS)
        if m:
            print(f"\n  SINGLE (no-cov): mae_rate={m['mae_rate']:.2f}  "
                  f"rmse_rate={m['rmse_rate']:.1f}  cor_rate={m['cor_rate']:.3f}")
            print(f"    {stage_label(agg, nc_sg)}")

    # Hurdle
    if nc_s1 and nc_pc:
        m = assemble_hurdle(nc_s1, nc_pc, cache, data, SEEDS, K_FOLDS)
        if m:
            print(f"\n  HURDLE (no-cov): mae_rate={m['mae_rate']:.2f}  "
                  f"rmse_rate={m['rmse_rate']:.1f}  cor_rate={m['cor_rate']:.3f}")
            print(f"    s1={stage_label(agg,nc_s1)} | count={stage_label(agg,nc_pc)}")

    # POT and DH — use threshold that gave best DH result
    best_dh = df[df['model_type'] == 'dh'].sort_values('mae_rate')
    best_thr = int(best_dh.iloc[0]['threshold_pct']) if not best_dh.empty else thresholds[0]

    nc_tl_bt = nc_stages_by_thr.get(best_thr, {}).get('tl')
    if nc_s1 and nc_pc and nc_tl_bt:
        m = assemble_pot(nc_s1, nc_pc, nc_tl_bt, best_thr, cache, data, SEEDS, K_FOLDS)
        if m:
            print(f"\n  POT@{best_thr} (no-cov): mae_rate={m['mae_rate']:.2f}  "
                  f"rmse_rate={m['rmse_rate']:.1f}  cor_rate={m['cor_rate']:.3f}")
            print(f"    s1={stage_label(agg,nc_s1)} | bulk={stage_label(agg,nc_pc)} | "
                  f"tail={stage_label(agg,nc_tl_bt)}")

    nc_pb_bt = nc_stages_by_thr.get(best_thr, {}).get('pb')
    nc_bk_bt = nc_stages_by_thr.get(best_thr, {}).get('bk')
    if nc_s1 and nc_pb_bt and nc_bk_bt and nc_tl_bt:
        m = assemble_dh(nc_s1, nc_pb_bt, nc_bk_bt, nc_tl_bt, cache, data, SEEDS, K_FOLDS)
        if m:
            print(f"\n  DH@{best_thr} (no-cov): mae_rate={m['mae_rate']:.2f}  "
                  f"rmse_rate={m['rmse_rate']:.1f}  cor_rate={m['cor_rate']:.3f}")
            print(f"    s1={stage_label(agg,nc_s1)} | pb={stage_label(agg,nc_pb_bt)} | "
                  f"bulk={stage_label(agg,nc_bk_bt)} | tail={stage_label(agg,nc_tl_bt)}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    from idd_climate_models.tc_models.constants import STAGE_RESULTS_DIR
    out_path = STAGE_RESULTS_DIR / 'top_models.csv'
    df.sort_values('mae_rate').to_csv(out_path, index=False)
    print(f"\n\nFull results saved to: {out_path}")
    print(f"Total assembled models evaluated: {len(df)}")


if __name__ == '__main__':
    main()
