"""
analyze_dh.py

Deep-dive analysis of double-hurdle (DH) model components.

Tables produced:
  1. S1 (binary P(Y>0)) — roc_auc, brier, log_loss; OOS + insample + gap
  2. S2 / pos_binary (binary P(tail|Y>0)) — same, per threshold
  3. Bulk / dh_bulk (count E[Y|bulk]) — mae_rate, rmse_rate, cor_rate, mae_count, rmse_count; OOS + insample
  4. Tail — same + within-tail top-x% coverage; OOS + insample
  5. Cross-threshold summary — for each threshold, best s2/bulk/tail and combined OOS metrics
  6. Full-model coverage — top-5/10/20% recall across assembled DH models

Usage:
    python src/idd_climate_models/tc_models/analyze_dh.py
"""

import sys
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[4] / 'src'))
warnings.filterwarnings('ignore')

from idd_climate_models.tc_models.cache import load_stage_manifest, STAGE_MODELS_DIR
from idd_climate_models.tc_models.data import load_tc_data
from idd_climate_models.tc_models.features import build_X, align_X
from idd_climate_models.tc_models.distributions import DISTRIBUTIONS, ML_MODELS

SEEDS   = [123, 456, 789, 101112, 131415]
K_FOLDS = 5
THREADS = 8
COVERAGE_TOPS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
THRESHOLDS = [70, 75, 80, 85, 90, 95]

# ML models are tree-based — invariant to monotone transforms, exclude log covariates
def _is_redundant(dist, covars):
    return dist in ML_MODELS and 'log' in str(covars)

_ABBREV = {
    'statsmodels_logistic': 'sm_logit',
    'sklearn_logistic':     'sk_logit',
    'statsmodels_nb':       'sm_nb',
    'statsmodels_poisson':  'sm_pois',
    'statsmodels_zinb':     'sm_zinb',
    'statsmodels_zip':      'sm_zip',
    'statsmodels_gamma':    'sm_gamma',
    'statsmodels_lognormal':'sm_lnorm',
    'statsmodels_gpd':      'sm_gpd',
    'scipy_gpd':            'sc_gpd',
    'sklearn_lognormal':    'sk_lnorm',
    'sklearn_rf':           'sk_rf',
    'sklearn_xgb':          'sk_xgb',
    'sklearn_gb':           'sk_gb',
    'sklearn_tweedie':      'sk_twee',
    'sklearn_poisson':      'sk_pois',
}
def ab(d): return _ABBREV.get(d, d)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_aggs():
    manifest = load_stage_manifest()
    oos = manifest[manifest['fit_type'] == 'oos']
    ins = manifest[manifest['fit_type'] == 'insample']

    group = ['stage_id', 'stage_type', 'dist', 'covars', 'threshold_pct']

    agg_o = (oos.groupby(group, dropna=False)
             .agg(n_folds    = ('fold',        'count'),
                  roc_auc    = ('roc_auc',     'mean'),
                  brier      = ('brier_score', 'mean'),
                  logloss    = ('log_loss',    'mean'),
                  mae_rate   = ('mae_rate',    'mean'),
                  rmse_rate  = ('rmse_rate',   'mean'),
                  mae_count  = ('mae_count',   'mean'),
                  rmse_count = ('rmse_count',  'mean'),
                  cor_rate   = ('cor_rate',    'mean'),
                  cor_count  = ('cor_count',   'mean'))
             .reset_index())

    agg_i = (ins.groupby(group, dropna=False)
             .agg(roc_auc_i  = ('roc_auc',     'mean'),
                  brier_i    = ('brier_score', 'mean'),
                  logloss_i  = ('log_loss',    'mean'),
                  mae_rate_i = ('mae_rate',    'mean'),
                  cor_rate_i = ('cor_rate',    'mean'))
             .reset_index())

    agg = (agg_o[agg_o['n_folds'] >= 20]
           .merge(agg_i[['stage_id','roc_auc_i','brier_i','logloss_i',
                          'mae_rate_i','cor_rate_i']], on='stage_id', how='left'))

    # Drop log-covariate ML stages (invariant to monotone transforms)
    agg = agg[~agg.apply(lambda r: _is_redundant(r['dist'], r['covars']), axis=1)]
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Table helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_binary_table(agg, stage_type, title, threshold=None):
    sub = agg[agg['stage_type'] == stage_type].copy()
    if threshold is not None:
        sub = sub[sub['threshold_pct'] == threshold]
    sub = sub.dropna(subset=['roc_auc']).sort_values('roc_auc', ascending=False)

    print(f"\n{'='*90}")
    print(title)
    print(f"{'='*90}")
    hdr = f"  {'dist':10s} {'covars':28s}  {'roc_auc_OOS':>11}  {'brier_OOS':>9}  {'logloss_OOS':>11}  {'roc_auc_ins':>11}  {'gap(auc)':>8}"
    print(hdr)
    print("  " + "-"*87)
    for _, r in sub.iterrows():
        gap = (r.roc_auc_i - r.roc_auc) if pd.notna(r.roc_auc_i) else float('nan')
        print(f"  {ab(r.dist):10s} {r.covars:28s}  {r.roc_auc:11.4f}  {r.brier:9.4f}  "
              f"{r.logloss:11.4f}  {r.roc_auc_i:11.4f}  {gap:+8.3f}")


def print_count_table(agg, stage_type, title, threshold=None):
    sub = agg[agg['stage_type'] == stage_type].copy()
    if threshold is not None:
        sub = sub[sub['threshold_pct'] == threshold]
    sub = sub.dropna(subset=['mae_rate']).sort_values('mae_rate')

    print(f"\n{'='*100}")
    print(title)
    print(f"{'='*100}")
    hdr = (f"  {'dist':10s} {'covars':28s}  {'mae_rate':>8}  {'rmse_rate':>9}  "
           f"{'cor_rate':>8}  {'mae_cnt':>8}  {'rmse_cnt':>9}  {'mae_r_ins':>9}  {'gap':>6}")
    print(hdr)
    print("  " + "-"*97)
    for _, r in sub.iterrows():
        gap = (r.mae_rate - r.mae_rate_i) if pd.notna(r.mae_rate_i) else float('nan')
        print(f"  {ab(r.dist):10s} {r.covars:28s}  {r.mae_rate:8.3f}  {r.rmse_rate:9.2f}  "
              f"{r.cor_rate:8.3f}  {r.mae_count:8.1f}  {r.rmse_count:9.1f}  "
              f"{r.mae_rate_i:9.3f}  {gap:+6.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Coverage metric computation (needs model pkl files)
# ─────────────────────────────────────────────────────────────────────────────

def coverage_at(observed, predicted, top_frac):
    """Fraction of top-(top_frac) observed storms also in top-(top_frac) predicted."""
    k = max(1, int(round(len(observed) * top_frac)))
    top_obs  = set(np.argsort(observed)[-k:])
    top_pred = set(np.argsort(predicted)[-k:])
    return len(top_obs & top_pred) / k


def _model_path(stage_id, seed, fold):
    return STAGE_MODELS_DIR / f'{stage_id}_seed{seed}_fold{fold}.pkl'


def _predict_stage(args):
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

    dist    = stage_meta['dist']
    covars  = stage_meta['covars']
    stype   = stage_meta['stage_type']
    thr_pct = stage_meta.get('threshold_pct')
    is_ml   = dist in ML_MODELS
    dist_mod = DISTRIBUTIONS[dist]

    try:
        if stype == 's1':
            X_tr = build_X(train, covars, include_log_exp=False)
            X_te = align_X(build_X(test, covars, include_log_exp=False), list(X_tr.columns))
            preds = dist_mod.predict(fitted, X_te, task='binary')
            sub_idx = test_idx

        elif stype == 'pos_binary':
            tr_pos = train[train['death_y_n'] == 1]
            te_pos = test[test['death_y_n'] == 1]
            sub_idx = test_idx[test['death_y_n'].values == 1]
            if len(tr_pos) == 0 or len(te_pos) == 0:
                return (stage_id, seed, fold, None)
            X_tr = build_X(tr_pos, covars, include_log_exp=False)
            X_te = align_X(build_X(te_pos, covars, include_log_exp=False), list(X_tr.columns))
            preds = dist_mod.predict(fitted, X_te, task='binary')

        elif stype == 'dh_bulk':
            u = float(np.percentile(train[train['death_y_n']==1]['total_deaths'].values, thr_pct))
            tr_b = train[(train['death_y_n']==1) & (train['total_deaths'] <= u)]
            te_b = test[(test['death_y_n']==1) & (test['total_deaths'] <= u)]
            sub_idx = test_idx[(test['death_y_n'].values==1) & (test['total_deaths'].values <= u)]
            if len(tr_b) == 0 or len(te_b) == 0:
                return (stage_id, seed, fold, None)
            X_tr = build_X(tr_b, covars, include_log_exp=is_ml)
            X_te = align_X(build_X(te_b, covars, include_log_exp=is_ml), list(X_tr.columns))
            preds = dist_mod.predict(fitted, X_te, exposure=te_b['exposed_population'].values, task='count')

        elif stype == 'tail':
            u = float(np.percentile(train[train['death_y_n']==1]['total_deaths'].values, thr_pct))
            tr_t = train[(train['death_y_n']==1) & (train['total_deaths'] > u)]
            te_t = test[(test['death_y_n']==1) & (test['total_deaths'] > u)]
            sub_idx = test_idx[(test['death_y_n'].values==1) & (test['total_deaths'].values > u)]
            if len(tr_t) == 0 or len(te_t) == 0:
                return (stage_id, seed, fold, None)
            X_tr = build_X(tr_t, covars, include_log_exp=False)
            X_te = align_X(build_X(te_t, covars, include_log_exp=False), list(X_tr.columns))
            preds = dist_mod.predict(fitted, X_te, exposure=te_t['exposed_population'].values)

        else:
            return (stage_id, seed, fold, None)

        preds = np.maximum(np.asarray(preds, dtype=float), 0)
        return (stage_id, seed, fold, pd.Series(preds, index=sub_idx))
    except Exception:
        return (stage_id, seed, fold, None)


def precompute(stage_ids, meta_lookup, data):
    jobs = [(sid, seed, fold, data, meta_lookup[sid])
            for sid in stage_ids
            for seed in SEEDS
            for fold in range(K_FOLDS)
            if _model_path(sid, seed, fold).exists()]
    print(f"  Loading {len(jobs)} stage×fold predictions ({THREADS} threads)...")
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        results = list(ex.map(_predict_stage, jobs))
    cache = {}
    for sid, seed, fold, preds in results:
        if preds is not None:
            cache[(sid, seed, fold)] = preds
    print(f"  Loaded {len(cache)}")
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# Assemble DH predictions for one combination across all folds
# ─────────────────────────────────────────────────────────────────────────────

def assemble_dh_folds(s1_id, pb_id, bk_id, tl_id, cache, data):
    """
    Return (obs_all, pred_all, exp_all) concatenated across all 25 OOS folds.
    """
    obs_all = []; pred_all = []; exp_all = []
    for seed in SEEDS:
        for fold in range(K_FOLDS):
            k1=(s1_id,seed,fold); k2=(pb_id,seed,fold)
            k3=(bk_id,seed,fold); k4=(tl_id,seed,fold)
            if not all(k in cache for k in (k1,k2,k3,k4)):
                continue
            rng = np.random.default_rng(seed)
            fold_ids = rng.integers(0, K_FOLDS, size=len(data))
            test_idx = np.where(fold_ids == fold)[0]
            n = len(test_idx)

            p_pos  = cache[k1].reindex(test_idx).fillna(0.5).values
            p_high = np.zeros(n); e_bulk = np.zeros(n); e_tail = np.zeros(n)
            for i, gi in enumerate(test_idx):
                if gi in cache[k2].index: p_high[i] = cache[k2][gi]
                if gi in cache[k3].index: e_bulk[i] = cache[k3][gi]
                if gi in cache[k4].index: e_tail[i] = cache[k4][gi]

            preds = p_pos * ((1 - p_high) * e_bulk + p_high * e_tail)
            obs_all.append(data.iloc[test_idx]['total_deaths'].values)
            pred_all.append(preds)
            exp_all.append(data.iloc[test_idx]['exposed_population'].values)

    if not obs_all:
        return None, None, None
    return (np.concatenate(obs_all),
            np.concatenate(pred_all),
            np.concatenate(exp_all))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading manifest...")
    agg = load_aggs()
    meta_lookup = (load_stage_manifest()
                   .drop_duplicates('stage_id')
                   [['stage_id','stage_type','dist','covars','threshold_pct']]
                   .set_index('stage_id')
                   .to_dict('index'))

    # ── TABLE 1: S1 ──────────────────────────────────────────────────────────
    print_binary_table(agg, 's1', "TABLE 1 — S1  (binary P(Y>0), all rows)  sorted by OOS roc_auc")

    # ── TABLE 2: S2 / pos_binary — one sub-table per threshold ───────────────
    print(f"\n\n{'='*90}")
    print("TABLE 2 — S2 / pos_binary  (binary P(tail|Y>0))  sorted by OOS roc_auc")
    for thr in THRESHOLDS:
        print_binary_table(agg, 'pos_binary',
                           f"  threshold = {thr}th percentile", threshold=thr)

    # ── TABLE 3: Bulk / dh_bulk ───────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print("TABLE 3 — Bulk / dh_bulk  (count E[Y|bulk])  sorted by OOS mae_rate")
    for thr in THRESHOLDS:
        print_count_table(agg, 'dh_bulk',
                          f"  threshold = {thr}th percentile", threshold=thr)

    # ── TABLE 4: Tail ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print("TABLE 4 — Tail  (count E[Y|tail])  sorted by OOS mae_rate")
    for thr in THRESHOLDS:
        print_count_table(agg, 'tail',
                          f"  threshold = {thr}th percentile", threshold=thr)

    # ── TABLE 5 & 6: Assembly — need model files ──────────────────────────────
    print("\n\nLoading data and stage models for tables 5 & 6...")
    data = load_tc_data()
    print(f"  {len(data)} rows")

    # Best s1, and best per-threshold s2/bulk/tail (top 2 each)
    TOP = 2
    s1_ids = (agg[agg['stage_type']=='s1']
              .sort_values('roc_auc', ascending=False)
              .head(TOP)['stage_id'].tolist())

    pb_ids = {}; bk_ids = {}; tl_ids = {}
    for thr in THRESHOLDS:
        pb_ids[thr] = (agg[(agg['stage_type']=='pos_binary') & (agg['threshold_pct']==thr)]
                       .sort_values('roc_auc', ascending=False).head(TOP)['stage_id'].tolist())
        bk_ids[thr] = (agg[(agg['stage_type']=='dh_bulk') & (agg['threshold_pct']==thr)]
                       .sort_values('mae_rate').head(TOP)['stage_id'].tolist())
        tl_ids[thr] = (agg[(agg['stage_type']=='tail') & (agg['threshold_pct']==thr)]
                       .sort_values('mae_rate').head(TOP)['stage_id'].tolist())

    all_ids = set(s1_ids)
    for thr in THRESHOLDS:
        all_ids |= set(pb_ids[thr]) | set(bk_ids[thr]) | set(tl_ids[thr])

    cache = precompute(list(all_ids), meta_lookup, data)

    def slabel(sid):
        r = agg[agg['stage_id']==sid]
        if r.empty: return sid[:8]
        r = r.iloc[0]
        return f"{ab(r.dist)}/{r.covars}"

    # ── TABLE 5: error metrics by threshold (full assembled model) ───────────
    cov_fracs = COVERAGE_TOPS
    cov_labels = [f"cov@{int(f*100)}%" for f in cov_fracs]

    print(f"\n\n{'='*100}")
    print("TABLE 5 — Full DH model error metrics by threshold")
    print(f"  All metrics on the same full OOS test set (all storms, all folds)")
    print(f"  s1 fixed = {slabel(s1_ids[0])};  s2/bulk/tail = best per threshold")
    print(f"\n  {'thr':>4}  {'mae_rate':>8}  {'rmse_rate':>9}  {'cor_rate':>8}  "
          f"{'mae_count':>9}  {'rmse_cnt':>9}  {'cor_count':>9}")
    print("  " + "-"*75)

    best_s1 = s1_ids[0]
    thr_rows = []
    for thr in THRESHOLDS:
        best_pb = pb_ids[thr][0] if pb_ids[thr] else None
        best_bk = bk_ids[thr][0] if bk_ids[thr] else None
        best_tl = tl_ids[thr][0] if tl_ids[thr] else None
        if not all([best_pb, best_bk, best_tl]):
            continue

        obs, pred, exp = assemble_dh_folds(best_s1, best_pb, best_bk, best_tl, cache, data)
        if obs is None:
            continue

        pred   = np.maximum(pred, 0)
        rate_o = (obs  / exp) * 1e5
        rate_p = (pred / exp) * 1e5
        mae_r  = float(np.mean(np.abs(rate_o - rate_p)))
        rmse_r = float(np.sqrt(np.mean((rate_o - rate_p)**2)))
        mae_c  = float(np.mean(np.abs(obs - pred)))
        rmse_c = float(np.sqrt(np.mean((obs - pred)**2)))
        nz = obs > 0
        cor_r  = float(np.corrcoef(rate_o[nz], rate_p[nz])[0,1]) if nz.sum()>2 else np.nan
        cor_c  = float(np.corrcoef(obs[nz],    pred[nz])[0,1])   if nz.sum()>2 else np.nan
        covs   = [coverage_at(obs, pred, f) for f in cov_fracs]

        print(f"  {thr:>4}  {mae_r:8.3f}  {rmse_r:9.2f}  {cor_r:8.3f}  "
              f"{mae_c:9.1f}  {rmse_c:9.1f}  {cor_c:9.3f}")

        thr_rows.append({'thr': thr, 'mae_rate': mae_r, 'rmse_rate': rmse_r,
                         'cor_rate': cor_r, 'mae_count': mae_c, 'rmse_count': rmse_c,
                         'cor_count': cor_c, 'obs': obs, 'pred': pred, 'exp': exp,
                         'pb': best_pb, 'bk': best_bk, 'tl': best_tl,
                         **{f'cov_{int(f*100)}': c for f, c in zip(cov_fracs, covs)}})

    # ── TABLE 6: coverage profile by threshold — the full picture ────────────
    print(f"\n\n{'='*100}")
    print("TABLE 6 — Coverage profile by threshold  (all evaluated on same full OOS test set)")
    print(f"  coverage(x%) = fraction of true top-x% storms also in predicted top-x%")
    print(f"  s1 fixed = {slabel(s1_ids[0])};  s2/bulk/tail = best per threshold")
    hdr = f"  {'thr':>4}" + "".join(f"  {lbl:>8}" for lbl in cov_labels)
    print(f"\n{hdr}")
    print("  " + "-"*(6 + 10*len(cov_fracs)))

    for row in thr_rows:
        cov_vals = [row[f'cov_{int(f*100)}'] for f in cov_fracs]
        cov_str  = "".join(f"  {v:8.3f}" for v in cov_vals)
        print(f"  {row['thr']:>4}{cov_str}")

    # ── TABLE 7: miss analysis for each threshold ─────────────────────────────
    print(f"\n\n{'='*100}")
    print("TABLE 7 — Miss analysis at top-10%: for each threshold,")
    print("  what are the observed death counts of the true top-10% storms the model misses?")
    for row in thr_rows:
        obs  = row['obs']
        pred = row['pred']
        k    = max(1, int(round(len(obs) * 0.10)))
        top_obs_set  = set(np.argsort(obs)[-k:])
        top_pred_set = set(np.argsort(pred)[-k:])
        missed_true  = sorted(top_obs_set - top_pred_set)   # true top, not predicted
        false_alarms = sorted(top_pred_set - top_obs_set)   # predicted top, not true
        m_obs = obs[missed_true]   if missed_true  else np.array([])
        f_obs = obs[false_alarms]  if false_alarms else np.array([])
        print(f"\n  thr={row['thr']}  coverage@10%={row['cov_10']:.3f}")
        if len(m_obs):
            print(f"    {len(m_obs)} true top-10% missed — deaths: "
                  f"min={m_obs.min():.0f}  p25={np.percentile(m_obs,25):.0f}  "
                  f"med={np.median(m_obs):.0f}  p75={np.percentile(m_obs,75):.0f}  "
                  f"max={m_obs.max():.0f}")
        if len(f_obs):
            print(f"    {len(f_obs)} false alarms — their actual deaths: "
                  f"min={f_obs.min():.0f}  p25={np.percentile(f_obs,25):.0f}  "
                  f"med={np.median(f_obs):.0f}  p75={np.percentile(f_obs,75):.0f}  "
                  f"max={f_obs.max():.0f}")


if __name__ == '__main__':
    main()
