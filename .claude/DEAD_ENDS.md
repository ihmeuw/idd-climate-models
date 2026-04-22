# Dead ends

<!-- Append-only. Never delete or overwrite entries. -->

## 2026-03-23: Spec-based model comparison architecture
**What I tried:** 3678 specs, each defining a full model structure (e.g., hurdle with specific distributions for each stage). Each spec re-fit all its component stages independently. ~13,260 model.fit() calls insample.
**Why I stopped:** Massively redundant — the same logistic regression for P(Y>0) was fit hundreds of times with identical data and covariates. Replaced with stage-based architecture where each unique fit happens once.
**Refs:** `src/idd_climate_models/tc_models/spec_grid.py`, `run_one_spec.py`, `orchestrate_tc_comparison.py` — files retained but superseded.

## 2026-03-23: Per-threshold isolated analysis of s2/bulk/tail
**What I tried:** Tables comparing s2, bulk, tail stages in isolation at each threshold, then comparing thresholds by their within-threshold metrics.
**Why I stopped:** Comparing thresholds this way is invalid — a thr=90 tail model trained on the top 10% can't be compared to a thr=70 tail model on the top 30% without assembling the full model. The right comparison is full assembled DH model evaluated on the same full test set. Replaced with cross-threshold table using full assembled models.
**Refs:** `analyze_dh.py` Tables 1-4 still valid for within-threshold component ranking; Table 5+ uses full assembly.

## 2026-03-23: Log-covariate ML stages (logwind_rf, logwind_xgb, etc.)
**What I tried:** Including logwind, logwind_sdi, logwind_logsdi_basin_island, etc. covariate sets for sklearn_rf and sklearn_xgb stages.
**Why I stopped:** Tree models are invariant to monotone transforms. logwind and wind produce mathematically identical RF/XGB predictions. Results were duplicated in rankings. Filter applied at analysis stage; stage_grid.py not yet fixed.
**Refs:** `stage_grid.py` line 47 `_REQUIRES_COVARS` — should add ML_MODELS to a `_NO_LOG_COVARS` set.

## 2026-03-30: tc_stages workflow 556389 has 77 failed jobs
**What happened:** Original tc_stages run (2026-03-23) has 77/2069 failed tasks. Example failure: stage 1970 (statsmodels_tweedie single stage) fails with "gamma requires y > 0" — error message says gamma but stage is tweedie. Likely tweedie internally uses gamma family for certain power parameters.
**Root cause:** Distribution compatibility not properly enforced for edge cases. Single stages include y=0 rows; gamma/tweedie with certain parameters can't handle zeros.
**Status:** Never rerun after identifying this. Analysis proceeded with incomplete results.
**Fix needed:** Either exclude problematic distributions from single stages, or handle zeros in the distribution wrappers.

## 2026-04-02: Rate-space predict_df assembly — session ended without fix
**What I tried:** Rewrote predict_df and predict_df_oos to predict in rate space (exposure=1 for GLMs, clip bulk to (0,u], clip tail to [u,∞)), then assemble as pred_rate = s1*((1-s2)*bulk_rate + s2*tail_rate) and convert to counts via pred_count = pred_rate * exp_all.
**Why I stopped:** Assembled insample pred/obs ratio came out at 4.56 (1.5M predicted vs 330K observed). Stage-level bulk (1.41) and tail (0.79) ratios look plausible. Rates are confirmed < 1. Root cause of inflated assembly not identified before session ended.
**Refs:** `src/idd_climate_models/tc_models/stage_plots.py` predict_df method; `notebooks/tc_models/predict_df_demo.ipynb`
