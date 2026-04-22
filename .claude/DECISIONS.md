# Decisions log

<!-- Append-only. Never delete or overwrite entries. -->

## 2026-03-23: Stage-based architecture replacing spec-based
**Decision:** Each unique model fit (defined by stage_type + dist + covars + threshold_pct) happens exactly once. Stages are combined into full models at evaluation time via fast numpy assembly.
**Why:** Old spec-based approach had ~3678 specs each redundantly fitting shared stages (e.g., the same logistic regression for P(Y>0) was fit hundreds of times). Stage-based approach: 2011 unique fits instead of ~13,260. Also enables each stage to pick its own covariate set independently.
**Revisit if:** Assembly step becomes slow at scale (currently fast); or if stage independence assumption breaks (it doesn't — all stages are mathematically independent).

## 2026-03-23: Double hurdle (DH) as model structure focus
**Decision:** Focus analysis and final model selection on DH structure. Single, hurdle, POT retained as baselines only.
**Why:** DH consistently best across all 6 OOS metrics. mae_rate 1.28 vs next-best POT 1.38. DH also best on coverage metrics at most thresholds.
**Revisit if:** New data substantially changes the outcome distribution; or if the tail model (scipy_gpd) proves unstable on future data.

## 2026-03-23: Exclude redundant sklearn wrappers from analysis
**Decision:** sklearn_logistic, sklearn_poisson, sklearn_tweedie, sklearn_lognormal are excluded from stage candidate selection. Only statsmodels equivalents are used for parametric models.
**Why:** sklearn_logistic = statsmodels_logistic (same unpenalized GLM, different implementation wrapper). sklearn_poisson = statsmodels_poisson. Including both inflates the model count with duplicates and pollutes rankings. These were not caught in stage_grid.py — filter applied at analysis stage.
**Revisit if:** Someone can demonstrate a meaningful numerical difference between pairs on real data.

## 2026-03-23: Log-covariate sets excluded for ML (RF, XGB) models
**Decision:** covariate sets containing 'log' (logwind, logsdi, etc.) are excluded from sklearn_rf and sklearn_xgb candidate selection.
**Why:** Tree-based models are invariant to monotone transforms — log(wind) and wind produce identical RF/XGB predictions (same splits, different threshold values). Including both is pure redundancy.
**Revisit if:** Never — this is mathematically certain for tree models.

## 2026-03-23: scipy_gpd as tail model
**Decision:** scipy_gpd is the tail distribution for all DH/POT models. statsmodels_gpd not considered.
**Why:** scipy_gpd dominates all OOS tail metrics at every threshold with minimal insample/OOS gap. No other distribution comes close.
**Revisit if:** On new data with different tail behavior; or if scipy_gpd produces unreasonable extrapolations.

## 2026-03-23: Covariate sets expanded to 15 (from 7)
**Decision:** Added log-transform variants (logwind, logsdi) and island dummy interactions to covariate grid.
**Why:** Each stage picks its own covariates by OOS performance — no reason to restrict. Log transforms are physically motivated for count models (but redundant for tree models, hence separate exclusion).
**Revisit if:** Overfitting becomes a concern (monitor insample/OOS gaps).

## 2026-03-23: OOS evaluation: 5 seeds × 5 folds = 25 folds per stage
**Decision:** Each stage is evaluated on 25 independent OOS folds using stratified random splits (np.random.default_rng(seed).integers).
**Why:** 5-fold CV with 5 different random seeds gives robust OOS estimates and allows checking seed stability. Single seed would be noisy given small dataset (1383 rows).
**Revisit if:** Computational cost becomes prohibitive; or sample size grows substantially.

## 2026-03-30: Restrict covariate sets to 3
**Decision:** Only consider covariate sets: none, wind_sdi, wind_sdi_basin_island.
**Why:** 15 covariate sets × 4 stages = 15^4 = 50,625 covariate combinations per distribution combo. Explosion is unnecessary — the core scientific question is whether wind+SDI predict mortality, not whether log transforms help. Three sets cover: intercept-only baseline, core predictors, full model with geographic controls.
**Revisit if:** Evidence that log transforms or intermediate sets substantially improve OOS performance.

## 2026-03-30: Never proceed to analysis with failed jobs
**Decision:** If any stage fits fail (missing pkl files, prediction errors, etc.), diagnose and fix before running analysis. Zero tolerance for partial results.
**Why:** Partial results hide problems and produce misleading rankings. A 40% failure rate means the "best" model is just the best among survivors — not the actual best.
**Revisit if:** Never.

## 2026-03-30: Exclude stages with any fold failures from DH comparison
**Decision:** For workflow 558864 (201 DH-only stages), exclude any stage that has even one failed fold from the model comparison. 5091/5226 pkl files created (97.4%); ~5 stages had partial failures due to statsmodels_gamma numerical instability on tail data.
**Why:** Including stages with partial OOS results would bias metrics toward folds that happened to converge. Better to exclude entirely than to compare apples to oranges.
**Revisit if:** Need gamma tail coverage for scientific reasons; would require fixing the distribution wrapper to handle edge cases.

## 2026-03-30: 70% threshold as current direction for expanded sweep
**Decision:** Start with 70% threshold only in expanded covariate grid. Other thresholds (75-95) deferred.
**Why:** TOPSIS and Condorcet analysis of 10,188 models showed threshold=70 dominated all OOS metrics. Higher thresholds consistently underperform. Starting with one threshold keeps computation tractable while we explore covariate space.
**Revisit if:** Final model selection reveals 70% has issues (coverage, calibration); or scientific reviewers question the threshold choice — then run comparison sweep with 75/80.

## 2026-03-30: Exclude poisson and tweedie from bulk stage
**Decision:** Bulk stage only considers NB, gamma, lognormal. Poisson and tweedie excluded.
**Why:** In 10k model comparison, poisson and tweedie bulk models consistently ranked bottom across all metrics. Poisson fails due to overdispersion; tweedie had numerical issues. Neither belongs in the Pareto frontier.
**Revisit if:** Never for poisson (fundamental mismatch). Tweedie only if numerical wrapper is fixed.

## 2026-03-30: Coefficients moved to dedicated module
**Decision:** Coefficient loading/display utilities live in `src/idd_climate_models/tc_models/coefficients.py`, not notebooks.
**Why:** These functions are needed for model interpretation across multiple notebooks and scripts. Having them in a notebook meant copy-paste or notebook imports. Module is the correct location.
**Revisit if:** Never — this is basic code hygiene.

## 2026-03-31: Covariate sets culled to 16 (2^4 factorial)
**Decision:** Use full factorial of 4 features: wind, sdi, basin, island. Drop all log transforms (logwind, logsdi). 16 covariate sets total.
**Why:** Log transforms added complexity without clear benefit. Full factorial is symmetric and interpretable. Reduces model space from 3.4M to 786k combinations while covering all feature subsets.
**Revisit if:** Evidence that log transforms substantially improve OOS performance; would need targeted comparison.

## 2026-03-31: Both IS and OOS metrics required for model selection
**Decision:** Model selection requires both in-sample (IS) and out-of-sample (OOS) metrics. OOS for ranking, IS/OOS gap for overfitting detection.
**Why:** OOS metrics alone can't detect overfitting. A model that performs well OOS but has huge IS/OOS gap may be unstable. Need both to make informed selection.
**Revisit if:** Never — this is basic model selection hygiene.

Files:
- OOS only: `dh_exhaustive_expanded.csv` (produced by `analyze_dh_exhaustive.py`)
- OOS + IS: `dh_exhaustive_with_is.csv` (produced by `add_insample_metrics.py`)
