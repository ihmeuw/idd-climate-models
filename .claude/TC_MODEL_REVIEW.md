# Storm Death Model: Codebase Review for idd-tc-mortality

Generated: 2026-04-03. Cross-references `src/idd_climate_models/tc_models/` against the
correct-logic specification document. This file is read-only orientation for the new
repo build. Do not move or modify source files based on it.

---

## 1. What the current code does

### Pipeline overview

The goal is to predict storm-level tropical cyclone mortality (death counts per
storm-country event) as a function of storm intensity (wind speed), country development
status (SDI), geography (basin, island indicator), and population exposure.

**Data:** ~1,383 storm-country observations. Most rows have zero deaths. A small subset
has extreme deaths (heavy right tail). This distributional shape motivates the
double-hurdle structure.

**Model structure — the Double Hurdle (DH):**

```
E[Y | X] = P(Y > 0 | X)                          [Stage 1: s1]
           × ( P(low | Y>0, X) × E[Y | bulk, X]  [Stage 3: dh_bulk]
             + P(high | Y>0, X) × E[Y | tail, X] [Stage 4: tail]  )
                     ^
                     [Stage 2: pos_binary]
```

The threshold separating bulk from tail is estimated as a percentile of death rates
(v3) or death counts (v1/v2) among training positives.

**Pipeline steps (as of the v3 work):**

1. **Stage grid** (`stage_grid_dh_v3.py`) — enumerate atomic fits: 3 covariate sets ×
   {s1, pos_binary, dh_bulk, tail} stage types × {70,75,80,85,90,95} thresholds ×
   per-stage distributions = ~120 unique stage specs.

2. **Stage fitting** (`run_one_stage.py`, dispatched via `orchestrate_dh_v3.py` →
   jobmon) — each stage is fit once in-sample and 25× OOS (5 seeds × 5 folds). Models
   persisted as `{stage_id}_insample.pkl` and `{stage_id}_seed{N}_fold{K}.pkl`.

3. **Assembly and evaluation** (`analyze_dh_exhaustive_v3.py`) — enumerate all valid
   combinations of stages into full DH models, load cached fold predictions, assemble
   via the DH formula, compute OOS and IS metrics, write `dh_exhaustive_v3.csv`.

4. **Model selection** — multi-criteria ranking in `model_selection.py` (Borda, Pareto,
   TOPSIS) applied to the CSV.

5. **Diagnostic plots** (`stage_plots.py`) — `StagePlotter.vet_stage()` / `predict_df()`
   for inspecting a selected model.

**Supporting infrastructure:**
- `distributions/` — 15 distribution modules, each with
  `fit(X, y, exposure, task)` / `predict(fitted, X, exposure, task)`. Uniform interface.
- `features.py` — `build_X(df, covars)` with one-hot basin dummies; `align_X` for
  test-time column alignment.
- `cache.py` — deterministic MD5 hash IDs, per-stage JSON result files (avoids
  concurrent write conflicts), pickle model files.
- `metrics.py` — `calc_metrics`, `calc_tail_metrics`, `calc_binary_metrics`.

---

## 2. What structure is worth keeping

### Keep as-is

- **Stage-based architecture** (DECISIONS.md 2026-03-23): each unique fit happens once;
  stages are shared across DH combinations. Correct and efficient.
- **Uniform distribution interface** (`fit`/`predict` with `exposure` and `task` args):
  well-designed. All 15 distributions are interchangeable.
- **Deterministic stage ID via MD5 hash**: makes caching reproducible.
- **Per-stage JSON result files** rather than a single shared manifest: correct decision
  for parallel writes.
- **`build_X` + `align_X` pattern**: handles one-hot basin dummies safely between train
  and test. (See C6 for the `drop_first` bug.)
- **Multiple fallback chains in `statsmodels_nb.py`**: appropriate for numerical GLM
  fitting.
- **`calc_metrics` signature** (obs, pred, exp → dict): clean.
- **OOS via 5 seeds × 5 folds = 25 folds per stage**: correct level of robustness for
  dataset size. (See C7 for the stratification gap.)
- **Coverage metrics** (`cov_1` through `cov_20`): appropriate given that tail
  performance is the scientific goal.

### Keep with modification

- `run_one_stage.py::_fit_and_eval` — the core fitting logic is correct for v3
  (rate-based thresholds throughout). Worth extracting as the canonical function; the
  surrounding CLI scaffolding is fine. See C1–C5 for what needs to change inside it.
- `analyze_dh_exhaustive_v3.py::build_aligned_cache` — the pre-concatenation trick for
  fast assembly of 25-fold predictions is a good pattern. The implementation has one
  flaw (see P7 in §4 below).

---

## 3. Conflicts between old code and correct logic

For each "worth keeping" item, specific functions or logic that conflict with the
specification document. Severity: **Critical** = model is theoretically wrong;
**Significant** = wrong outcome/scale; **Moderate** = biases results; **Minor** =
clean-up required; **Gap** = feature does not exist.

---

### C1 — Link function wrong for S1/S2 (Critical)

**Spec:** S1 and S2 use Binomial GLM with **complementary log-log (cloglog)** link.
Theoretical basis: `P(deaths=0) = (1−p)^N ≈ exp(−N·p)`, so the log probability of zero
deaths scales linearly with log(N). Cloglog is the natural link for this process.

**Conflict:**
- `distributions/statsmodels_logistic.py:fit` — `families.Binomial()` uses the default
  logit link. No cloglog anywhere in the file.
- `distributions/sklearn_logistic.py:fit` — `LogisticRegression(...)` with logit link.
- Both are called in `run_one_stage.py:_fit_and_eval` for `stage_type == 's1'` and
  `stage_type == 'pos_binary'`.

This is a theoretically incorrect choice, not a minor numerical difference. Logit and
cloglog produce different predictions whenever exposure varies across storms, which it
always does. The sklearn logistic wrapper can be discarded entirely for the new repo.

---

### C2 — Exposure completely absent from S1/S2 (Critical)

**Spec:** `log(exposed)` must appear in S1 and S2 as an **offset with coefficient fixed
at 1**. It must never be freely estimated.

**Conflict:**
- `run_one_stage.py:_fit_and_eval`, `stage_type == 's1'` (line 43):
  `build_X(train, covars, include_log_exp=False)` — exposure excluded.
- `run_one_stage.py:_fit_and_eval`, `stage_type == 'pos_binary'` (line 79): same.
- `distributions/statsmodels_logistic.py:fit` — no `offset` parameter accepted or
  passed to GLM.
- `stage_plots.py:predict_df`, `stage_key == 's1'` (line 437):
  `build_X(data, cov, include_log_exp=False)`.
- `stage_plots.py:predict_df`, `stage_key == 's2'` (line 441): same.

Without the offset, a storm hitting a population of 10,000 and a storm hitting
10,000,000 are treated identically in S1/S2. The intercept absorbs a mixture of effects
and the coefficients are uninterpretable.

---

### C3 — Bulk outcome is counts, not death_rate (Significant)

**Spec:** Bulk model outcome is `death_rate = deaths / exposed`, among storms where
`0 < death_rate < X`.

**Conflict:**
- `run_one_stage.py:_fit_and_eval`, `stage_type == 'dh_bulk'` (line 104):
  `y_tr = train_bulk['total_deaths'].values` — raw counts.
- `structures/double_hurdle.py:fit` (line 84): `y_pos[bulk_mask]` — raw counts.
- `analyze_dh_exhaustive_v3.py:_predict_insample_stage`, `stype == 'dh_bulk'` (line
  224): model stored was fitted on counts, returns counts.
- `add_insample_metrics.py:predict_insample_stage`, `stype == 'dh_bulk'` (line 133):
  `dm.predict(fitted, X, exposure=bulk_data['exposed_population'].values, ...)` —
  returns counts.

Consequence: the bulk model's linear predictor conflates storm lethality with population
size. Two bulk storms with identical wind speeds and SDI but 10× different exposed
population get different predicted counts even before any covariate effect.

---

### C4 — Bulk and tail exposure is an offset, not a covariate (Significant)

**Spec:** For all rate-scale bulk and tail models (beta, scaled logit, Gamma, lognormal
for bulk; GPD, Gamma, lognormal for tail): `log(exposed)` is a **covariate** (included
in X) plus `var_weights=exposed` to downweight imprecise small-N rate estimates.
Exposure as an offset is reserved for count-outcome models (NB tail only).

**Conflict — distribution modules:**
- `distributions/statsmodels_gamma.py:fit` (line 27): `offset=log_exp` — fixed offset.
- `distributions/statsmodels_gamma.py:predict` (line 40):
  `offset=safe_log_exp(exposure)`.
- `distributions/statsmodels_nb.py:fit` (line 36):
  `NegativeBinomial(..., offset=offset)` — offset in count-scale model. Correct for the
  NB tail case per spec, but the module is also used for bulk, where the spec requires
  rate outcome with var_weights instead.
- `distributions/statsmodels_lognormal.py:fit` (line 41):
  `log_rate = np.log(y) - safe_log_exp(exposure)` implicitly treats exposure as offset.
  Uses `weights=y` (count-proportional) instead of `var_weights=exposed` as spec
  requires.
- `distributions/statsmodels_lognormal.py:predict` (line 54):
  `exp(log_rate + safe_log_exp(exposure) + sigma²/2)` — adds log(exposure) back,
  returning counts.
- `features.py:build_X` — `include_log_exp=True` adds `log_exp` as a column, which is
  the mechanism for passing it as a covariate. But `var_weights=exposed` has no path
  into the current distribution interface — no distribution module accepts or uses
  `var_weights`.

The lognormal wrapper is closest to the spec (it implicitly works on log(rate)) but
returns counts and uses the wrong weights.

---

### C5 — Tail outcome is counts, not excess rate (Significant)

**Spec:** For GPD, Gamma, and lognormal tail models, the outcome is
`death_rate − X` (excess death rate). Predictions must add `X` back before combining.
For NB, the outcome remains counts with log(exposed) as offset — the spec explicitly
permits this, with predictions divided by exposed at combine time.

**Conflict:**
- `run_one_stage.py:_fit_and_eval`, `stage_type == 'tail'` (line 125):
  `y_tr = train_tail['total_deaths'].values` — counts, not excess rate.
- `distributions/scipy_gpd.py:fit` (line 33): `z = y - threshold` where `y` is counts
  and `threshold` is a count-scale threshold. Spec says `z = death_rate − X` where X
  is a rate-scale threshold.
- `distributions/scipy_gpd.py` (line 46) — scale model is
  `log(scale_i) = Xβ + log(exposure_i)`. This makes scale proportional to exposure,
  appropriate for count exceedances. For rate exceedances, log(exposure) should be a
  covariate, not baked into the scale definition.
- `distributions/scipy_gpd.py:predict` (line 74): returns
  `threshold + scale / (1 − shape)`. When threshold and scale are count-scale, this is
  correct for counts. For the new repo, this should return a rate-scale prediction;
  adding X (rate threshold) back is the caller's responsibility per spec Implementation
  Note 6.

The GPD module is architecturally the closest to correct (it fits on exceedances and has
a covariate-dependent scale), but the entire scale of the outcome is wrong.

---

### C6 — Basin dummies: all K levels retained (Moderate)

**Spec:** "Drop one reference category."

**Conflict:**
- `features.py:build_X` (line 114):
  `pd.get_dummies(..., drop_first=False)` — generates all K basin dummies. With an
  intercept in the model, this is rank-deficient. Statsmodels GLM resolves this silently
  via QR (dropping a column internally), but the dropped column is
  implementation-defined, not user-controlled. The `align_X` function then aligns to
  the training columns, which may differ from what was actually used in the fit.

**Knock-on effect:** `coefficients.py:load_model_coefficients` and
`stage_plots.py:_predict` both reconstruct `feature_names` from the covariate string
and then reindex columns to that list. If statsmodels silently dropped a basin dummy
during fitting, the `_X_cols` attribute stored on the result object will be inconsistent
with what `build_X` produces.

---

### C7 — OOS splits are random, not stratified by basin (Moderate)

**Spec:** "OOS evaluation should stratify folds by basin to avoid geographic leakage."

**Conflict:**
- `run_one_stage.py:main` (line 222):
  `fold_ids = rng.integers(0, k_folds, size=n)` — uniform random assignment.
- `analyze_dh_exhaustive_v3.py:_predict_stage` (line 103): same.
- `analyze_dh_exhaustive.py:_predict_stage` (line 109): same.
- `stage_plots.py:predict_df_oos` (line 510): same.
- `engine.py:_run_one_split`: same.

With random splits, training and test folds share storms from the same basin. Since
storms in the same basin have correlated covariates and outcomes (shared environmental
drivers, regional GDP, healthcare infrastructure), OOS metrics from random splits
overestimate true generalization performance.

---

### C8 — `scipy_gpd.py` has no dense `hess_inv` for uncertainty propagation (Moderate)

**Spec:** GPD uncertainty uses `result.hess_inv` from the BFGS optimizer as the
covariance matrix, dropped into the same `draw_coefficients` pattern as GLMs.

**Conflict:**
- `distributions/scipy_gpd.py:fit` — uses `scipy.optimize.minimize` with
  `method='L-BFGS-B'`. The L-BFGS-B result object does **not** expose a reliable
  Hessian inverse. BFGS (`method='BFGS'`) returns `res.hess_inv` as a dense matrix;
  L-BFGS-B returns a `LbfgsInvHessProduct` (implicit representation), not a dense
  covariance matrix.
- No uncertainty output is stored in `GPDResult` at all.

To support the spec's uncertainty pattern, the GPD optimizer needs to be `method='BFGS'`
and `GPDResult` needs to store `hess_inv` explicitly.

---

### C9 — `calc_tail_metrics`: threshold is count-scale, not rate-scale (Minor)

**Spec:** Threshold X is a percentile of `death_rate` among positive storms (rate
scale).

**Conflict:**
- `metrics.py:calc_tail_metrics` (line 135): `tail_mask = observed > threshold` where
  `threshold` is described in the docstring as "the death-count threshold u". The
  function takes counts for `observed` and `predicted`, and the tail is defined by a
  count threshold.

If this function is reused in the new repo, the signature and docstring need to reflect
a rate-scale threshold: `tail_mask = (observed / exposed) > threshold`.

---

### C10 — Missing distribution families (Gap)

The spec defines the full model grid as:
- **Bulk:** beta, scaled_logit, gamma, lognormal — **beta and scaled_logit do not exist
  anywhere in the codebase.**
- **Tail:** gpd, gamma, lognormal, nb — all present, but GPD needs rework per C5.

`distributions/__init__.py` registry would need to be extended. The beta and scaled
logit modules are entirely new work; they cannot be adapted from anything existing.

---

### C11 — No uncertainty quantification anywhere (Gap)

**Spec:** All GLM models expose coefficient draws via `result.params` and
`result.cov_params()`, propagated jointly through all four stages per draw:

```python
def draw_coefficients(result, n_draws=1000):
    mean = result.params
    cov = result.cov_params()
    return np.random.multivariate_normal(mean, cov, size=n_draws)
```

**Conflict:**
- No distribution module, engine function, or analysis script implements
  `draw_coefficients` or `predict_with_uncertainty`.
- `statsmodels_gamma.py`, `statsmodels_nb.py`, `statsmodels_lognormal.py` all return
  statsmodels result objects that expose `.params` and `.cov_params()`, so the interface
  is available but no caller uses it.
- `_LogNormalResultSM` in `statsmodels_lognormal.py` wraps the WLS result and exposes
  `.params`, `.bse`, `.pvalues` — correct building block, but `.cov_params()` is not
  exposed on the wrapper class directly.
- `GPDResult` stores only `params` — no covariance. See C8.

The entire uncertainty pipeline is new work.

---

### Summary table

| ID | Description | Severity | Files affected |
|----|-------------|----------|----------------|
| C1 | Logit link instead of cloglog for S1/S2 | Critical | `statsmodels_logistic.py:fit`, `run_one_stage.py:_fit_and_eval` |
| C2 | No exposure offset in S1/S2 | Critical | `statsmodels_logistic.py:fit`, `run_one_stage.py:_fit_and_eval`, `stage_plots.py:predict_df` |
| C3 | Bulk outcome is counts, not rate | Significant | `run_one_stage.py:_fit_and_eval`, `double_hurdle.py:fit` |
| C4 | Exposure as offset not covariate+weights | Significant | `statsmodels_gamma.py`, `statsmodels_nb.py`, `statsmodels_lognormal.py`, `run_one_stage.py:_fit_and_eval` |
| C5 | Tail outcome is counts, not excess rate | Significant | `scipy_gpd.py:fit/predict`, `run_one_stage.py:_fit_and_eval` |
| C6 | Basin dummies `drop_first=False` | Moderate | `features.py:build_X` |
| C7 | OOS splits random, not basin-stratified | Moderate | `run_one_stage.py:main`, `analyze_dh_exhaustive_v3.py:_predict_stage`, `stage_plots.py:predict_df_oos` |
| C8 | GPD uses L-BFGS-B, no dense `hess_inv` | Moderate | `scipy_gpd.py:fit`, `GPDResult` class |
| C9 | `calc_tail_metrics` uses count threshold | Minor | `metrics.py:calc_tail_metrics` |
| C10 | Beta and scaled logit distributions missing | Gap | No existing file |
| C11 | No uncertainty quantification | Gap | All distribution modules, all analysis scripts |

---

## 4. Known logic bugs in the existing pipeline

These are internal inconsistencies in the current code, separate from the spec
conflicts above.

### P1 (Critical): `predict_df` applies bulk and tail models to all rows

`stage_plots.py:predict_df` (lines 436–454) calls the bulk and tail models on all 1,383
rows using `exposure=1`. The bulk model was trained only on positive-death rows with
rate ≤ u. Applied to zero-death rows, it extrapolates outside its training distribution
and may return high rates for high-wind-speed zeros. This inflates total predicted
counts and is the most likely root cause of the `pred_obs_ratio = 4.56` observed in the
last session.

The analyze scripts (`analyze_dh_exhaustive_v3.py`) do the opposite — predicting only
within each relevant subset and filling 0 elsewhere. These two assembly approaches give
different totals and are not comparable.

### P2 (Critical): Count-based threshold in v2 analyze script vs rate-based in run_one_stage.py

`analyze_dh_exhaustive.py` computes `u` on raw death counts during assembly, but
`run_one_stage.py` computed it on death rates when fitting the models. The OOS metrics
in `dh_exhaustive_expanded.csv` and `dh_exhaustive_with_is.csv` (v2 outputs) are
therefore evaluating the wrong test rows against the wrong models. **v3 fixed this.**
The v2 result files are contaminated.

### P3 (Significant): `structures/double_hurdle.py` uses count-based threshold

`double_hurdle.py:fit` (line 62): `u = float(np.percentile(y_pos, threshold_pct))` —
counts. Not used in the v3 pipeline but still in `__init__.py`'s public API.

### P4 (Significant): Three inconsistent `assemble_dh_insample` implementations

`add_insample_metrics.py`, `analyze_dh_exhaustive.py`, and `analyze_dh_exhaustive_v3.py`
each implement `assemble_dh_insample` with different threshold definitions, prediction
spaces, and fallback values. `fillna(0.5)` for missing `p_pos` is wrong — it implies
50% probability of death for any storm.

### P5 (Significant): `stage_plots.py` hardcodes v2 model directory

```python
DH_MODELS_DIR = Path('.../stage_results_dh_v2/models')
```

Any call to `StagePlotter` or `predict_df` on a v3 model row loads from the wrong
directory (or raises `FileNotFoundError`).

### P6 (Moderate): `predict_df_oos` — `cnt` only tracks s1 folds

`stage_plots.py:predict_df_oos` (line 547): `cnt[te_idx] += 1` only in the s1 branch.
All four prediction arrays are divided by `safe_cnt` at the end. If an s2 or bulk fold
fails while s1 succeeds, the average is computed with a wrong denominator.

### P7 (Moderate): `build_aligned_cache` fills missing predictions with 0 silently

`analyze_dh_exhaustive_v3.py:build_aligned_cache` (line 363):
`preds_series.reindex(test_idx).fillna(0.0)`. If a positive-death row is in `test_idx`
but missing from `preds_series` due to a failed prediction, it silently gets
`p_high = 0` rather than NaN, biasing the assembly without any warning.

### P8 (Moderate): Poisson included as tail distribution (v3)

`stage_grid_dh_v3.py`: `TAIL_DISTS = ['statsmodels_gamma', 'statsmodels_lognormal',
'statsmodels_poisson']`. Tail data consists of extreme high-death events with extreme
overdispersion. Poisson was excluded from bulk for this reason (DECISIONS.md
2026-03-30). The argument applies with more force to the tail.

### P9 (Minor): Version sprawl — 3 parallel output directories

`stage_results_dh/`, `stage_results_dh_v2/`, `stage_results_dh_v3/` contain
incompatible artifacts. Multiple orphaned scripts point to different directories. The
relationship between versions — specifically which changed the threshold definition — is
documented only in `DECISIONS.md`, not in the code.

### P10 (Minor): Single stages + tweedie/gamma failure on zeros

DEAD_ENDS.md (2026-03-30): `statsmodels_tweedie` fails on single stages with y=0 rows.
The fix was identified but never applied. `stage_grid.py` still includes these
distributions for single stages.
