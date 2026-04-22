# Task: Rate-Based Threshold Re-Run of DH Model Pipeline

## CRITICAL CONTEXT: Two Separate Investigations

**Investigation 1 (COMPLETED, FLAWED):** Count-based threshold. All existing results in `stage_results_dh_v2/` use `np.percentile(total_deaths, threshold_pct)` — the threshold is on raw death counts. All fitted models, pkl files, metrics CSVs, and TOPSIS results in that directory are from this investigation. **Do NOT mix these artifacts with Investigation 2.**

**Investigation 2 (THIS TASK):** Rate-based threshold. The threshold should be `np.percentile(total_deaths / exposed_population, threshold_pct)`. All outputs must go to a NEW directory: `stage_results_dh_v3/`. No code should ever read from `v2` for `v3` analysis.

**Why this matters:** 27% of storms are classified differently (bulk vs tail) depending on count vs rate threshold. The bulk/tail models are fit in rate space (with exposure offset), so a count-based threshold is inconsistent. Using v2 models in v3 diagnostics would silently produce wrong comparisons.

---

## Project Location

**Repo:** `/ihme/homes/bcreiner/repos/idd-climate-models`  
**Conda env:** `idd-climate-models` (activate with `source /ihme/homes/bcreiner/miniconda/etc/profile.d/conda.sh && conda activate idd-climate-models`)

## What This Project Does

A Double Hurdle (DH) model for tropical cyclone mortality with 4 stages:
- **s1** (logistic): P(storm causes any deaths) — binary 0/1. No exposure offset.
- **s2** (logistic): P(storm is high-severity | deaths > 0) — binary 0/1. No exposure offset. **This is where the threshold matters: currently `total_deaths > u`, should be `death_rate > u_rate`.**
- \**bulk** (NB/gamma/lognormal GLM): E[deaths | deaths > 0 AND death_rate ≤ u_rate]. Fit with `exposure=exposed_population` (log-offset). **The data subset changes with rate threshold.**
- **tail** (gamma/GPD): E[deaths | death_rate > u_rate]. Fit with exposure. **The data subset changes with rate threshold.**

## What Needs to Change

### 1. Threshold computation (in `run_one_stage.py`)

Current (WRONG for rate models):
```python
u = float(np.percentile(train_pos['total_deaths'].values, threshold_pct))
```

New (rate-based):
```python
train_rates = train_pos['total_deaths'].values / train_pos['exposed_population'].values
u_rate = float(np.percentile(train_rates, threshold_pct))
```

Then s2 outcome becomes:
```python
train_rates = train_pos['total_deaths'].values / train_pos['exposed_population'].values
y_tr = (train_rates > u_rate).astype(int)
```

And bulk/tail splits become:
```python
train_rates = train_pos['total_deaths'].values / train_pos['exposed_population'].values
train_bulk = train_pos[train_rates <= u_rate]
train_tail = train_pos[train_rates > u_rate]
```

**IMPORTANT:** s1 is unaffected (it's just deaths > 0, no threshold involved). The s1 stages from v2 are still valid, but for cleanliness, re-fit everything into v3.

### 2. Output directory

All outputs go to:
```
/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/stage_results_dh_v3/
├── models/          # pkl files
├── results/         # assembled model CSV
├── stage_logs/      # JSON completion logs + exhaustive CSV
├── logs/            # jobmon logs
└── stage_grid.json  # the grid definition
```

Update `orchestrate_dh_expanded.py` to use `stage_results_dh_v3/`.  
Update `analyze_dh_exhaustive.py` to use `stage_results_dh_v3/`.  
Update `coefficients.py` DH_MODELS_DIR to `stage_results_dh_v3/models/`.

### 3. Stage grid: constrained covariate sets + threshold sweep

**Covariate sets (3):** `none`, `wind_sdi`, `wind_sdi_basin_island`

**Thresholds:** 70, 75, 80, 85, 90, 95

**Distributions:**
- s1: `statsmodels_logistic`
- s2: `statsmodels_logistic`
- bulk: `statsmodels_nb`, `statsmodels_gamma`, `statsmodels_lognormal`, `sklearn_rf`, `sklearn_xgb`
- tail: `statsmodels_gamma`, `scipy_gpd`, `statsmodels_lognormal`, `sklearn_rf`, `sklearn_xgb`

**Stage count estimate:**
- s1: 3 covars × 1 dist = 3 stages (no threshold dependency)
- s2: 3 covars × 1 dist × 6 thresholds = 18 stages
- bulk: 3 covars × 5 dists × 6 thresholds = 90 stages
- tail: 3 covars × 5 dists × 6 thresholds = 90 stages
- **Total: ~201 stages**

Each stage: 1 insample fit + 25 OOS folds (5 seeds × 5 folds) = 26 fits per stage = ~5,226 pkl files

**Model assembly constraint (IMPORTANT):** When assembling DH models from individual stages for comparison, only combine stages that share the same covariate set. No cross-covariate mixing (e.g., s1=`wind_sdi` + bulk=`wind_sdi_basin_island` is not evaluated). This keeps the comparison interpretable and the model count tractable. The constraint is applied at the analysis/assembly step — individual stages are still fit independently.

**Goals of this sweep:**
- (a) Find which threshold percentile(s) warrant a deeper dive
- (b) Identify stages/distributions that perform poorly (diagnostic, not just winner selection)

### 4. Key files to modify

- **`src/idd_climate_models/tc_models/stage_grid_dh_expanded.py`** — New grid with 3 covariate sets, 6 thresholds, ML models included
- **`src/idd_climate_models/tc_models/run_one_stage.py`** — Change threshold from count-based to rate-based (lines ~76, ~96, ~121)
- **`src/idd_climate_models/tc_models/orchestrate_dh_expanded.py`** — Change OUTPUT_DIR to `stage_results_dh_v3/`
- **`src/idd_climate_models/tc_models/analyze_dh_exhaustive.py`** — Change `_DH_ROOT` to `stage_results_dh_v3/`
- **`src/idd_climate_models/tc_models/coefficients.py`** — Change `DH_MODELS_DIR` to `stage_results_dh_v3/models/`
- **`src/idd_climate_models/tc_models/analyze_dh_comparison.py`** — Same threshold logic change if used

### 5. What NOT to change

- `features.py` — covariate building is fine
- `data.py` — data loading is fine
- `model_selection.py` — TOPSIS/Borda pipeline is fine
- `model_query.py` — querying is fine (just needs new data)
- The v2 directory — leave it intact for reference

### 6. Verification steps

After fitting completes:
1. Check that the threshold values are different from v2 (they should be since they're rates not counts)
2. Check that the bulk/tail split counts differ from v2
3. Verify that `exposed_population` is never zero in the data (would cause division by zero)
4. Run `analyze_dh_exhaustive.py` and confirm output goes to `stage_results_dh_v3/stage_logs/`
5. Update notebook paths to point to v3

### 7. Design decisions (confirmed)

- **3 covariate sets only** — full 2^4 factorial is too many; constrained to `none`, `wind_sdi`, `wind_sdi_basin_island` for tractability
- **Full 6-threshold sweep** — need evidence across all thresholds for justification, not just the winner
- **No cross-covariate DH models** — s1/s2/bulk/tail must all use the same covariate set in assembled models
- **s2 stays exposure-free logistic** — it's a binary classifier on whether death_rate > u_rate; no exposure offset needed or appropriate
- **`statsmodels_lognormal`** not `scipy_lognormal`
- **All v3 results separate from v2** — v2 left intact for reference

---

## Files Reference

### Data columns available from `load_tc_data()`:
`total_deaths`, `max_wind_speed`, `sdi`, `basin`, `island_nation`, `death_y_n`, `exposed_population`

### How stages are fit (in `run_one_stage.py`):
- Logistic stages: `statsmodels.Logit(y, X).fit()`
- Count stages: `dist_mod.fit(X, y, exposure=exposed_population)` — exposure is passed as log-offset
- ML stages: `sklearn_rf`/`sklearn_xgb` fit on features including `log_exp` column

### How `build_X` works:
- `build_X(df, 'wind_sdi')` → DataFrame with columns `wind_speed_var`, `sdi_var`
- `build_X(df, 'wind_sdi_basin_island')` → `wind_speed_var`, `sdi_var`, `basin_EP`, `basin_NA`, ..., `island_nation`
- `build_X(df, 'none')` → empty DataFrame (intercept-only model)
- For ML models, `include_log_exp=True` adds a `log_exp` column
