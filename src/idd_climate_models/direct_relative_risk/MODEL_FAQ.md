# Tropical Cyclone Model Framework - FAQ

Common questions and answers about using the TC mortality modeling framework.

> **Architecture note (2026-03-24):** The system has moved from a spec-based architecture (`tc_comparison.py`, ~1,100 specs) to a **stage-based architecture** (`stage_grid.py`, 2011 unique stages). The Double Hurdle (DH) structure is the current focus. The Q&A below is still accurate for the underlying API, but the primary entry point is now `notebooks/dh_model_analysis.ipynb`, not `tc_comparison.py`. See `model_univese.md` for the full architecture overview.

---

## -1. What is the current model architecture?

The system fits each unique model component (stage) exactly once, then assembles full models post-hoc.

**Double Hurdle (DH) structure — four stages:**
1. **S1:** P(Y > 0 | X) — logistic, whether any deaths occurred
2. **S2 (pos_binary):** P(high | Y > 0, X) — covariate-dependent tail split (unlike POT's fixed percentile)
3. **Bulk:** E[Y | bulk, X] — count model for non-tail events
4. **Tail:** E[Y | tail, X] — tail model (scipy_gpd dominates)

**Key scripts:**
```bash
python tc_models/stage_grid.py          # Generate 2011 stage specs
python tc_models/orchestrate_stages.py  # Submit to cluster
python tc_models/analyze_stages.py      # Assemble + rank full models
python tc_models/analyze_dh.py          # DH component deep-dive
```

**Primary notebook:** `notebooks/dh_model_analysis.ipynb`

---

## 0. How do I load the data?

Use `load_tc_data()` from the `tc_models` package:

```python
from idd_climate_models.tc_models import load_tc_data

tc_df = load_tc_data()
```

**Key columns:**
- `total_deaths` - Death count
- `exposed_population` - Exposed population (for offset/rate)
- `max_wind_speed` - Maximum wind speed
- `sdi` - Socio-demographic index
- `basin` - Cyclone basin (factor)
- `data_year` - Year
- `is_island` - Binary indicator (0=not island, 1=island)
- `death_y_n` - Binary indicator (computed automatically)

**Data characteristics:**
- ~68% zero-inflated (most tropical cyclones cause zero deaths)
- Highly overdispersed (large variance relative to mean)
- Heavy right tail (rare catastrophic events with hundreds/thousands of deaths)

---

## 0.5 What's the difference between sklearn and statsmodels Tweedie?

**Two implementations available:**

### Sklearn Tweedie (`tweedie`, `tweedie_int`, `tweedie_island`)
- Uses `sklearn.linear_model.TweedieRegressor`
- **Pros:** Fast, stable, easy to use
- **Cons:** No parameter uncertainty (`.cov_params()` not available)
- **Use case:** Quick screening, when you don't need confidence intervals

### Statsmodels Tweedie (`tweedie_sm`, `tweedie_sm_int`, `tweedie_sm_island`)
- Uses `statsmodels.genmod.GLM` with `families.Tweedie()`
- **Pros:** Full statistical inference via `.cov_params()`, can sample parameter distributions
- **Cons:** Slightly slower, may have convergence issues on difficult data
- **Use case:** Final models, uncertainty quantification, parameter draws

**Recommendation:** Use `tweedie*` for initial screening (k_folds=0), then switch to `tweedie_sm*` for top models if you need parameter uncertainty.

```python
from idd_climate_models.tc_models import run_multiple_models, load_tc_data

tc_df = load_tc_data()

# Quick screening
specs_screening = [
    {'type': 'hurdle', 'family': 'tweedie', 'covars': 'wind_sdi'},
    {'type': 'hurdle', 'family': 'tweedie_int', 'covars': 'wind_sdi_basin'},
]

# Final models with uncertainty
specs_final = [
    {'type': 'hurdle', 'family': 'tweedie_sm', 'covars': 'wind_sdi'},
    {'type': 'hurdle', 'family': 'tweedie_sm_int', 'covars': 'wind_sdi_basin'},
]
```

---

## 1. Can I run model comparison using only in-sample?

**YES — very easy!**

Set `k_folds=0` to skip cross-validation entirely:

```python
from idd_climate_models.tc_models import run_multiple_models, load_tc_data

tc_df = load_tc_data()

results = run_multiple_models(
    tc_df,
    model_specs=your_specs,
    seeds=[42],      # Doesn't matter for in-sample only
    k_folds=0,       # THIS SKIPS OOS!
    verbose=True
)

print(f"Evaluated {len(results['summary'])} models")
top_models = results['summary'].sort_values('mae_rate').head(20)
```

**Workflow:**
1. Run all models with `k_folds=0` (fast, in-sample only)
2. Filter to top 20-50 models by MAE
3. Re-run just those with `k_folds=5` for proper OOS evaluation

---

## 2. Can I use a basin-specific baseline instead of global?

**Moderate Difficulty — needs implementation**

Currently the baseline is a single global rate. A basin-specific baseline would require modifying `calc_metrics()` in `tc_models/metrics.py`.

### Basin-Specific Baseline (manual):
```python
import numpy as np

basin_baseline_rates = (
    tc_df.groupby('basin')
    .apply(lambda x: (x['total_deaths'].sum() / x['exposed_population'].sum() * 1e5))
    .to_dict()
)

baseline_pred_by_basin = np.zeros(len(tc_df))
for basin, rate in basin_baseline_rates.items():
    mask = tc_df['basin'] == basin
    baseline_pred_by_basin[mask] = (rate / 1e5) * tc_df.loc[mask, 'exposed_population'].values
```

This creates a **tougher benchmark** since models must beat basin-specific historical rates.

---

## 3. Can I plot fitted coefficients?

**YES — for statsmodels-based models!**

Models with `.params` or `.coef_` attributes support coefficient extraction:
- `glm_sm*` (statsmodels GLMs)
- `nb*` (Negative Binomial — statsmodels)
- `zip`, `zinb` (zero-inflated — statsmodels)
- `tweedie_sm*` (Tweedie via statsmodels)

Would require adding `return_models=True` option to the engine's internal `_run_one_split()`.

---

## 4. Can I transform continuous variables (log, sqrt)?

**YES — already implemented!**

Log transformations are built-in via the `log_main_covs` parameter:

```python
model_specs = [
    # Log both variables
    {'type': 'hurdle', 'family': 'glm', 'covars': 'wind_sdi_basin',
     'log_main_covs': {'max_wind_speed': True, 'sdi': True}},
    
    # Log wind only
    {'type': 'hurdle', 'family': 'glm', 'covars': 'wind_sdi_basin',
     'log_main_covs': {'max_wind_speed': True, 'sdi': False}},
    
    # No logs (default)
    {'type': 'hurdle', 'family': 'glm', 'covars': 'wind_sdi_basin',
     'log_main_covs': {'max_wind_speed': False, 'sdi': False}},
]
```

---

## 5. Can I sample from the joint distribution of fitted parameters?

**YES — for statsmodels models!**

Statsmodels models expose `.cov_params()`:

```python
params = model.params.values
vcov = model.cov_params()

n_draws = 1000
param_draws = np.random.multivariate_normal(params, vcov, size=n_draws)

predictions_draws = [X @ draw for draw in param_draws]
```

**Models that support this:**
- `glm_sm*`, `nb*`, `tweedie_sm*`, `zip`, `zinb`

**Models that do NOT:**
- `glm`, `tweedie*` (sklearn), `rf`, `xgb` (no covariance matrices)

---

## 6. How hard is it to get skill by basin?

**Moderate — needs implementation**

Would require modifying metric calculation to stratify by basin (~30-40 lines in `tc_models/metrics.py`).

---

## 7. How hard is it to trim years?

**Very easy!**

Just filter the dataframe before running models:

```python
from idd_climate_models.tc_models import load_tc_data, run_multiple_models

tc_df = load_tc_data()
tc_df_filtered = tc_df[
    (tc_df['data_year'] >= 2000) & (tc_df['data_year'] <= 2020)
].copy()

results = run_multiple_models(tc_df_filtered, model_specs, ...)
```

---

## 8. What happened to GPD / ZIGPD?

**Removed as standalone families.**

Investigation showed that the GPD shape parameter consistently hit its upper bound
regardless of constraint width, meaning the true shape is >= 1 and the GPD mean is
infinite — fundamentally wrong for this data.

GPD is now only used as a **tail component** inside the POT (Peaks Over Threshold)
model (`pot_nb`), which is a 3-part model:
1. Logistic binary stage
2. Negative Binomial body (below threshold)
3. GPD tail (above threshold)

Use `run_pot_threshold_sweep()` to evaluate POT at different thresholds.

---

## Summary Table

| Question | Difficulty | Status |
|----------|-----------|--------|
| 0. Load data | Easy | `load_tc_data()` |
| 1. In-sample only | Easy | Set `k_folds=0` |
| 2. Basin baseline | Moderate | Needs implementation |
| 3. Plot coefficients | Easy | Needs `return_models=True` |
| 4. Variable transforms | Easy | `log_main_covs` parameter |
| 5. Parameter draws | Easy | For statsmodels via `.cov_params()` |
| 6. Skill by basin | Moderate | Needs implementation |
| 7. Trim years | Easy | Simple dataframe filter |
| 8. GPD/ZIGPD? | N/A | Removed; GPD only in POT tail |

---

## Additional Resources

- **Package**: `idd_climate_models.tc_models` — all modeling functions
- **Data**: `tc_models/data.py` — `load_tc_data`, `prepare_tc_data`
- **Distributions**: `tc_models/distributions/` — one module per distribution family
- **Structures**: `tc_models/structures/` — `single`, `hurdle`, `pot`, `double_hurdle`
- **Stage pipeline**: `stage_grid.py` → `run_one_stage.py` → `orchestrate_stages.py` → `analyze_stages.py` → `analyze_dh.py`
- **Primary notebook**: `notebooks/dh_model_analysis.ipynb`
- **Architecture overview**: `direct_relative_risk/model_universe.md`

**Deprecated (spec-based era):**
- `tc_comparison.py` — replaced by stage pipeline
- `tc_models/spec_grid.py`, `run_one_spec.py`, `orchestrate_tc_comparison.py` — do not use

---

*Last updated: 2026-03-24*
