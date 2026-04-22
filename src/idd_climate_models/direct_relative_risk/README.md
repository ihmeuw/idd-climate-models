# Tropical Cyclone Mortality Prediction Toolkit

Modeling framework for tropical cyclone death prediction using parametric models
(GLM, Negative Binomial, Gamma, Lognormal, ZIP/ZINB), machine learning (Random Forest, XGBoost),
and threshold-split models (POT, Double Hurdle).

**Current focus:** Double Hurdle (DH) models — best structure as of 2026-03-23 (mae_rate 1.28).

## Package Structure

All modeling code lives in the `tc_models` package:

```
tc_models/
    __init__.py           # Public API exports
    constants.py          # Paths, default parameters
    data.py               # load_tc_data(), prepare_tc_data()
    engine.py             # run_model_evaluation(), run_multiple_models()
    features.py           # build_X(), covariate set definitions
    metrics.py            # calc_metrics(), calc_tail_metrics()
    cache.py              # Stage result caching
    distributions/        # One module per distribution family
        statsmodels_logistic.py
        statsmodels_nb.py
        statsmodels_gamma.py
        statsmodels_lognormal.py
        statsmodels_poisson.py
        statsmodels_tweedie.py
        statsmodels_zip.py
        statsmodels_zinb.py
        scipy_gpd.py      # GPD tail — dominates tail stage
        sklearn_rf.py     # Random Forest
        sklearn_xgb.py    # XGBoost
    structures/           # Model assembly logic
        single.py
        hurdle.py
        pot.py
        double_hurdle.py  # Current best structure
```

### Stage Pipeline (Current)

```
stage_grid.py           # Generate 2011 unique stage specs
run_one_stage.py        # Fit a single stage (called by orchestrator)
orchestrate_stages.py   # Submit all stages to cluster via jobmon
analyze_stages.py       # Assemble stages → full models, rank by structure
analyze_dh.py           # DH deep-dive: per-threshold tables, coverage profiles
```

### Deprecated (Spec-Based Era — Do Not Use)

```
spec_grid.py                    # Old: 3678 redundant specs
run_one_spec.py                 # Old: full model per spec
orchestrate_tc_comparison.py    # Old: spec-based orchestrator
direct_relative_risk/tc_comparison.py  # Old: monolithic comparison script
```

### Primary Analysis Notebook

- **`notebooks/dh_model_analysis.ipynb`** — Candidate ranking, coverage curves, coefficient tables

## Quick Start

```python
from idd_climate_models.tc_models import load_tc_data

tc_df = load_tc_data()
```

### Explore results (primary workflow)
```bash
jupyter notebook notebooks/dh_model_analysis.ipynb
```

### Run stage analysis scripts (after stages are fit)
```bash
python tc_models/analyze_stages.py   # Rank all structures
python tc_models/analyze_dh.py       # DH component deep-dive
```

### Fit new stages (cluster required)
```bash
python tc_models/stage_grid.py          # Generate stage specs
python tc_models/orchestrate_stages.py  # Submit to cluster via jobmon
```

### Programmatic (single model evaluation)
```python
from idd_climate_models.tc_models import run_model_evaluation, load_tc_data

tc_df = load_tc_data()
result = run_model_evaluation(
    tc_df,
    {'type': 'hurdle', 'family': 'glm_sm', 'covars': 'wind_sdi_basin'},
    seeds=[123, 456, 789],
    k_folds=5,
)
print(result['oos_summary'])
```

## Model Specifications

### Model Types
- `'hurdle'` — Two-stage: binary (death or not) → count (on non-zeros)
- `'single'` — Single-stage: direct count prediction

### Distributions

| Module | Description | Stage Roles |
|--------|-------------|-------------|
| `statsmodels_logistic` | Logistic regression | s1, pos_binary (s2) |
| `statsmodels_nb` | Negative Binomial | pos_count, dh_bulk, tail, single |
| `statsmodels_gamma` | Gamma GLM | pos_count, dh_bulk, tail, single |
| `statsmodels_lognormal` | WLS on log(rate) | pos_count, dh_bulk, tail, single |
| `statsmodels_poisson` | Poisson GLM | pos_count, dh_bulk, tail, single |
| `statsmodels_tweedie` | Tweedie GLM | pos_count, dh_bulk, tail, single |
| `statsmodels_zip` | Zero-inflated Poisson | single |
| `statsmodels_zinb` | Zero-inflated NB | single |
| `scipy_gpd` | GPD via scipy | tail (**best tail distribution**) |
| `sklearn_rf` | Random Forest (500 trees) | s1, s2, bulk, single |
| `sklearn_xgb` | XGBoost (100 rounds) | s1, s2, bulk, single |

**Redundant (excluded from production analysis):** `sklearn_logistic`, `sklearn_poisson`, `sklearn_tweedie`, `sklearn_lognormal` — near-identical to their statsmodels counterparts.

To add a new distribution: implement `fit()` and `predict()` in `tc_models/distributions/`, then add to the stage grid in `stage_grid.py`.

### Covariate Sets

| Set | Variables |
|-----|-----------|
| `wind_sdi` | max_wind_speed + sdi |
| `wind_sdi_basin` | + basin dummies |
| `wind_sdi_island` | + is_island |
| `wind_sdi_basin_island` | + basin + is_island |
| `wind_sdi_year` | + data_year |
| `wind_sdi_basin_year` | + basin + data_year |
| `wind_sdi_island_year` | + is_island + data_year |
| `wind_sdi_basin_island_year` | all covariates |

Old-style names (`base`, `basin`, `island_basin`, etc.) are accepted via aliases.

## Output Metrics

### Rate-Based (Primary)
- `mae_rate` — Mean Absolute Error (deaths per 100k)
- `rmse_rate` — Root Mean Squared Error (per 100k)
- `cor_rate` — Correlation (non-zero events only)
- `skill_mae_rate` / `skill_rmse_rate` — Improvement over baseline

### Count-Based
- `mae_count` / `rmse_count` / `cor_count` — Raw death counts

### Other
- `mae_log` — MAE on log(deaths + 1) scale
- `zero_acc` — Accuracy classifying zero vs non-zero events

## Performance Notes

- **In-sample only**: Set `k_folds=0` for fast screening
- **Test mode**: `TEST_MODE = True` in tc_comparison.py for 1 seed x 2 folds
- **Full mode**: 5 seeds x 5 folds = 25 test sets per model

## Contact

Bobby Creiner
University of Washington
bcreiner@uw.edu
