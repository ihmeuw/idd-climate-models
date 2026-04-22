# Task: Rewrite stage_plots.py with vet_stage() and vet_model()

## Project Context

**Repo:** `/ihme/homes/bcreiner/repos/idd-climate-models`
**Conda env:** `idd-climate-models` (activate with `source /ihme/homes/bcreiner/miniconda/etc/profile.d/conda.sh && conda activate idd-climate-models`)

## What This Project Does

A Double Hurdle (DH) model for tropical cyclone mortality with 4 stages:
- **s1** (logistic): P(storm causes any deaths) — binary 0/1
- **s2** (logistic): P(storm causes many deaths | deaths > 0) — binary 0/1, threshold = 70th percentile of nonzero deaths
- **bulk** (negative binomial): E[deaths | 0 < deaths ≤ threshold]
- **tail** (gamma): E[deaths | deaths > threshold]

Each stage was fit with different covariate combinations from 4 tokens: `wind`, `sdi`, `basin`, `island`. The winning model (TOPSIS rank 1) uses:
- s1_cov: `wind_sdi_basin_island`
- s2_cov: `sdi_basin`
- bulk_cov: `wind_sdi`
- tail_cov: `none`

## Key Files

### Data & Features
- `src/idd_climate_models/tc_models/data.py` — `load_tc_data()` returns DataFrame with columns including `total_deaths`, `max_wind_speed`, `sdi`, `basin`, `island_nation`, `death_y_n`
- `src/idd_climate_models/tc_models/features.py` — `build_X(df, covars_string)` builds feature matrix from covar tokens. `basin` gets one-hot encoded, `wind` → `max_wind_speed` renamed to `wind_speed_var`, `sdi` → `sdi` renamed to `sdi_var`, `island` → `island_nation`

### Model Loading
- Fitted models (statsmodels results objects) are pickled at: `/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/stage_results_dh_v2/models/{stage_id}_insample.pkl`
- `src/idd_climate_models/tc_models/coefficients.py` — `load_model_coefficients(model_row)` loads all 4 stage pkl files and returns dict with params, pvalues, std_errors, feature_names. `get_feature_names_from_covars(covars_string, sample_data)` reconstructs feature names (the pkl files only have x1, x2... because they were fit with numpy arrays)
- Model rows (pd.Series) have columns: `s1_sid`, `s2_sid`, `bulk_sid`, `tail_sid` (stage IDs for pkl lookup), `s1_cov`, `s2_cov`, `bulk_cov`, `tail_cov` (covar strings), `s1_dist`, `s2_dist`, `bulk_dist`, `tail_dist`

### Existing Plot File (BROKEN — rewrite it)
- `src/idd_climate_models/tc_models/stage_plots.py` — has `StagePlotter` class. The `_get_subset_for_stage()` method works (uses `total_deaths`). The plotting methods exist but have issues with prediction logic. Rewrite this file.

## The Task

Rewrite `src/idd_climate_models/tc_models/stage_plots.py` to add a `vet_stage()` method that produces all vetting plots for ONE stage of ONE model. Then add `vet_model()` that calls it for all 4 stages.

### What `vet_stage(model_row, stage)` should produce:

The covariates we consider are: `max_wind_speed` (continuous), `sdi` (continuous), `basin` (categorical, ~7 levels), `island_nation` (binary categorical).

For EACH covariate (all 4, even if the stage didn't use it):

**If the covariate is continuous (wind, sdi):**
1. **Plot 1 — Overall:** x-axis = covariate, y-axis = outcome. Gray dots = observed data. Colored line = model's mean prediction across a grid of x values (hold other covariates at median/mode). If the stage doesn't use this covariate, the line will be flat — that's fine, show it anyway.
2. **Plot 2 — By basin:** Same as Plot 1, but with separate prediction lines for each basin level (different line styles). Hold other covariates at median.
3. **Plot 3 — By island:** Same as Plot 1, but with separate prediction lines for island=0 and island=1.

**If the covariate is categorical (basin, island_nation):**
1. **Plot 1 — Beeswarm:** x-axis = category levels, y-axis = outcome. Gray jittered dots = observed data. Thick colored horizontal line = model's mean prediction for that category (computed by predicting on all observations in that category).

For binary stages (s1, s2): y-axis is probability (0-1), observed data is 0/1 with jitter.
For continuous stages (bulk, tail): y-axis is deaths, use log scale.

### How to build predictions

```python
# For a grid prediction:
from idd_climate_models.tc_models.features import build_X
from idd_climate_models.tc_models.coefficients import get_feature_names_from_covars
import statsmodels.api as sm

# 1. Create pred_data DataFrame with the x-axis values you want
# 2. Fill in other covariates at reference values (median for continuous, mode for categorical)
# 3. Build X matrix:
X_pred = build_X(pred_data, covars_string, include_log_exp=False)
X_pred = sm.add_constant(X_pred, has_constant='add')
# 4. Align columns with what the model expects:
feature_names = get_feature_names_from_covars(covars_string, sample_data)
for col in feature_names:
    if col not in X_pred.columns:
        X_pred[col] = 0
X_pred = X_pred[feature_names]
# 5. Predict:
y_pred = fitted_model.predict(X_pred)
```

### Data subsets per stage
```python
data = load_tc_data()
nonzero = data[data['total_deaths'] > 0]['total_deaths']
threshold = np.percentile(nonzero, 70)

# s1: all data, y = (total_deaths > 0).astype(int)
# s2: data[total_deaths > 0], y = (total_deaths > threshold).astype(int)
# bulk: data[(total_deaths > 0) & (total_deaths <= threshold)], y = total_deaths
# tail: data[total_deaths > threshold], y = total_deaths
```

### Desired calling convention in notebook:
```python
from idd_climate_models.tc_models.stage_plots import StagePlotter
plotter = StagePlotter()

# All vetting plots for one stage
plotter.vet_stage(topsis_winner, stage='s1')
plotter.vet_stage(topsis_winner, stage='bulk')

# All 4 stages at once
plotter.vet_model(topsis_winner)
```

### Important details
- Death column is `total_deaths`, NOT `deaths`
- The `build_X` function handles basin one-hot encoding and column renaming internally
- For covars='none' (e.g., tail_cov='none'), `build_X` returns empty DataFrame — the model is intercept-only. Predictions are constant. Still plot the data with a flat line.
- `basin` column in raw data has values like 'EP', 'NA', 'NI', 'SI', 'SP', 'WP'
- `island_nation` column is 0/1
