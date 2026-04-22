# Tropical Cyclone Model Universe - Complete Guide

> **Note:** This file previously described the spec-based architecture (~1,100 model specs via `tc_comparison.py`). That system has been replaced. See below for the current stage-based architecture.
> (Filename typo "univese" retained for backward compatibility.)

---

## TL;DR

- **`stage_grid.py`**: Generates the grid of individual stages to fit (2011 unique stages)
- **`orchestrate_stages.py`**: Submits stages to the cluster via jobmon
- **`analyze_stages.py`**: Assembles stages into full models, computes OOS metrics, ranks by structure
- **`analyze_dh.py`**: Deep-dive DH component tables and coverage profiles
- **`notebooks/dh_model_analysis.ipynb`**: Primary analysis notebook — candidate ranking, coverage, coefficients

---

## Architecture: Stage-Based (Current)

The system fits each unique model component exactly once, then assembles full models post-hoc.

### Stage Types

| Stage | Description | Used In |
|-------|-------------|---------|
| `s1` | P(Y > 0 \| X) — binary death/no-death | Hurdle, DH |
| `pos_count` | E[Y \| Y > 0, X] — count on positives | Hurdle |
| `pos_binary` (s2) | P(high \| Y > 0, X) — covariate-dependent tail split | DH only |
| `dh_bulk` | E[Y \| bulk, X] — count below threshold | DH only |
| `tail` | E[Y \| tail, X] — count above threshold | DH, POT |
| `single` | E[Y \| X] — direct count prediction | Single |

### Full Model Structures

| Structure | Stages | Notes |
|-----------|--------|-------|
| `single` | single | One model over all data |
| `hurdle` | s1 + pos_count | Standard two-part model |
| `pot` | s1 + pos_count (below) + tail (above) | Fixed-percentile threshold |
| `double_hurdle` (DH) | s1 + pos_binary (s2) + dh_bulk + tail | **Best structure** — covariate-dependent threshold |

### Why DH Wins

DH replaces POT's fixed percentile split with a covariate-dependent `P(high | positive, X)` (s2). This allows the model to route storms to the tail distribution based on their characteristics, not just their rank. Best results: **mae_rate 1.28** (vs POT 1.38, hurdle 1.44, single 1.49).

---

## Distributions

### Active in Production Analysis

| Name | Type | Notes |
|------|------|-------|
| `statsmodels_logistic` | Binary (s1, s2) | Standard logistic regression |
| `statsmodels_nb` | Count | Negative binomial |
| `statsmodels_gamma` | Count | Gamma GLM |
| `statsmodels_lognormal` | Count | WLS on log(rate) |
| `statsmodels_poisson` | Count | Poisson GLM |
| `statsmodels_tweedie` | Count | Tweedie GLM |
| `statsmodels_zip` | Count | Zero-inflated Poisson |
| `statsmodels_zinb` | Count | Zero-inflated NB |
| `scipy_gpd` | Tail | GPD via scipy — **dominates tail stage** |
| `sklearn_rf` | Binary/Count | Random Forest (500 trees) |
| `sklearn_xgb` | Binary/Count | XGBoost (100 rounds) |

### Redundant — Excluded from Analysis

These produce near-identical results to their statsmodels counterparts and inflate the grid:

| Name | Redundant With |
|------|---------------|
| `sklearn_logistic` | `statsmodels_logistic` |
| `sklearn_poisson` | `statsmodels_poisson` |
| `sklearn_tweedie` | `statsmodels_tweedie` |
| `sklearn_lognormal` | `statsmodels_lognormal` |

> **⚠️ `stage_grid.py` fix required:** These are currently excluded at analysis time in `analyze_stages.py`. They must also be excluded at grid generation time before any future run.

### Log-Covariate Note for ML Models

Tree models (`sklearn_rf`, `sklearn_xgb`) are invariant to monotone transforms — log-covariate sets produce identical predictions to untransformed sets. Log combos are excluded for these models at analysis time; must also be fixed in `stage_grid.py`.

---

## Covariate Sets

15 sets used in the current grid (expanded from the original 8):

| Name | Variables |
|------|-----------|
| `wind_sdi` | max_wind_speed + sdi |
| `wind_sdi_basin` | + basin dummies |
| `wind_sdi_island` | + is_island |
| `wind_sdi_basin_island` | + basin + is_island |
| `wind_sdi_year` | + data_year |
| `wind_sdi_basin_year` | + basin + data_year |
| `wind_sdi_island_year` | + is_island + data_year |
| `wind_sdi_basin_island_year` | all covariates |
| *(+ 7 log-transform variants)* | log(max_wind_speed), log(sdi), or both |

Log variants are excluded for tree models (see above).

---

## Grid Scale

- **2011 unique stages** across all (distribution × covariate × stage_type × threshold) combinations
- **25 OOS folds per stage** — 5 seeds × 5 folds
- **Thresholds evaluated:** 70, 75, 80, 85, 90, 95 percentiles (for DH/POT)

---

## Key Scripts

### Current (Stage-Based)

```
tc_models/
    stage_grid.py           # Generates 2011 stage specs
    run_one_stage.py        # Fits a single stage (called by orchestrator)
    orchestrate_stages.py   # Submits jobs to cluster via jobmon
    analyze_stages.py       # Assembles stages → full models, ranks
    analyze_dh.py           # DH-specific deep-dive analysis
    distributions/          # One module per distribution family
    structures/             # single.py, hurdle.py, pot.py, double_hurdle.py
```

### Deprecated (Spec-Based — Do Not Use)

```
tc_models/spec_grid.py                  # Old: generated 3678 redundant specs
tc_models/run_one_spec.py               # Old: ran a full model per spec
tc_models/orchestrate_tc_comparison.py  # Old: spec-based orchestrator
direct_relative_risk/tc_comparison.py   # Old: monolithic comparison script
```

---

## Primary Analysis Entry Point

```python
# After stages have been fit and results are in stage_results/:
python analyze_stages.py     # Rank all model structures
python analyze_dh.py         # Deep-dive DH components

# Or interactively:
jupyter notebook notebooks/dh_model_analysis.ipynb
```

---

## What Happened to GPD / ZIGPD?

GPD is no longer a standalone single/hurdle family. Investigation showed the shape parameter
consistently hits its upper bound (shape ≥ 1 → infinite mean). GPD is now only used as
a **tail component** (`scipy_gpd`) inside DH and POT models, where it's constrained
to the tail-only subpopulation. It dominates at this role.

---

## Performance Expectations

| Task | Time |
|------|------|
| Single stage, insample only | < 1 minute |
| Full grid (2011 stages × 25 folds) | ~4–8 hours on cluster |
| `analyze_stages.py` | ~10–20 minutes |
| `dh_model_analysis.ipynb` | ~5–15 minutes |
