# Project status
Updated: 2026-04-20

## Goals
Build a pipeline to estimate tropical cyclone (TC) mortality risk under climate change scenarios.
Two tracks:
1. **TC risk pipeline** — raw climate data → TC risk models → CLIMADA impact assessment (paused)
2. **Direct relative risk modeling** — parametric/ML models predicting TC mortality from wind speed + SDI

## Recent steps
- 2026-04-20: Added `plot_all_tracks()` to `storm_data/visualizers.py` — multi-track map with date/basin/country filters
- 2026-04-20: Created `storm_data/geo_utils.py` with `reproject_gdf_to_360()` for shapefile longitude wrapping
- 2026-04-20: Fixed netCDF file handle leak in `visualize_tc_risk_output.ipynb` cell 10
- 2026-04-20: Added 3 example cells to notebook (all storms, storms hitting land, storms hitting US)
- 2026-03-31: Culled covariate sets from 23 → 16; ran 587,776 DH model combinations

## Key results
- **587,776 models** assembled from 16^4 covariate combinations × 12 dist combinations
- **Threshold**: 70% only (previous TOPSIS showed it optimal)
- **Bulk dists**: NB, gamma, lognormal (no poisson/tweedie)
- **Tail dists**: scipy_gpd, NB, gamma, lognormal
- **15/144 stages failed**: All statsmodels_gamma — numerical instability

## Next steps
1. Test `plot_all_tracks()` with the 0-360 shapefile reprojection — restart kernel and run notebook
2. Delete `notebooks/visualizations/rotate_shapefiles.py` (logic moved to geo_utils.py)
3. **Run TOPSIS** on `dh_exhaustive_expanded.csv` to rank models
4. Select final model from top candidates
5. Implement uncertainty quantification / draw generation

## File locations
- Stage results: `/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/stage_results_dh/`
- Model pkl files: `stage_results_dh/models/`
- Stage grid: `stage_results_dh/stage_grid_expanded.json`
- **Exhaustive results: `stage_results_dh/results/dh_exhaustive_expanded.csv`** (587,776 rows)
- TOPSIS notebook: `notebooks/dh_model_selection_topsis.ipynb`
- Code: `src/idd_climate_models/tc_models/{stage_grid_dh_expanded.py, analyze_dh_exhaustive.py}`

## Parking lot
- TC risk pipeline (levels 1-4 orchestrator) — paused
- `tests/test_plot_all_tracks.py` exists but pytest not in env
- README.md still has placeholder text
- Old spec-based files can be deleted once final model is selected
