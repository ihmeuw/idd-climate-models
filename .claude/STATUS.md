# Project status
Updated: 2026-09-23

## Goals
Build a pipeline to estimate tropical cyclone (TC) mortality risk under climate change scenarios.
Two tracks:
1. **TC risk pipeline** — raw climate data → TC risk models → CLIMADA impact assessment (paused)
2. **Direct relative risk modeling** — parametric/ML models predicting TC mortality from wind speed + SDI

## Orientation
Two tracks, both parked at decision points.

1. **TC risk pipeline.** CMIP6/ERA5 → TempestExtremes storm counts → BayesPoisson time bins
   (`notebooks/detect_time_bins.ipynb`; bins 10–30 yr, chunked to ≤5-year periods by
   `MAX_PERIOD_DURATION = 5` in the orchestrator) → TC Risk downscaling → `add_tc_params`
   (central pressure, RMW, ROCI) → CLIMADA input → CLIMADA Holland-2008 hazard.
   The 2026-09-23 audit found that central pressure is the hazard's sole intensity driver
   (H08 derives peak wind from the pressure deficit, not from track vmax) and that it comes
   from a hardcoded, unfitted 0.6·CLIMADA + 0.4·"Dvorak" blend in
   `tc_estimation.ensemble_estimates`, always on via `use_ensemble=True` in
   `tc_risk_functions.py`. The second curve matches neither Dvorak basin table nor
   Knaff & Zehr 2007, and CLIMADA contains no Knaff & Zehr code. Ensemble RMW/ROCI never
   reach CLIMADA (radius fields are zero-filled; CLIMADA re-derives RMW from the blended
   pressure). Relative to CLIMADA-only, the blend gives hurricanes +5–8% peak wind and
   sub-hurricane storms −3 to −21%. Whether to change the estimator is Bobby's decision and
   is undecided.
2. **Direct relative risk (DH models).** 587,776 assembled models evaluated; TOPSIS ranking
   and final selection pending.

## Recent steps
- 2026-09-23: Audited central-pressure estimation — blend provenance (first seen 2025-10-28 in
  `~/repos/climada_python/notebooks`, committed 2025-11-19), KZ07 Eq. 7 verified from the paper,
  Dvorak/Atkinson-Holliday tables, CLIMADA regression. No code changed.
- 2026-09-23: Traced the blend into the CLIMADA H08 hazard; quantified the peak-wind shift; found
  ensemble RMW/ROCI unused downstream.
- 2026-09-23: `detect_time_bins.ipynb` — explained 1965 padding vs the scaling multiplier
  (changing `YEAR_RANGES` shifts multipliers up to ±12%; MPI-ESM1-2-HR has real 1965–69 counts);
  supplied `plot_clipping_basin_comparison` with a `panel_by` toggle; diagnosed the NA-basin NaN
  (`read_csv` without `keep_default_na=False`) and missing bin lines (`bins_df` not in kernel).
- 2026-09-23: Confirmed TC Risk runs BayesPoisson bins chunked to ≤5 years; summarized the method
  (Poisson MLE with synchronized breaks, BIC model selection).
- 2026-09-23: Committed the April `plot_all_tracks` extensions (c818d54) and the pyarrow /
  idd-spatial-analysis dependency change (e5aeb03).
- 2026-04-20: Added `plot_all_tracks()` to `storm_data/visualizers.py`; `geo_utils.reproject_gdf_to_360()`.

## Key results
- **587,776 models** assembled from 16^4 covariate combinations × 12 dist combinations
- **Threshold**: 70% only (previous TOPSIS showed it optimal)
- **Bulk dists**: NB, gamma, lognormal (no poisson/tweedie)
- **Tail dists**: scipy_gpd, NB, gamma, lognormal
- **15/144 stages failed**: All statsmodels_gamma — numerical instability

## Next steps
1. Decide the central-pressure estimator (keep blend / CLIMADA-only / KZ07 with S=1 / Dvorak by
   basin) and correct the slide description to match the code
2. Re-run `detect_time_bins.ipynb` cells in order; reload `combined_df` with
   `keep_default_na=False`; rebuild `bins_df` (or read `rfc.TIME_BINS_DF_PATH`) before the bin plots
3. Delete `notebooks/visualizations/rotate_shapefiles.py` (logic moved to geo_utils.py; still present)
4. **Run TOPSIS** on `dh_exhaustive_expanded.csv` to rank models
5. Select final model from top candidates
6. Implement uncertainty quantification / draw generation

## File locations
- Stage results: `/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/stage_results_dh/`
- Model pkl files: `stage_results_dh/models/`
- Stage grid: `stage_results_dh/stage_grid_expanded.json`
- **Exhaustive results: `stage_results_dh/results/dh_exhaustive_expanded.csv`** (587,776 rows)
- TOPSIS notebook: `notebooks/dh_model_selection_topsis.ipynb`
- Code: `src/idd_climate_models/tc_models/{stage_grid_dh_expanded.py, analyze_dh_exhaustive.py}`

## Parking lot
- TC risk pipeline (levels 1-4 orchestrator) — paused
- `estimate_roci_knaff()` in `tc_estimation.py` reuses KZ07 Eq. 6 (Rmax in km, with lat−25) as
  ROCI in nm and drops the −25
- Central-pressure audit detail: `.claude/memory.md` and DEAD_ENDS (2026-09-23); KZ07 PDF saved
  under `~/.claude/projects/<repo-slug>/<session>/tool-results/knaffzehr2007.pdf`
- README.md still has placeholder text
- Old spec-based files can be deleted once final model is selected
