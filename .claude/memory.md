# Claude Session Memory

**Last Updated:** 2026-10-08 (stage 1 launched)

## Current Task
CLIMADA stage 1 (v3): option to re-estimate central pressure with CLIMADA's
own fit (non-ensemble) inside `climada/main/01_climada_intensity_main.py`
("Option A": no input files change). Implemented, tests pass, notebook verified.
Not yet run on the cluster; not committed.

## Why
Input files carry the ensemble central pressure (0.6 CLIMADA fit + 0.4 Dvorak).
Environmental pressure doesn't depend on ensemble/roci by default, so only
central pressure needs replacing.

## Key findings
- Input files store TC-risk vmax in **m/s** labelled "kn". The fit converts with
  MS_TO_KN = 1.943844 (same as add_tc_params).
- Default H08 windfield ignores track vmax (uses pressure), so
  `--convert_wind_to_knots` has no effect on intensity (verified: 0 m/s diff).
- Longitude: user chose 0..360 to match the original pipeline (add_tc_params fits
  on TC-risk's raw lon). Re-estimate now equals use_ensemble=False exactly.
  (IBTrACS check: 0-360 makes NA/EP fit ~17 mb too deep vs obs, -180..180 ~4 mb;
  for SP east of 180, 0-360 fits better. User accepted pipeline consistency.)
- EC-Earth3 ssp585 2020-2024 EP draw 1 (0..360 lon): re-estimate is ~3 mb lower
  than ensemble on average, ~10 mb higher at storm peaks.

- Test storm vs v2 (notebook section 6; storm_0028, EC-Earth3 ssp585 2020-2024
  EP draw 1): v3 with ensemble pressure == v2 exactly. CLIMADA fit (0..360):
  footprint IoU 0.985, intensity corr 0.998, same landfall location; peak
  78.9 -> 75.5 m/s, land pixels >=17 m/s 1824 -> 1850, mean exposure +2.0 h;
  min cp 886.7 -> 897.9 mb.

## What changed
- `climada/main/stage1_run_config.py` (new): flags, run config, log_dir_for.
- `01_climada_intensity_main.py`: `estimate_central_pressure_with_climada`,
  flags `--reestimate_central_pressure --convert_wind_to_knots --output_path
  --overwrite_okay`, per-draw run config under `<output>_log/draw_run_configs`,
  config attrs on zarr groups and markers, `__main__` guard.
- `01_climada_intensity_launcher.py`: same flags, passed through to each task;
  completion scan only counts markers with a matching run config.
- `climada/main/compare_central_pressure_to_pipeline.py`: one-draw check against
  `add_parameters_to_dataset(use_ensemble=False)`.
- `tests/test_climada_stage1_central_pressure.py` (unittest, 13 tests).
- `notebooks/2026_10_08_climada_v3.ipynb` (gitignored).

## Readiness review (2026-10-08)
- Stage 1 CLI end-to-end test into scratch: SP draw 1 OK (exit 0, markers +
  run config); resume skips; conflicting settings exit 1 with no writes.
- All 3,668 combos have 100 track files. No stage1_v3 output exists yet.
- Launcher marker scan made threaded (366,800 markers: ~160s vs ~10 min).
- Fixed 4B launcher ZeroDivisionError when every group is already done.
- Fixed after review: stage 2 + 4A launchers exit unless stage 1 markers exist
  for every draw they submit (require_stage1_complete in stage1_run_config.py);
  stage 3 writes `.no_intersection` markers for years with no admin hit;
  4B post-processing NUM_CORES=40 and resume signature includes 4B sentinels;
  deleted
  03_admin_level_paf_launcher_modifed_memory.py (git rm). STAGES.md updated.
- NA prep-count fix was reverted: correct NA counts would move NA into stage 3's
  >100 resource bin (shorter runtime) that was profiled on the old counts.
- All committed on feature/mfiking/climada_06_03_2026 (not pushed).

## Production v3 check (2026-10-09, notebook section 7)
- 4,421 stage 1 draws done (AU, SI, NA, EP) at time of check, all with the
  CLIMADA-fit run config. Section 7 compares completed v3 storms with v2.
- v2 NA output is mostly 0-360 lon (v3: -180..180) and ~68% of v2 storms have
  stretched grids; section 7 normalizes and flags these.
- CMCC-ESM2/historical/2001-2005/NA: v3 code + ensemble pressure reproduces v2
  exactly (storm_0032); with CLIMADA pressure footprints grow (median +42% land
  pixels on clean-grid storms, peak +3 m/s). v2-only landfalls all stretched-grid;
  of 10 v3-only, 3 land with ensemble too, 7 only with the deeper CLIMADA pressure.

## Mid-run check (2026-10-10, notebook section 8)
- 188,120 / 366,800 draws done, all markers CLIMADA-fit config. Sample of 224
  draws: bookkeeping/contents clean; v3 vs v2 consistent with pressure change.
- Known quirk: storms whose land footprint is 1 row/column get an empty landfall
  parquet (to_raster dlon NaN in compute_storm_landfall_locations); zarrs fine,
  nothing downstream reads landfall parquets. Fix = ensure_min_grid before
  to_raster there; not applied mid-run.
- Backfill prepared (2026-10-10): climada/main/backfill_landfall_locations.py
  (dry run default; --apply only once all 366,800 draws have markers; atomic
  rewrite) + tests/test_climada_landfall_backfill.py (9 pass). Plan: run
  --apply after workflow 635223 finishes, THEN add ensure_min_grid to
  compute_storm_landfall_locations in 01_climada_intensity_main.py.
- 10,560 SLURM TIMEOUTs (EP/WP mostly); retries resume per storm.

## FATAL tasks + fix (2026-10-12)
- Workflow 635223: 18,831 FATAL (EP 9,221, WP 6,422, ...), all exit code 1.
  Cause: 1-ROW land footprint -> to_raster IndexError in
  compute_storm_landfall_locations (1-COLUMN case = silent empty parquet).
  Sampled 261 fatal tasks: every unfinished draw has a 1-row orphan intensity
  group. Fixed: ensure_min_grid at top of compute_storm_landfall_locations.
  Verified on a scratch copy of EP draw 96. Then: resume workflow (cold, no
  SLURM jobs exist), backfill 1-column empties after the run.

## Last 37 EP draws (2026-10-12)
- Workflow 637013 (climada_env) finished all but 37 EP draws (CMCC/MRI, long
  storms 410-714 steps). Dense windfield n_time x 843,804 centroids x 2 float64
  -> one storm peaks ~22 GB (measured 586 steps, 474 s); 5 draws/35G task OOMs,
  pool hangs to TIMEOUT. Launcher got --draws_per_batch/--memory_gb/--runtime_min;
  relaunch: --draws_per_batch 1 --memory_gb 64 --runtime_min 1440.

## Next Steps
1. Stage 1 launched by the user 2026-10-08 with --reestimate_central_pressure
   into stage1_v3. Re-run the launcher until it reports 0 batches to submit.
2. Stage 2 launcher (now also runs the stage 3 count generators as downstream
   tasks), then stage 3 admin 0/1, then 4A -> 4B at admin 0/1.

## Resume Prompt
```
Continuing idd-climate-models CLIMADA v3 work: stage 1 can re-estimate central
pressure with CLIMADA's fit (--reestimate_central_pressure). Code + tests done
and uncommitted. See .claude/memory.md for findings and next steps.
```
