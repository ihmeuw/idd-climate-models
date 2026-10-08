# Claude Session Memory

**Last Updated:** 2026-10-08

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
  prep counts use -180..180 shapes for NA; deleted
  03_admin_level_paf_launcher_modifed_memory.py (git rm). STAGES.md updated.
- stage1_run_config.py is untracked: must be committed with the launchers.

## Next Steps
1. User reviews; decides output path (stage1_v3 not yet run; stages 2-4 read stage1_v3).
2. Run one draw, then the launcher.
3. Commit.

## Resume Prompt
```
Continuing idd-climate-models CLIMADA v3 work: stage 1 can re-estimate central
pressure with CLIMADA's fit (--reestimate_central_pressure). Code + tests done
and uncommitted. See .claude/memory.md for findings and next steps.
```
