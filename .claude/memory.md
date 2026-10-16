# Claude Session Memory

**Last Updated:** 2026-10-13 14:00 (stage 2 stopped: filesystem metadata storm)

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

## Resource respec (2026-10-12)
- Stage 1 done (366,800/366,800), validated; landfall backfill apply ~done.
- Stage 2: 3 cores/10 workers (a195abe); pilot 637734 OK; full run going.
- 4A: 120 min tasks, x1.5 retry (686b53a). User will not run 4A for now.
- Output roots renamed (02886fd): stage3_v3_admin{0,1}/, stage4b_v3_admin{0,1}/.
- 4B (d82a24e): ADMIN_LEVEL = 1; ADMIN1_MEMORY_GB {3:9,4:9,5:10,10:12,29:20}
  (admin 1 baseline ~7.3 GB); runtime request 0.75x packed estimate (v2 max
  0.69x). Post-processing left 40c/100G/240m. Profile in STAGES.md.

## Status check (2026-10-13 08:40)
- Backfill apply 637726: fix shards all done, 2,276/2,276 `rewritten`;
  summarize task queued (Priority) -> writes report.csv.
- Stage 2 637740: 18,333/90,797 draws done, ~2,000 running. CPU efficiency fell
  0.71 -> 0.09 (I/O-bound); 4,068 timeouts (37% of finished core-hours),
  retries x1.5. User chose NOT to cap concurrency. ~1,250 done/h now.

## Stage 3 respec (2026-10-13)
- v2 admin0 wfs 583912/584066/584230/584480/584688 (584230+ match the 50G
  bins), admin1 589639/589651/589680/589842. ~2,900 sampled tasks.
- Memory follows basin, not estimated admin count. Launcher now uses
  _BASIN_RESOURCES per level (peak x1.15, longest x1.2); bins removed.
  Admin0 EP needs ~88G (100% OOM at 50G in v2).

## Stage 2 stopped (2026-10-13 ~13:54) — filesystem metadata storm
- Storage admins flagged 4.2M NOENT lookups from uid 701454 (zarr.json 1.54M,
  .zgroup/.zattrs/.zmetadata ~386K each). User stopped 637740 via GUI; no
  stage 2 jobs left in SLURM at 13:54. Do not restart until reads are fixed.
- stage1_v2 and stage1_v3 are BOTH Zarr format 3 (not a pipeline v2/v3 issue).
- Causes: (1) each stage 2 task reads all 100 draws of its combo, ~25 tasks
  per combo -> each storm read ~25x, opened twice per task (metadata pass +
  data pass); (2) no zarr_format=3 -> v2 metadata probes per open (~28%);
  (3) 64x64 chunks + NaN fill -> ~50% of chunk files absent, probed on read.
- Fix options: zarr_format=3 in stage 2/4A/4B opens; read storm dates from
  group zarr.json; restructure stage 2 to read each combo once; stage 1 write
  as single chunk or sharded. zarr.json misses (1.54M) not reproduced — ask
  admins for sample paths.
- v2 vs v3 stage 2 comparison: v3 PAF ~0.74x v2 (grid-fix lowers, CLIMADA
  pressure raises except EP where it lowers). AU decomposition: grid 0.66x,
  pressure 1.13x, net 0.74x.

## Filesystem fixes, step by step (2026-10-13)
1. DONE 33c5438 (+7d1e1ab CRLF restore): zarr_format=3 in stage 2/4A/4B opens.
   Test (stage 2 read pass, SP draw e0, 10 storms): 358 -> 298 lookups,
   118 -> 58 misses, format probes 60 -> 0, data identical. Test harness:
   scratchpad zread_test.py (strace; count misses by checking paths on disk —
   strace -f splits unfinished/resumed lines).
2. DONE cc11a3b: storm dates from group zarr.json (stage 2/4A/4B
   iter_storms_metadata). Full stage 2 task test (SP MRI historical 1970-1974,
   storm_draw_0100, resp; scratchpad run_stage2_task.py writes to scratch):
   35,471 -> 21,296 lookups, 12,558 -> 6,888 misses; rasters identical to
   production; 125 s on 2 cores. zarr.json misses = 0 in stage 2 reads.
3. DONE 95b41f1: main/storm_zarr.py open_storm (ChunkManifestStore lists
   <array>/c once; absent chunks never looked up) used in stage 2/4A/4B data
   reads; tests/test_climada_storm_zarr.py (5). Same task: lookups 18,447,
   misses 0, dir listings 1,145 -> 5,527; rasters identical; 107 s.
4. DONE 54c6fad: stage 2 one task per combo (3,612 tasks), resample intensity
   once/storm (bit-identical), draws in one process (workers pickled ~360 MiB
   per draw). Launcher defaults 1 core/4G/60 min, max 300 running, --dry_run.
   SP combo 103 s 0.3 GB; EP 58-job combo 16 min 2.6 GB; 60/60 rasters
   identical to production; resume test OK. Dry run: 3,612 combos.
   bfe0f34: gdal_env sets GDAL_DATA/PROJ_DATA (rasterio searched conda env on
   /mnt/share/code every resample: 14k lookups/9.3k misses per old task).
5. DONE c319616 (4B): lru_cache shapefile/pop, prepare_storm_rasters caches
   last storm, rows sorted by storm; 11/11 records identical. cc62853: GDAL env
   in prep count generators. Stage 3 caching (lru_cache + preload before fork)
   committed 2026-10-13 (opens 20 -> 4 per task, 5/5 parquets identical).
6. Pilot 638451 (42 combos) clean: max 20.8 min, 2.9 GB, 4,606/4,606 outputs.
   FULL stage 2 = workflow 638496 (3,570 combos + 2 count tasks), launched
   2026-10-13 16:24 detached (nohup setsid), log
   /mnt/share/homes/mfiking/climada_logs/stage2_combo_full_20261013_162430.log

## Stage 2 full run 638496 result (2026-10-13 ~19:15): FAILED, 3,570 done, 2 fatal
- Both fatal combos = damaged stage 1 INTENSITY groups (partial write, no lat,
  2026-10-11 19:23-19:27; exposure + landfall intact):
  EC-Earth3 r1 historical 1980-1984 WP draw 28 (_e27) storm_0136 (only 1984 missing)
  EC-Earth3 r1 historical 1985-1988 WP draw 82 (_e81) storm_0129 (empty attrs ->
  dates pass KeyError -> all 4 years missing).
- Regenerated with stage 1 code + v3 run config, verified (exposure + landfall
  identical to prod), staged in /mnt/share/homes/mfiking/climada_repair/
  20261013_storm_{0136,0129}. USER must swap them into stage1_v3 (org rule: no
  overwrites on shared drives by Claude). User swapped both in (verified identical).
- Repair run: workflow 638551 (--combo x2 + 2 count tasks), launched 20:47,
  log climada_logs/stage2_combo_repair_20261013_204730.log. 6c8c404: launcher
  --combo + threaded completion scan (sequential full scan took hours).

## Stage 2 complete (2026-10-13 21:46): repair wf 638551 done incl. both count parquets.

## Stage 3 restructure (c64c1dd, 2026-10-13)
- One task per (combo, year): process_combo_year reads/masks each 100 m population
  window once for all storm draw x cause jobs; per-job resample remains (~95 s/job-yr
  SP admin0; shared ~200 s). 16,968 tasks per admin level. Bit-identical: combo vs
  per-job (SP a0 6/6, NA a1 8/8) and on v2 inputs vs saved stage3_v2(_admin1) (6/6, 8/8).
  Pixel-lookup variant identical but slower -> dropped.
- Launcher: --admin_level CLI, --combo, --pilot, --dry_run, threaded scan, runtime =
  1.5 x (v2 per-job min/5) x (0.67+0.33*jobs), memory by basin, max 300 running.
- LAUNCHED both levels 2026-10-13 22:30 detached: admin 0 = wf 638563, admin 1 = wf 638562; logs
  climada_logs/stage3_admin{0,1}_combo_year_20261013_223051.log. User raised
  concurrency to 450 for both in the GUI. Launchers run inside interactive SLURM
  job 56876825 (3d17h left at 22:35) — relaunch the launcher if it dies.
- 23:10: all ~300 NA admin1 first attempts OOM at 4G (after 12-16 min); Jobmon
  retries at 9G (scale 1.25 = +125%). AU admin1 3G borderline. 5a15b71 raises
  NA a1 -> 10G, AU a1 -> 4G for relaunches. CHECK in the morning that NA a1
  retries succeeded. Expect admin 1
  overnight; admin 0 ~30+ h at 300 concurrent (raise with jobmon workflow concurrency).

## 2026-10-14 morning
- Stage 3 at ~06:45: admin1 16,573/16,968 done, admin0 1,436 done, 0 fatal.
- notebooks/2026_10_14_climada_v3_stage1_3.ipynb (gitignored, executed): v2 vs v3
  stages 1-3 for CMCC-ESM2 r1 ssp245 2097-2100 AU 2097, storm_draw_0002, stage 1
  storm_0001 draw 1. Stage 1 peak -6%, footprint -2%; stage 2 raster PAF sum 0.67x;
  stage 3 admin0 PAF sum 1.03x (Australia -33%, NZ -27%, islands +10%), admin1
  (Indonesia only) 1.04x. Built/executed via esfg_env nbclient + temp kernelspec
  for climada_env (scratch).

## Stage 4 (2026-10-14)
- db83e16: 4A/4B rasterize each admin piece once (piece_mask/apply_mask), 4A caches
  shapefile/pop per process; outputs identical (4A 67 parquets excl.
  processing_time_seconds; 4B 11/11). 4A admin0 draw 215 -> 179 s.
- f2d167d: 4A launcher --max_concurrent (default 300).
- 4A LAUNCHED 07:58 detached: workflow 638708, 30,100 group tasks per level (admin 0
  then admin 1), log climada_logs/stage4a_admin0_admin1_20261014_075854.log.
  4B launches after 4A's resource assignment per level (ADMIN_LEVEL = 1 set).
- Stage 3 at 08:00: admin1 16,926/16,968; admin0 1,650 done, ~450 running, median
  100 min (NA 260 min, pinned at 53G; 9 timeouts retrying at 990 min); ~2.5 days
  left. Interactive job 56876825 ends ~Oct 17 14:30 — relaunch launchers if needed.

## 2026-10-16: node failure + resume
- Old interactive job 56876825 hit NODE_FAIL 2026-10-14 22:25 -> both launchers died
  (also Jobmon server MySQL 500 errors then). Workflows 638563 (stage3 a0, 3,947 done)
  and 638708 (4A, 8,772 done, 1 fatal) FAILED; 638562 (stage3 a1) DONE.
- 4A fatal group 4019 = damaged stage1 EXPOSURE group EC-Earth3 r1 historical 1980-1984
  WP draw 33 (_e32) storm_0125 (written 2026-10-11 19:25, same incident). Regenerated,
  verified (intensity + landfall identical), user swapped in. Scan of both affected
  combos (54,450 storm groups): no other damage.
- Resumed via sbatch long.q clients (climada_logs/jobmon_resume.sbatch, hot resume,
  --execution-timeout 11 days): stage3 a0 = job 58307814 (wf run 416607),
  4A = job 58308078. New interactive job 58307758 (p0005) ends Oct 21.

## Next Steps
1. Stage 2 (637740) finishing -> count tasks -> launch stage 3 admin 0 and
   admin 1 (ADMIN_LEVEL edited by hand in 03 launcher).
2. Backfill summarize task writes report.csv (fix shards: 2,276 rewritten).
3. 4A (admin 1) when the user decides; then 4B (ADMIN_LEVEL = 1).

## Resume Prompt
```
Continuing idd-climate-models CLIMADA v3 rerun (non-ensemble central pressure).
Stage 1 done; stage 2 running; 4A/4B resources tightened and committed. Next:
respec stage 3 resources from v2 workflows. See .claude/memory.md.
```
