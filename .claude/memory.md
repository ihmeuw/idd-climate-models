# Claude Session Memory

**Last Updated:** 2026-01-26

## Current Task
Updated `00_orchestrator.py` to use the new simplified `02_process_variable.py` script. Ready for testing.

## Workflow Structure (4 Levels)
1. **Level 1**: Create folders (`01_create_folders.py`)
2. **Level 2**: Process raw → time-period files directly (`02_process_variable.py`)
3. **Level 3**: Run global TC-risk (`04_run_global_tc_risk.py`)
4. **Level 4**: Run basin TC-risk (`05_run_basin_tc_risk.py`)

## What `02_process_variable.py` Does
1. Find raw files that overlap with time period (parses dates from filenames)
2. Load with `xr.open_mfdataset()` (lazy)
3. Clip to time period years with `ds.sel(time=slice(...))`
4. Load into memory with `.compute()`
5. Fill NaNs (optimized: detects constant mask, computes EDT once)
6. Optionally regrid
7. Save directly to `TC_RISK_INPUT_PATH/.../time_period/`

## Resource Allocation
- Daily data (ua, va): 100G memory, 4h runtime
- Monthly data (tos, psl, hus, ta): 16G memory, 1h runtime

## Files in Pipeline
- `00_orchestrator.py` - orchestrates the workflow
- `01_create_folders.py` - creates directories
- `02_process_variable.py` - process raw → time-period file directly
- `04_run_global_tc_risk.py` - global TC-risk
- `05_run_basin_tc_risk.py` - basin TC-risk

## Next Steps
1. Run the orchestrator with TEST_RUN=1 to test
2. If successful, set TEST_RUN=0 for full run

## Resume Prompt
```
Continuing idd-climate-models work. Updated orchestrator to use new 4-level workflow:
folders → process_variable → global_tc_risk → basin_tc_risk. Ready to test.
See .claude/memory.md for details.
```
