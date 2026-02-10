# TC-Risk Pipeline Orchestration (Refactored)

This folder contains the refactored orchestration scripts for the TC-risk pipeline.

## Files

- `00_orchestrator.py` - Main orchestrator for running the full pipeline
- `00_orchestrator_targeted.py` - Targeted orchestrator for specific reruns (reads from CSV)
- `orchestrator_utils.py` - Shared utility functions for output checking, cleaning, etc.

## Key Changes from Original

1. **Simplified configuration**: Removed complex flag permutations
2. **Always-on dynamic draw counting**: Automatically resumes from existing draws
3. **Explicit cleaning steps**: `CLEAN_STARTING_LEVEL` and `CLEAN_ENDING_LEVEL` control what gets cleaned
4. **No ADD_DEPENDENCIES flag**: Dependencies are always added when running multiple levels
5. **Async deletion for Level 1**: Renames paths first, then deletes in background

## Worker Scripts

The actual worker scripts (01-05) remain in the original `01_process_raw_through_climada_input` folder:
- `01_create_folders.py`
- `02_process_variable.py`
- `04_run_global_tc_risk.py`
- `05_run_basin_tc_risk.py`

These are referenced by path in the orchestrator templates.

## Usage

### Main Orchestrator

For full pipeline runs or large-scale processing:

```bash
python 00_orchestrator.py
```

Configure via constants at top of file:
- `STARTING_LEVEL` / `ENDING_LEVEL`: Which levels to run (1-4)
- `CLEAN_STARTING_LEVEL` / `CLEAN_ENDING_LEVEL`: Which levels to clean before running
- `TEST_RUN`: Limit number of tasks for testing

### Targeted Orchestrator

For specific reruns (models, scenarios, time periods, basins):

```bash
python 00_orchestrator_targeted.py --input targeted_runs.csv
```

CSV format:
```csv
model,variant,scenario,time_period,basin,starting_level,ending_level
MPI-ESM1-2-HR,r1i1p1f1,ssp126,2015-2040,NA,3,4
MRI-ESM2-0,r1i1p1f1,ssp245,,,3,4
```

Empty fields = run all options for that dimension.
