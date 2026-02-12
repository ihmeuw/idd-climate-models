# Task-Based Level 4 Workflow

The orchestrator now uses **task-based Level 4 execution** exclusively. Level 4 tasks read from `level_4_task_assignments.csv` which maps task IDs to basin/draw combinations.

## Two Workflow Options

### Option A: Smart Resumable Mode (Run Level 0 First)

**Best for**: Production runs, handling corrupted files, resuming after timeouts

1. **Run orchestrator with Level 0** to create status files and smart task assignments:
   ```bash
   python 00_orchestrator.py \
       --starting_level 0 \
       --ending_level 0 \
       --data_source cmip6 \
       --queue all.q \
       --project proj_rapidresponse
   ```
   
   This creates:
   - `draw_status.csv` for each basin (validates NetCDF + Zarr, auto-deletes corrupted files)
   - `level_4_task_assignments.csv` with only incomplete draws distributed evenly

2. **Run orchestrator with Level 3-4** to execute basin TC-risk:
   ```bash
   python 00_orchestrator.py \
       --starting_level 3 \
       --ending_level 4 \
       --data_source cmip6 \
       --queue all.q \
       --project proj_rapidresponse
   ```

**Benefits:**
- Only processes incomplete draws (no wasted computation)
- Automatic corruption detection and cleanup
- Resumable after timeouts or failures
- Even load distribution across tasks

---

### Option B: Simple "Run All" Mode

**Best for**: First runs, testing, simple batch processing

1. **Manually create task assignments** for all draws:
   ```bash
   python 00_create_level_4_task_assignments.py \
       --data_source cmip6 \
       --draws_per_batch 25 \
       --full_run
   ```
   
   This creates uniform assignments:
   - Task 1: All basins, draws 0-24
   - Task 2: All basins, draws 25-49
   - Task 3: All basins, draws 50-74
   - ...

2. **Run orchestrator with Level 3-4** (same as Option A Step 2)

**Benefits:**
- Simple, predictable task structure
- No need to run Level 0
- Good for initial runs when no files exist

---

## Task Assignment File Structure

`level_4_task_assignments.csv` contains three columns:
```csv
task_id,basin,draw
1,EP,0
1,EP,1
...
1,EP,24
1,NA,0
1,NA,1
...
```

- Each task processes ~25 draws (configurable via `--draws_per_batch`)
- Draws may be distributed across multiple basins per task
- Task IDs are sequential starting from 1

---

## Updated 05_run_basin_tc_risk.py Interface

The basin script now accepts `--task_id` instead of `--basin --draw_start --draw_end`:

```bash
# Old (no longer supported)
python 05_run_basin_tc_risk.py \
    --data_source cmip6 \
    --model EC-Earth3 \
    --variant r1i1p1f1 \
    --scenario historical \
    --time_period 1970-1974 \
    --basin EP \
    --draw_start 0 \
    --draw_end 24

# New (task-based)
python 05_run_basin_tc_risk.py \
    --data_source cmip6 \
    --task_id 1
```

The script reads `level_4_task_assignments.csv` to determine which basin/draw combinations to process.

---

## Configuration

Key parameters in the orchestrator:

- `NUM_DRAWS = 250` (total draws to process)
- `DRAWS_PER_BATCH = 25` (target draws per task, configurable)
- `STARTING_LEVEL = 0-4` (now supports Level 0 for status files)
- `ENDING_LEVEL = 0-4`

Task assignments are stored at:
```
{CLIMADA_INPUT_PATH}/{data_source}/level_4_task_assignments.csv
```

Status files are stored at:
```
{CLIMADA_INPUT_PATH}/{data_source}/{model}/{variant}/{scenario}/{time_period}/{basin}/draw_status.csv
```
