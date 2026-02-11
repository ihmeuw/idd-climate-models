# TC-Risk Pipeline - Current State Summary
**Last Updated:** February 11, 2026

## 🎯 Project Overview

This is a climate modeling pipeline that processes CMIP6 data through TC-risk downscaling to produce tropical cyclone track datasets. The pipeline uses Jobmon for distributed task execution across an HPC cluster.

### Pipeline Structure (5 Levels)
```
Level 0: Create draw status files + task assignments
         ↓
Level 1: Create folders (TC-risk input/output directories)
         ↓
Level 2: Process raw files to time-period files
         ↓
Level 3: Run global TC-risk
         ↓
Level 4: Run basin TC-risk (21,338 parallel tasks)
```

---

## 📊 Current State

### Task Distribution
- **Total tasks:** 21,338
- **Total incomplete draws:** 508,311
- **Task sizes:**
  - 19,300 tasks with 25 draws (full batches)
  - 2,038 tasks with 1-24 draws (partial batches)
  - Distribution: 56 tasks with 1 draw, 50 with 2 draws, ... up to 19,300 with 25 draws

### Resources Per Task
- **Memory:** 8G
- **Cores:** 5 (4 workers + 1 scheduler)
- **Runtime:** Capped at 6 hours (formula often predicts higher, but actual ~1.2s/storm)

### Key Paths
- Task assignments: `/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/task_assignments.csv`
- CLIMADA input: `/mnt/team/rapidresponse/pub/tropical-storms/climada/input/`
- TC-risk output: `/mnt/team/rapidresponse/pub/tropical-storms/tc_risk/output/`
- Draw status files: `{CLIMADA_INPUT_PATH}/{data_source}/{model}/{variant}/{scenario}/{time_period}/{basin}/draw_status.csv`
- Completion markers: `{CLIMADA_INPUT_PATH}/{data_source}/{model}/{variant}/{scenario}/{time_period}/{basin}/.draw_####.complete`

---

## 🔧 Recent Major Fixes (This Session)

### 1. **NA Basin NaN Bug** ✅
**Problem:** Basin code "NA" was being interpreted as NaN by pandas, causing it to disappear from task assignments.

**Fix:** Added `keep_default_na=False` to all `pd.read_csv()` calls that handle basin data:
- `00_orchestrator.py` (line 782)
- `orchestrator_utils.py` (lines 717, 759)
- `00_create_task_assignments.py` (now uses marker-based detection)
- `05_run_basin_tc_risk.py` (task reading logic)

### 2. **Task Assignment Structure Overhaul** ✅
**Problem:** Original design aggregated by basin only, creating 70 tasks total for 508K draws. Tasks didn't specify which model/variant/scenario/time_period to process.

**Solution:** Complete rewrite of task assignment system:

**New CSV Format:**
```csv
task_id,model,variant,scenario,time_period,basin,draw
1,CMCC-ESM2,r1i1p1f1,historical,1965-1969,AU,28
1,CMCC-ESM2,r1i1p1f1,historical,1965-1969,AU,36
1,CMCC-ESM2,r1i1p1f1,historical,1965-1969,AU,38
...
```
- **One row per draw** (not ranges) - handles gaps correctly
- Each task has explicit model/variant/scenario/time_period/basin context
- Tasks group up to 25 draws per combination
- File has ~500K rows for 21,338 tasks

**Changes to `00_create_task_assignments.py`:**
- Iterates through each `draw_status.csv` file individually
- Extracts full path components (model/variant/scenario/time_period/basin)
- Reads completion markers (not CSV) to find incomplete draws
- Creates one row per incomplete draw, batched into groups of 25
- Output: enumerated draw list with full combination context

**Changes to `05_run_basin_tc_risk.py`:**
- **Task-based mode:** Accepts `--task_id` parameter
- Reads `task_assignments.csv` and filters by task_id
- Groups rows by combination (should be only 1 per task)
- Extracts explicit draw list: `[28, 36, 38, 63, 65, ...]`
- Added `draws_list` parameter to `process_single_combination()` and `validate_batch_output()`
- Only processes/validates the exact draws specified (no gaps filled)

### 3. **Completion Tracking via Markers** ✅
**How it works:**
1. TC-risk validates each draw after processing
2. Creates atomic `.draw_####.complete` marker file (0 bytes, 775 permissions)
3. `00_create_task_assignments.py` uses `get_completed_draws_from_markers()` to identify complete draws
4. Task assignments only include draws WITHOUT markers

**Race Condition Prevention:**
- Marker files are atomic (created/absent only)
- No concurrent CSV writes during task execution
- `draw_status.csv` is **only updated by Level 0** (00_create_draw_status_file.py)
- This happens **before** task assignments are created, so no race conditions

**Auto-Resume:**
- If a task crashes mid-run, completed draws have markers
- On restart, `process_single_combination()` checks markers first
- Only processes draws without markers
- No redundant work

### 4. **Validation Optimization** ✅
**Removed redundant validation:**
- TC-risk execution already validates each draw and creates markers
- Previous code had a second `validate_batch_output()` step after TC-risk
- This re-validated the same draws (took extra time)
- **Fix:** Removed secondary validation from `process_single_combination()`

**Current flow:**
1. Check missing draws (via markers)
2. Run TC-risk (validates internally, creates markers)
3. Done ✅

### 5. **Resource Limits** ✅
**Problem:** Empirical formula predicted up to 34 hours for largest tasks (235 storms × 25 draws).

**Reality:** Actual timing is ~1.2 seconds per storm:
- 235 storms × 25 draws × 1.2s ≈ **2 hours** (not 34!)

**Fix:** Added 6-hour cap to runtime calculation in `resource_functions.py`:
```python
runtime_hours = min(int(np.ceil(runtime_minutes / 60)), 6)
```

---

## 📁 Key Files and Their Purposes

### Level 0 Scripts (Setup & Task Assignment)

#### `00_create_draw_status_file.py`
**Purpose:** Validate existing NetCDF/Zarr files for a single basin, create status CSV and markers.
- Called once per basin combination during Level 0
- Reads existing completion markers first (fast path)
- Validates NetCDF → Zarr pairs using shared `validate_single_draw()`
- Creates `.draw_####.complete` markers after validation
- Cleans up orphaned Zarr files
- Outputs: `draw_status.csv` + completion markers

#### `00_create_task_assignments.py`
**Purpose:** Create task distribution by reading completion markers across all combinations.
- Called once after all draw status files created
- Scans all `draw_status.csv` files in tree
- For each combination:
  - Reads completion markers to identify incomplete draws
  - Batches incomplete draws into groups of 25
  - Creates one CSV row per draw with full context
- Output: `task_assignments.csv` (one row per incomplete draw)
- Uses: `get_completed_draws_from_markers()` from `zarr_functions.py`

### Level 4 Executor

#### `05_run_basin_tc_risk.py`
**Purpose:** Execute TC-risk downscaling for a batch of draws.
**Two modes:**

**1. Task-based mode** (used by orchestrator):
```bash
python 05_run_basin_tc_risk.py --task_id 123 --total_memory 8G --data_source cmip6
```
- Reads `task_assignments.csv`
- Filters to rows matching task_id
- Groups by combination (validates only 1 combination per task)
- Extracts explicit draw list
- Calls `process_single_combination()` with `draws_list` parameter

**2. Direct mode** (manual testing):
```bash
python 05_run_basin_tc_risk.py \
  --model CMCC-ESM2 --variant r1i1p1f1 --scenario ssp126 \
  --time_period 2020-2024 --basin NI \
  --draw_start 0 --draw_end 24 --total_memory 20G
```

**Key function: `process_single_combination()`**
- Accepts optional `draws_list` parameter for explicit draw enumeration
- Step 1: Check for completion markers, identify missing draws
- Step 2: Run TC-risk only for missing draws (validation happens internally)
- If all draws complete, returns immediately

### Orchestrator

#### `00_orchestrator.py`
**Purpose:** Master workflow controller using Jobmon.

**Configuration:**
- `STARTING_LEVEL` / `ENDING_LEVEL`: Control which levels run (default: 0-4)
- `DRAWS_PER_BATCH = 25`: Draws per task
- `ADD_DEPENDENCIES`: Sequential vs independent execution
- `DATA_SOURCE = "cmip6"`

**Level 0 workflow:**
1. Create draw status file tasks (one per basin)
2. Create task assignment task (depends on all status files)

**Level 4 workflow:**
1. Reads `task_assignments.csv` to get total task count
2. Creates one Jobmon task per task_id
3. Each task calls: `05_run_basin_tc_risk.py --task_id <id> --total_memory 8G`

### Shared Utilities

#### `zarr_functions.py`
- `validate_single_draw()`: 3-layer validation (existence → openable → integrity)
- `get_completed_draws_from_markers()`: Scans for `.draw_####.complete` files
- `create_draw_completion_marker()`: Creates atomic marker files

#### `resource_functions.py`
- `get_level4_resources()`: Calculate memory/cores/runtime for tasks
- `predict_memory_requirement_gib()`: Based on time period duration
- `predict_runtime_requirement_minutes()`: Based on storm count, capped at 6 hours

#### `orchestrator_utils.py`
- `get_total_tasks_from_assignments()`: Count unique task_ids in CSV
- `read_task_assignment()`: Read task data for specific task_id

---

## 🧪 Testing Results

### Test Case: Task 123
Initially: CMCC-ESM2/r1i1p1f1/ssp126/2020-2024/NI with 1 draw (249)

**First run:**
```bash
python 05_run_basin_tc_risk.py --task_id 123 --total_memory 20G
```
- ✅ Read task assignments correctly
- ✅ Identified single draw [249]
- ✅ Processed draw in 30.9s (15.5s TC-risk + 15.4s post)
- ✅ Created completion marker
- ✅ Validation passed

**Re-ran task assignment creation:**
```bash
python 00_create_task_assignments.py --data_source cmip6 --draws_per_batch 25
```
- ✅ Draw 249 excluded from new assignments
- ✅ Task count decreased: 21,339 → 21,338
- ✅ Total draws decreased: 508,312 → 508,311
- ✅ Task 123 now refers to different combination (task IDs shifted)

---

## ⚠️ Important Technical Details & Gotcas

### Pandas and 'NA' Basin
**Critical bug:** Pandas interprets "NA" as NaN by default.

**Solution:** Always use `keep_default_na=False` when reading CSVs with basin codes:
```python
df = pd.read_csv(file_path, keep_default_na=False)
```

**Files requiring this:**
- `00_orchestrator.py`
- `orchestrator_utils.py`
- `00_create_task_assignments.py`
- `05_run_basin_tc_risk.py`
- Any script that reads basin codes from CSV

### Completion Tracking: Why Two Systems?

**System 1: Atomic Marker Files** (`.draw_####.complete`)
- Created immediately after draw validation
- Atomic (no race conditions)
- Fast to check (file existence)
- Used during task execution

**System 2: CSV Status Files** (`draw_status.csv`)
- Created/updated only by Level 0
- Aggregated view for reporting
- Used by task assignment creation
- Not updated during task runs (prevents race conditions)

**Workflow:**
1. Tasks execute → create markers (fast, atomic, parallel-safe)
2. Re-run Level 0 → syncs CSV from markers (single-threaded, organized)
3. Create task assignments → reads current CSV state

**Why not update CSV during execution?**
- Multiple tasks processing same basin simultaneously would cause file lock conflicts
- CSV writes are not atomic
- Marker files solve both problems

### Task Assignment CSV Structure

**Old (broken) format:**
```csv
task_id,basin,draw_start,draw_end,num_draws
1,NA,0,24,25
```
Problems:
- Draw ranges don't handle gaps (what if draws 5-10 are complete?)
- No model/variant/scenario/time_period context

**Current (working) format:**
```csv
task_id,model,variant,scenario,time_period,basin,draw
1,CMCC-ESM2,r1i1p1f1,historical,1965-1969,AU,28
1,CMCC-ESM2,r1i1p1f1,historical,1965-1969,AU,36
1,CMCC-ESM2,r1i1p1f1,historical,1965-1969,AU,38
```
Advantages:
- One row per draw
- Handles gaps naturally (missing rows = complete draws)
- Full combination context
- ~500K rows is fine for pandas

### Draw List vs Draw Range

**In `process_single_combination()`:**
```python
def process_single_combination(..., draws_list=None):
    if draws_list is not None:
        # Use explicit list: [0, 1, 2, 15, 16, 17]
        missing_draws = [d for d in draws_list if d not in existing]
    else:
        # Use range: all draws from start to end
        missing_draws = [d for d in range(start, end+1) if d not in existing]
```

**Why this matters:**
- Task-based mode passes explicit `draws_list` from CSV
- Direct mode uses range (for manual testing)
- Explicit lists avoid processing unwanted draws in gaps

### Resource Estimation

**Memory formula (based on time period duration):**
```python
memory_gib = (2.4066 + 0.5090 * years + 1.3232) * 1.25
# All 5-year periods → 8G
```

**Runtime formula (based on storm count):**
```python
runtime_min = ((29.18 + 0.393 * storms) + 6.59) * (draws/2) * 1.25
runtime_hours = min(ceil(runtime_min / 60), 6)  # CAPPED AT 6 HOURS
```

**Reality check:**
- Formula was fitted on old data
- Actual performance: ~1.2 seconds per storm
- 235 storms × 25 draws × 1.2s = 2 hours (not 34!)
- 6-hour cap prevents waste

### File Permissions

All output files use **775 permissions** for team collaboration:
- `draw_status.csv`: 775
- `.draw_####.complete`: 775
- Zarr stores: 775
- Set via: `os.chmod(path, 0o775)`

### Path Structure

**Directory hierarchy:**
```
CLIMADA_INPUT_PATH/
└── cmip6/
    ├── task_assignments.csv          # Task distribution
    └── {model}/
        └── {variant}/
            └── {scenario}/
                └── {time_period}/
                    └── {basin}/
                        ├── draw_status.csv              # Status tracking
                        ├── .draw_0000.complete           # Markers
                        ├── .draw_0001.complete
                        ├── tracks_*_e000.zarr/           # Output data
                        └── tracks_*_e001.zarr/
```

### Constants

**From `constants.py`:**
- `NUM_DRAWS = 250` (draws per basin/combination)
- `tc_risk_n_procs = 4` (Dask workers)
- Cores requested = workers + 1 scheduler = 5

**From `00_orchestrator.py`:**
- `DRAWS_PER_BATCH = 25` (draws per task)
- `DATA_SOURCE = "cmip6"`
- Expected tasks = 250 draws / 25 per batch = 10 tasks per complete combination

### Basins
Seven ocean basins tracked:
- AU: South Pacific (Australia)
- EP: Eastern Pacific
- NA: North Atlantic (⚠️ Watch for pandas NaN!)
- NI: North Indian
- SI: South Indian
- SP: South Pacific
- WP: Western Pacific

---

## 🚀 Ready to Run

### Prerequisites
1. Level 0-3 completed (or running Level 4 independently)
2. `task_assignments.csv` exists with current state
3. Time bins CSV available at: `/mnt/team/rapidresponse/pub/tropical-storms/tempestextremes/outputs/cmip6/bayespoisson_time_bins_wide_max_bin_5.csv`

### To Run Full Level 4
```bash
cd /ihme/homes/bcreiner/repos/idd-climate-models/src/idd_climate_models/01_process_raw_through_climada_input
python 00_orchestrator.py
```
- Set `STARTING_LEVEL = 4` and `ENDING_LEVEL = 4` in orchestrator
- Will create 21,338 Jobmon tasks
- Each task runs: `05_run_basin_tc_risk.py --task_id <id> --total_memory 8G`

### To Refresh Task Assignments
```bash
cd /ihme/homes/bcreiner/repos/idd-climate-models/src/idd_climate_models/01_process_raw_through_climada_input

# First: Update all draw_status.csv files from markers
python 00_orchestrator.py  # With STARTING_LEVEL=0, ENDING_LEVEL=0

# Then check the task count
python -c "
import pandas as pd
df = pd.read_csv('/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/task_assignments.csv', keep_default_na=False)
print(f'Tasks: {df[\"task_id\"].nunique()}')
print(f'Draws: {len(df)}')
"
```

### To Test a Single Task
```bash
cd /ihme/homes/bcreiner/repos/idd-climate-models/src/idd_climate_models/01_process_raw_through_climada_input

# Find a task with few draws
python -c "
import pandas as pd
df = pd.read_csv('/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/task_assignments.csv', keep_default_na=False)
sizes = df.groupby('task_id').size()
min_task = sizes.idxmin()
print(f'Smallest task: {min_task} ({sizes.min()} draws)')
print(df[df['task_id'] == min_task])
"

# Run that task
python 05_run_basin_tc_risk.py --task_id <id> --total_memory 20G --data_source cmip6
```

---

## 🔍 Debugging Tips

### Check completion status
```bash
# Count completion markers for a combination
find /mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/CMCC-ESM2/r1i1p1f1/ssp126/2020-2024/NI/ -name ".draw_*.complete" | wc -l

# Check draw_status.csv
cat /mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/CMCC-ESM2/r1i1p1f1/ssp126/2020-2024/NI/draw_status.csv | grep "^249,"
```

### Verify task assignments
```bash
# Check specific combination
grep "CMCC-ESM2,r1i1p1f1,ssp126,2020-2024,NI" /mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/task_assignments.csv

# Count tasks by size
python -c "
import pandas as pd
df = pd.read_csv('/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/task_assignments.csv', keep_default_na=False)
print(df.groupby('task_id').size().value_counts().sort_index())
"
```

### Check for NA basin issues
```bash
# If NA basin is missing, check for keep_default_na=False
grep -n "pd.read_csv" 05_run_basin_tc_risk.py | grep -v "keep_default_na=False"
```

---

## 📝 Next Steps / TODO

Nothing critical - system is ready to run Level 4!

**Optional improvements:**
1. Update runtime formula based on actual performance (currently capped at 6h)
2. Add progress tracking dashboard for Level 4 execution
3. Consider dynamic batching based on storm count (high-storm basins = smaller batches)

---

## 🆘 If Starting Fresh Session

**Context you need:**
1. This is a distributed TC-risk pipeline using Jobmon
2. Level 4 has 21,338 parallel tasks processing 508,311 incomplete draws
3. Completion tracked via atomic `.draw_####.complete` marker files
4. Task assignments in CSV: one row per draw with full combination context
5. Key bug: pandas treats "NA" as NaN → use `keep_default_na=False`
6. Scripts: `00_orchestrator.py` (master), `05_run_basin_tc_risk.py` (executor)
7. System is tested and ready to run

**Quick health check:**
```bash
# Verify task assignments exist and have correct structure
python -c "
import pandas as pd
df = pd.read_csv('/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/task_assignments.csv', keep_default_na=False)
print(f'Columns: {list(df.columns)}')
print(f'Tasks: {df[\"task_id\"].nunique()}')
print(f'Draws: {len(df)}')
print(f'Basins: {sorted(df[\"basin\"].unique())}')
"
```

Expected output:
```
Columns: ['task_id', 'model', 'variant', 'scenario', 'time_period', 'basin', 'draw']
Tasks: 21338
Draws: 508311
Basins: ['AU', 'EP', 'NA', 'NI', 'SI', 'SP', 'WP']
```

If NA is missing from basins list → check for `keep_default_na=False` issue!