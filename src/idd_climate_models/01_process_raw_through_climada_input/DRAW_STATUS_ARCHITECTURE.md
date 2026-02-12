# New Draw Status & Task Assignment Architecture

## Overview

This implements a robust, resumable system for running basin TC-risk draws using status tracking files and task assignments.

## Key Changes

### 1. Draw Status Files (`draw_status.csv`)
- **Location**: `CLIMADA_INPUT_PATH/{model}/{variant}/{scenario}/{time_period}/{basin}/draw_status.csv`
- **Why CLIMADA?** A draw is only complete when BOTH NetCDF (in tc_risk/output) AND Zarr (in climada/input) exist and validate
- **Format**: CSV with columns `draw` (0-249) and `complete` (0 or 1)
- **Created by**: `00_create_draw_status.file.py` (Level 0-A in orchestrator)

### 2. Task Assignments File (`level_4_task_assignments.csv`)
- **Location**: `CLIMADA_INPUT_PATH/{model}/{variant}/{scenario}/{time_period}/level_4_task_assignments.csv`
- **Format**: CSV with columns `task_id`, `basin`, `draw`
- **Purpose**: Maps each task_id to specific draws it should run, distributing incomplete draws evenly
- **Created by**: `00_create_level_4_task_assignments.py` (Level 0-B in orchestrator)

## Workflow

### Level 0-A: Create Draw Status Files (Parallel)
```bash
# One task per basin - runs in parallel
python 00_create_draw_status_file.py \
    --data_source cmip6 \
    --model EC-Earth3 \
    --variant r1i1p1f1 \
    --scenario historical \
    --time_period 1970-1974 \
    --basin SP
```

**What it does:**
- Validates all existing NetCDF files (checks if openable)
- Checks if corresponding Zarr files exist
- Marks draw as complete (1) only if both exist and validate
- Deletes corrupted files automatically
- Creates `draw_status.csv` with 250 rows

### Level 0-B: Create Task Assignments (Single Task)
```bash
# Reads all draw_status.csv files and creates task assignments
python 00_create_level_4_task_assignments.py \
    --data_source cmip6 \
    --model EC-Earth3 \
    --variant r1i1p1f1 \
    --scenario historical \
    --time_period 1970-1974 \
    --basins EP NA NI SI AU SP WP \
    --draws_per_batch 25
```

**What it does:**
- Reads all `draw_status.csv` files for specified basins
- Collects all incomplete draws (where `complete == 0`)
- Distributes them evenly across tasks (~25 draws per task)
- **Crucially**: Assigns specific draws to each task_id (not ranges!)
- Creates `level_4_task_assignments.csv`

**Example output:**
```csv
task_id,basin,draw
0,SP,13
0,SP,47
0,SP,102
...  (25 draws for task 0)
1,SP,5
1,SP,88
...  (25 draws for task 1)
2,NA,3
2,NA,12
...
```

### Level 4: Run Basin TC-Risk (With Task IDs)
```bash
# Each task gets a task_id and looks up its draws
python 05_run_basin_tc_risk.py \
    --data_source cmip6 \
    --model EC-Earth3 \
    --variant r1i1p1f1 \
    --scenario historical \
    --time_period 1970-1974 \
    --task_id 0 \  # <-- NEW: task looks up its draws from file
    --total_memory 25G
```

**What it does:**
1. Reads `level_4_task_assignments.csv` for its task_id
2. Gets list of draws to run (e.g., [13, 47, 102, ...])
3. For each draw in that list:
   - Checks `draw_status.csv` - skip if already complete
   - Runs TC-risk for JUST THIS ONE DRAW
   - Validates the output immediately
   - Updates `draw_status.csv` to mark draw complete (with file locking)
   - Moves to next draw
4. If job times out and restarts, it automatically skips completed draws

## Benefits

### 1. Automatic Corruption Detection
- Level 0-A validates ALL files and deletes corrupted ones
- No more "HDF error -101" surprises during jobs

### 2. Even Load Distribution
- Tasks don't have fixed draw ranges (0-24, 25-49, etc.)
- Incomplete draws distributed evenly: Task 0 might run draws [13, 47, 102, ...]
- No tasks submitted if all draws in that task are complete

### 3. Resumable After Timeouts
- Job times out after completing 15/25 draws?
- Restarts and checks status file
- Skips those 15, runs the remaining 10
- Updates status file after each draw

### 4. Concurrent-Safe
- Uses `filelock` library for status file updates
- Multiple tasks can update same basin's status file safely

### 5. No Wasted Computation  
- `RERUN_ALL_BASINS = False` now works correctly
- Only missing draws get rerun
- Status file always reflects reality

## Orchestrator Integration

```python
# Level 0-A: Create draw status files (one task per basin, in parallel)
for task in level4_basin_tasks:
    status_task = status_template.create_task(
        model=task['model'],
        variant=task['variant'],
        scenario=task['scenario'],
        time_period=task['time_period'],
        basin=task['basin']
    )

# Level 0-B: Create task assignments (one task per time_period)
for mvst in unique_mvsts:
    assign_task = assign_template.create_task(
        model=mvst[0],
        variant=mvst[1],
        scenario=mvst[2],
        time_period=mvst[3],
        basins=' '.join(BASINS)
    )
    # Depends on ALL status tasks for this time_period
    
# Level 4: Read task assignments and submit only needed tasks
assignments_df = pd.read_csv(level_4_task_assignments_file)
for task_id in assignments_df['task_id'].unique():
    basin_task = basin_template.create_task(
        model=model,
        variant=variant,
        scenario=scenario,
        time_period=time_period,
        task_id=task_id  # <-- No more draw ranges!
    )
    # Depends on Level 0-B task
```

## File Locations Summary

```
CLIMADA_INPUT_PATH/cmip6/
└── EC-Earth3/
    └── r1i1p1f1/
        └── historical/
            └── 1970-1974/
                ├── level_4_task_assignments.csv          # Level 0-B output
                ├── SP/
                │   ├── draw_status.csv          # Level 0-A output
                │   ├── draw_status.lock         # File lock
                │   └── tracks_*.zarr/           # Validated zarr files
                ├── NA/
                │   ├── draw_status.csv
                │   └── ...
                └── ...
```

## Migration Notes

**Don't forget to:**
1. Install filelock: `pip install filelock` (or run from project root after updating pyproject.toml)
2. Update existing orchestrator to add Level 0-A and 0-B
3. Refactor `05_run_basin_tc_risk.py` to use task_id instead of draw ranges
4. Update jobmon templates to pass task_id instead of draw_start/draw_end

## Testing

Test with a single basin first:
```bash
# 1. Create status file
python 00_create_draw_status_file.py --data_source cmip6 --model EC-Earth3 --variant r1i1p1f1 --scenario historical --time_period 1970-1974 --basin SP

# 2. Create task assignments
python 00_create_level_4_task_assignments.py --data_source cmip6 --model EC-Earth3 --variant r1i1p1f1 --scenario historical --time_period 1970-1974 --basins SP --draws_per_batch 25

# 3. Check the files
cat /mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/EC-Earth3/r1i1p1f1/historical/1970-1974/SP/draw_status.csv
cat /mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/EC-Earth3/r1i1p1f1/historical/1970-1974/level_4_task_assignments.csv
```
