# Storm-Admin0 Impact Analysis Pipeline

Parallel analysis pipeline to create storm-admin0 impact dataframes across all draws, basins, and climate scenarios.

## Files

1. **orchestrator_storm_admin_analysis.py** - Main orchestrator that creates Jobmon workflow
2. **run_storm_admin_analysis.py** - Worker script executed by each Jobmon task

## How It Works

Uses the same task batching as the main TC-risk pipeline:
- Reads `level_4_task_assignments.csv` (created by main orchestrator Level 0)
- Each task processes multiple draws for a specific basin/model/variant/scenario/time_period
- Results are saved as CSV files per task

## Usage

### Step 1: Ensure task assignments exist

If not already created, run the main orchestrator with Level 0:
```bash
cd /ihme/homes/bcreiner/repos/idd-climate-models/src/idd_climate_models/01_process_raw_through_climada_input
python 00_orchestrator.py  # with STARTING_LEVEL=0
```

This creates: `/ihme/climate/tc_risk_output/climada/input/cmip6/level_4_task_assignments.csv`

### Step 2: Run the analysis orchestrator

```bash
cd /ihme/homes/bcreiner/repos/idd-climate-models/notebooks/visualizations
python orchestrator_storm_admin_analysis.py
```

### Step 3: Monitor progress

Check the Jobmon GUI link printed by the orchestrator.

### Step 4: Combine results

After completion, combine all task outputs:
```python
import pandas as pd
from pathlib import Path

output_dir = Path("/ihme/homes/bcreiner/repos/idd-climate-models/outputs/storm_admin_impacts")
all_files = sorted(output_dir.glob("task_*.csv"))

dfs = [pd.read_csv(f) for f in all_files]
combined_df = pd.concat(dfs, ignore_index=True)

combined_df.to_csv(output_dir / "all_storm_admin_impacts.csv", index=False)
print(f"Combined {len(all_files)} files into {len(combined_df)} rows")
```

## Configuration

Edit `orchestrator_storm_admin_analysis.py`:

- **TEST_RUN**: Set to N to process only first N tasks (0 = all)
- **TASK_MEMORY**: Memory per task (default: "10G")
- **TASK_CORES**: Cores per task (default: 2)
- **TASK_RUNTIME**: Runtime per task (default: "30m")

## Output Format

Each task creates a CSV with columns:
- `model`, `variant`, `scenario`, `time_period`, `basin`, `draw`
- `storm_track` - Storm index (0-39)
- `year`, `month` - Storm timing
- `ADM0_CODE`, `ADM0_NAME`, `loc_id` - Admin region
- `max_wind_speed` - Maximum windspeed over that admin region
- `storm_category` - Storm category at max windspeed

Multiple rows per storm if it crosses multiple countries.

## Performance

- ~1-5 minutes per task (10 draws)
- ~8-10 seconds per draw
- Can run 1000 tasks concurrently
- Full dataset (~3000 tasks) should complete in under 1 hour
