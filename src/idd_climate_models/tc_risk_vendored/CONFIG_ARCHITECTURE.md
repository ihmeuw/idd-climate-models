# TC-Risk Configuration Architecture

## Problem Solved

**Race Condition**: Multiple parallel jobs trying to create/read Python namelist files leads to:
- ModuleNotFoundError when Dask workers spawn as separate processes
- Incorrect time bounds (workers loading wrong namelist)
- File creation/deletion conflicts

## Solution

**Dictionary-Based Configuration** (NO Python namelist files!)

### Architecture

```
default_config.json  (static template)
        ↓
create_tc_risk_config()  (customize for job)
        ↓
config_dict  (in memory, job-specific)
        ↓
execute_tc_risk_with_config()
        ↓
compute.compute_downscaling_inputs(config_dict)
        ↓
env_wind.gen_wind_mean_cov(config_dict)
        ↓
Dask workers receive config_dict as parameter
        ↓
NO IMPORTS!
```

### Key Files

1. **default_config.json**: Static JSON template with defaults
2. **config_utils.py**: Helper functions to create/customize config dicts
3. **tc_risk_functions.py**: 
   - `create_tc_risk_config_dict(args)` - Creates job-specific dict
   - `execute_tc_risk_with_config(dict)` - Runs TC-risk with dict
4. **04_run_global_tc_risk.py**: Updated to use dict approach
5. **05_run_basin_tc_risk.py**: Updated to use dict approach

### Benefits

- ✅ **No race conditions**: Each job has its own in-memory dict
- ✅ **No file conflicts**: No namelist.py creation/deletion
- ✅ **No import errors**: Workers receive dict as parameter, never import
- ✅ **Parallel-safe**: Multiple jobs can run simultaneously
- ✅ **Simpler**: No sys.modules manipulation or dynamic imports

### Usage Example

```python
# OLD WAY (causes race conditions)
create_custom_namelist(args)  # Creates namelist.py file
execute_tc_risk(args)  # Imports the file

# NEW WAY (parallel-safe)
config_dict = create_tc_risk_config_dict(args)  # Creates dict in memory
execute_tc_risk_with_config(config_dict)  # Passes dict as parameter
```

### Configuration Dict Structure

```json
{
  "model": "CMCC-ESM2",
  "variant": "r1i1p1f1",
  "scenario": "historical",
  "time_period": "1986-2014",
  "basin": "GL",
  "start_year": 1986,
  "end_year": 2014,
  "base_directory": "/path/to/input/data",
  "output_directory": "/path/to/output",
  "exp_prefix": "CMCC-ESM2_historical_r1i1p1f1",
  "n_procs": 10,
  "dataset_type": "CMIP6",
  "var_keys": { ... },
  "tracks_per_year": 123,
  ...
}
```

## Backward Compatibility

Old functions remain for reference but are deprecated:
- `create_custom_namelist()` ❌ DEPRECATED
- `execute_tc_risk()` ❌ DEPRECATED

Use new functions instead:
- `create_tc_risk_config_dict()` ✅ USE THIS
- `execute_tc_risk_with_config()` ✅ USE THIS
