# TC-Risk Vendored Code

This directory contains vendored code from the `tropical_cyclone_risk` repository.

## Why Vendored?

The original TC-risk code uses a global `namelist_loader` module that creates race
conditions when multiple jobs run in parallel. By vendoring the code, we can:

1. **Eliminate race conditions**: Pass namelist parameters as function arguments instead of importing from shared files
2. **Version control**: Lock in the exact TC-risk version we're using
3. **Isolation**: Each job runs with its own code, no shared state
4. **Maintainability**: Make changes without affecting other users of TC-risk

## Key Modifications

### 1. Import Changes
- All cross-module imports changed from absolute to relative
- Example: `from util import input` → `from ..util import input`

### 2. Namelist Parameter Passing
- **track/env_wind.py**: 
  - Modified `wnd_stat_wrapper()` to accept `(start_year, start_month, end_year, end_month)` as parameters
  - `gen_wind_mean_cov()` extracts these from namelist in parent process before spawning Dask workers
  - Workers receive values as function arguments, avoiding module import issues

### 3. Debug Logging
- Added `[DEBUG]` print statements for troubleshooting
- Can be removed once stable

## Original Source

- **Repository**: `/mnt/share/homes/bcreiner/repos/tropical_cyclone_risk`
- **Vendored Date**: 2026-02-03
- **Reason**: Fix race conditions from shared `namelist_loader.py` file

## Usage

```python
from idd_climate_models.tc_risk_vendored.util import compute
from idd_climate_models.tc_risk_vendored.scripts import generate_land_masks
from idd_climate_models.tc_risk_vendored.track import env_wind

# Namelist must be loaded in parent process and accessible to vendored code
import namelist_loader as namelist
sys.modules['namelist'] = namelist

# Then call vendored functions
compute.compute_downscaling_inputs()
```

## Architecture

```
tc_risk_vendored/
├── track/          # Storm tracking algorithms
│   ├── env_wind.py # Environmental wind statistics (MODIFIED)
│   └── bam_track.py
├── util/           # I/O and utilities
│   ├── input.py    # File loading
│   ├── compute.py  # Main computation entry points
│   └── ...
├── intensity/      # Storm intensity models
│   └── ...
└── scripts/        # Preprocessing scripts
    └── generate_land_masks.py
```

## Testing

After vendoring, test with a single job:
```bash
python src/idd_climate_models/01_run_tc_risk/04_run_global_tc_risk.py \
    --model CMCC-ESM2 --variant r1i1p1f1 --scenario historical \
    --time_period 1986-2014 --basin GL
```

## Maintenance

If you need to update from upstream TC-risk:
1. Re-copy the folders
2. Re-apply import fixes (sed commands in vendor script)
3. Re-apply namelist parameter passing changes to env_wind.py
4. Test thoroughly
