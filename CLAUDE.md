# Project: IDD Climate Models

## Session Memory Instructions
**IMPORTANT:** Update `.claude/memory.md` after EVERY substantive action (running commands, completing task steps, changing focus). This ensures context is preserved if the session dies unexpectedly.

The memory file should always contain:
1. Current task summary
2. Why we're doing it
3. Next steps
4. A ready-to-paste resume prompt

## Project Overview
Climate modeling pipeline for infectious disease dynamics (IDD). Key components:
- `src/idd_climate_models/01_process_raw_through_climada_input/` - Process raw climate data for TC input
- `src/idd_climate_models/01_run_tc_risk/` - Run tropical cyclone risk models
- `src/idd_climate_models/02_run_climada/` - Run CLIMADA impact modeling

## Environment
- Conda environment: `idd-climate-models`
- Activate with: `source /ihme/homes/bcreiner/miniconda/etc/profile.d/conda.sh && conda activate idd-climate-models`
- Uses IHME's jobmon for workflow orchestration
