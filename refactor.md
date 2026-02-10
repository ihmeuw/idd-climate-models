Here's a detailed prompt to resume your orchestrator refactoring work:

CONTEXT: TC-Risk Pipeline Orchestrator Refactoring

I'm working on the idd-climate-models project at idd-climate-models. This processes climate model data through a TC-risk pipeline.

RECENTLY COMPLETED:
✅ Fixed ALL namelist references in vendored TC-risk code (52 references across basins.py, geo.py, thermo.py, calc_thermo.py, compute.py, coupled_fast.py)
✅ Updated default_config.json with all missing parameters from namelist.py
✅ All functions now accept and pass config_dict parameter
✅ Scripts 04_run_global_tc_risk.py and 05_run_basin_tc_risk.py are ready to run

CURRENT FOCUS: Orchestrator Refactoring

I created a new folder structure at 01_process_raw_to_climada_input with:

README.md documenting the new pipeline
00_orchestrator.py skeleton
ORCHESTRATOR REQUIREMENTS:
The orchestrator at 00_orchestrator.py (lines 1-513) needs refactoring to:

Add cleaning logic before TC-risk pipeline:

Currently starts at Level 2 (process_variable)
Need to add Level 1: Clean raw CMIP6 data and create yearly splits
Reference: 01_fill_and_yearly_split_parallel.ipynb
Key Jobmon workflow executes:

Level 2: 02_process_variable.py (interpolation, processing)
Level 3: 04_run_global_tc_risk.py (global TC tracks, recently fixed)
Level 4: 05_run_basin_tc_risk.py (basin-specific tracks with 250 draws)
Level 5: CLIMADA risk calculations (future work)
Critical paths:

Input: /mnt/share/homes/bcreiner/data/climate_model_data/raw/{data_source}/{model}/...
Processed: /mnt/share/homes/bcreiner/data/climate_model_data/processed/{data_source}/...
TC-risk outputs: /mnt/share/homes/bcreiner/data/tc_risk_model/{data_source}/...
Recent error context:

✅ Fixed NameError issues with namelist references 
✅ Fixed missing config_dict in calc_T_rho() call in thermo.py
✅ Fixed missing config_dict in get_fn_thermo() calls (2 locations in compute.py)
✅ Fixed missing config_dict in get_env_wnd_fn() calls (2 locations in compute.py)
Pipeline tested and all config_dict parameters now properly passed
NEXT STEPS:
Help me refactor the orchestrator to add the cleaning/yearly-split preprocessing level and ensure smooth flow through all 5 levels. The key files are:

00_orchestrator.py
02_clean_and_create_input.py (may need creation)
Reference notebook: 01_fill_and_yearly_split_parallel.ipynb