"""
Orchestrator script for the full TC-risk pipeline.

This script creates a single Jobmon workflow with 4 levels:
  Level 1: Create folders (tc_risk input/output)
  Level 2: Process raw files directly to time-period files (one task per variable per time_period)
  Level 3: Run global TC-risk
  Level 4: Run basin TC-risk (one task per basin)

Time periods are model/variant/scenario-specific, loaded from the BayesPoisson
changepoint detection results in time_bins.csv.
"""

import sys
import os
import getpass
import uuid
import pandas as pd 
from pathlib import Path
from jobmon.client.tool import Tool

import idd_climate_models.constants as rfc
import idd_climate_models.orchestrator_utils as utils
from idd_climate_models.dictionary_utils import parse_results
from idd_climate_models.io_compare_utils import compare_model_validation
from idd_climate_models.resource_functions import (
    get_level2_resources, 
    get_level3_resources, 
    get_level4_resources
)
from idd_climate_models.time_period_functions import get_time_bins_path

# --- CONSTANT DEFINITIONS ---
repo_name = rfc.repo_name
package_name = rfc.package_name
RAW_DATA_PATH = rfc.RAW_DATA_PATH
PROCESSED_DATA_PATH = rfc.PROCESSED_DATA_PATH
TC_RISK_INPUT_PATH = rfc.TC_RISK_INPUT_PATH
TC_RISK_OUTPUT_PATH = rfc.TC_RISK_OUTPUT_PATH
TIME_BINS_DF_PATH = rfc.TIME_BINS_DF_PATH

hard_coded_level_4_memory = "35G"
# hard_coded_level_4_dask_memory = "30G"
hard_coded_level_4_dask_memory = hard_coded_level_4_memory

BUFFER = 0.2
project = "proj_rapidresponse"
queue = 'all.q'
# Script locations
SCRIPT_ROOT = rfc.REPO_ROOT / repo_name / "src" / package_name / "01_process_raw_through_climada_input"

# Configuration
DATA_SOURCE = "cmip6"
NUM_DRAWS = rfc.NUM_DRAWS
DRAWS_PER_BATCH = 10
MAX_PERIOD_DURATION = 5  # Maximum time period duration in years (None = use original bins)
TEST_RUN = 0  # Set to 0 for full run, >0 for test runs
STARTING_LEVEL = 4  # Level to start execution from (0, 1, 2, 3, or 4)
ENDING_LEVEL = 4    # Level to end execution at (0, 1, 2, 3, or 4)
ADD_DEPENDENCIES = True  # Set to False to run levels independently without upstream dependencies
BASINS = ['EP', 'NA', 'NI', 'SI', 'AU', 'SP', 'WP']
VERBOSE = False
DELETE_EXISTING_FOLDERS = True

DRY_RUN = False

# Validation configuration
INPUT_DATA_TYPE = "data"
INPUT_IO_TYPE = "raw"


# ============================================================================
# STAGE 1: LOAD TIME BINS (Model/Variant/Scenario-specific)
# ============================================================================

print("=" * 80)
print("STEP 1: Loading time bins from BayesPoisson changepoint detection")
print("=" * 80)

# Get appropriate time bins file (chunked if MAX_PERIOD_DURATION is set)
time_bins_path = get_time_bins_path(MAX_PERIOD_DURATION)
time_bins_df = pd.read_csv(time_bins_path)

# Filter to BayesPoisson method only (already filtered if using chunked file)
if MAX_PERIOD_DURATION is None:
    time_bins_df = time_bins_df[time_bins_df['method'] == 'BayesPoisson']

# Get unique (model, variant, scenario, time_period) combinations
time_bins_df['time_period'] = time_bins_df['start_year'].astype(str) + '-' + time_bins_df['end_year'].astype(str)

unique_time_bins = time_bins_df[['model', 'variant', 'scenario', 'time_period', 'start_year', 'end_year']].drop_duplicates()

print(f"Loaded {len(unique_time_bins)} unique (model, variant, scenario, time_period) combinations")

# Create a lookup: (model, variant, scenario) -> list of time_periods
time_bins_lookup = {}
for _, row in unique_time_bins.iterrows():
    key = (row['model'], row['variant'], row['scenario'])
    if key not in time_bins_lookup:
        time_bins_lookup[key] = []
    time_bins_lookup[key].append({
        'time_period': row['time_period'],
        'start_year': row['start_year'],
        'end_year': row['end_year']
    })

print(f"Found {len(time_bins_lookup)} unique (model, variant, scenario) combinations with time bins")

# ============================================================================
# STAGE 1.5: FILTER TO RERUN COMBINATIONS (if enabled)
# ============================================================================

# Initialize rerun_set to avoid NameError (used in Stage 5)
rerun_set = set()

# ============================================================================
# STAGE 2: VALIDATE RAW DATA (Only needed if Level 1 or 2 will run)
# ============================================================================

if STARTING_LEVEL <= 2:
    print("\n" + "=" * 80)
    print("STEP 2: Validating raw data completeness")
    print("=" * 80)

    validation_info = compare_model_validation(
        input_data_type=INPUT_DATA_TYPE,
        input_io_type=INPUT_IO_TYPE,
        output_data_type="tc_risk",
        output_io_type="input",
        data_source=DATA_SOURCE,
        verbose=VERBOSE,
        rerun_all=True,
    )

    input_complete_model_variants = validation_info["input_complete_model_variants"]
    print(f"\nFound {len(input_complete_model_variants)} complete model/variant combinations in raw data")

    # Get the full variable details for each model/variant
    full_path_list = parse_results(validation_info["input_validation_dict"], 'all')

    # Build variable detail map: {(model, variant, scenario, variable): {'grid': 'gn', 'frequency': 'day', 'files': [...]}}
    variable_detail_map = {}
    for item in full_path_list:
        key = (item['model'], item['variant'], item['scenario'], item['variable'])

        # Get raw files for this variable
        raw_dir = RAW_DATA_PATH / DATA_SOURCE / item['model'] / item['variant'] / item['scenario'] / item['variable'] / item['grid'] / item['frequency']
        raw_files = sorted(raw_dir.glob("*.nc")) if raw_dir.exists() else []

        variable_detail_map[key] = {
            'grid': item['grid'],
            'frequency': item['frequency'],
            'raw_files': raw_files
        }
else:
    # Skip validation but still need variable_detail_map for Level 2 if it runs
    input_complete_model_variants = []
    variable_detail_map = {}

# ============================================================================
# STAGE 3: CROSS-REFERENCE AND BUILD TASK LIST
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Cross-referencing validation with time bins")
print("=" * 80)

# Find model/variants that are both complete in raw data AND have time bins
tasks_to_run = []

if STARTING_LEVEL <= 2:
    for model_variant_tuple in input_complete_model_variants:
        model_name, variant_name = model_variant_tuple

        for scenario in rfc.SCENARIOS:
            key = (model_name, variant_name, scenario)

            if key not in time_bins_lookup:
                if VERBOSE:
                    print(f"  Skipping {model_name}/{variant_name}/{scenario}: no time bins defined")
                continue

            time_periods = time_bins_lookup[key]

            for tp_info in time_periods:
                tasks_to_run.append({
                    'model': model_name,
                    'variant': variant_name,
                    'scenario': scenario,
                    'time_period': tp_info['time_period'],
                    'start_year': tp_info['start_year'],
                    'end_year': tp_info['end_year'],
                })
else:
    # For levels 3+ only, build task list from the FILTERED time_bins_df
    unique_tasks = time_bins_df[['model', 'variant', 'scenario', 'time_period', 'start_year', 'end_year']].drop_duplicates()
    
    for _, row in unique_tasks.iterrows():
        tasks_to_run.append({
            'model': row['model'],
            'variant': row['variant'],
            'scenario': row['scenario'],
            'time_period': row['time_period'],
            'start_year': row['start_year'],
            'end_year': row['end_year'],
        })

print(f"\nTotal tasks: {len(tasks_to_run)} (model/variant/scenario/time_period combinations)")

if not tasks_to_run:
    print("\nNo tasks to run. Exiting.")
    sys.exit(0)

# Apply test run limit if specified
if TEST_RUN > 0:
    print(f"\nTEST MODE: Limiting to {TEST_RUN} tasks")
    tasks_to_run = tasks_to_run[:TEST_RUN]

# Validate level configuration
if STARTING_LEVEL < 0 or ENDING_LEVEL > 4 or STARTING_LEVEL > ENDING_LEVEL:
    print(f"\nERROR: Invalid level configuration. STARTING_LEVEL={STARTING_LEVEL}, ENDING_LEVEL={ENDING_LEVEL}")
    print("Valid range: 0-4 and STARTING_LEVEL <= ENDING_LEVEL")
    sys.exit(1)

print(f"\nRunning Levels {STARTING_LEVEL} through {ENDING_LEVEL}")
print(f"Add dependencies between levels: {ADD_DEPENDENCIES}")

# ============================================================================
# STAGE 4: FILTER TASKS BASED ON STARTING LEVEL
# ============================================================================

# Initialize variables that may be used later (avoid NameError)
level3_finished = []
level3_not_finished = []
level3_tasks_to_create = []
level4_tasks_to_check = []

if STARTING_LEVEL >= 3:
    print("\n" + "=" * 80)
    print("STEP 4: Filtering tasks based on current Level 3 and Level 4 status")
    print("=" * 80)
    
    # Identify which tasks have finished L3 and which haven't
    tc_risk_base = TC_RISK_OUTPUT_PATH / DATA_SOURCE
    for task in tasks_to_run:
        if utils.check_level3_output_finished(
            task['model'], task['variant'], task['scenario'], 
            task['time_period'], tc_risk_base
        ):
            level3_finished.append(task)
        else:
            level3_not_finished.append(task)
    
    print(f"\nLevel 3 status across all tasks:")
    print(f"  Finished:     {len(level3_finished)}")
    print(f"  Not finished: {len(level3_not_finished)}")
    
    # ========== LOGIC FOR STARTING_LEVEL = 3 ==========
    if STARTING_LEVEL == 3:
        print(f"\n✓ STARTING_LEVEL = 3")
        print(f"  - Will RUN Level 3 for the {len(level3_not_finished)} unfinished tasks")
        print(f"  - Will RUN Level 4 for both:")
        print(f"    (a) L4 basins under the {len(level3_not_finished)} L3 tasks being run (with L3 dependency)")
        print(f"    (b) L4 basins that haven't passed L4 test under finished L3 tasks")
        
        # Tasks for which we'll create Level 3 (unfinished only)
        level3_tasks_to_create = level3_not_finished
        
        # Tasks for which we'll check and potentially create Level 4
        # = all tasks (both finished and unfinished L3)
        level4_tasks_to_check = tasks_to_run
        
        print(f"\nLevel 3 tasks to create: {len(level3_tasks_to_create)}")
        if level3_tasks_to_create and VERBOSE:
            print(f"  First 10:")
            for task in level3_tasks_to_create[:10]:
                print(f"    {task['model']}/{task['variant']}/{task['scenario']}/{task['time_period']}")
    
    # ========== LOGIC FOR STARTING_LEVEL = 4 ==========
    elif STARTING_LEVEL == 4:
        print(f"\n✓ STARTING_LEVEL = 4")
        print(f"  - Will SKIP Level 3 entirely")
        print(f"  - Will RUN Level 4 ONLY for basins where:")
        print(f"    (a) The time-period passed L3 test ({len(level3_finished)} tasks)")
        print(f"    (b) The basin failed L4 test (NUM_DRAWS+1 files)")
        
        # Tasks for which we'll check Level 4
        # = only finished L3 tasks
        level4_tasks_to_check = level3_finished
        
        print(f"\nLevel 4 will be checked for: {len(level4_tasks_to_check)} tasks with finished L3")

# ============================================================================
# STAGE 4B: DETERMINE WHICH BASINS NEED DRAW STATUS FILES (LEVEL 0 ONLY)
# ============================================================================
level0_basin_tasks = []

if STARTING_LEVEL <= 0 and ENDING_LEVEL >= 0:
    print("\n" + "=" * 80)
    print("STEP 4B: Determining basins for Level 0 (draw status files)")
    print("=" * 80)
    
    # For Level 0, we need to create draw status files for ALL basins
    # across all model/variant/scenario/time_period combinations
    for task in tasks_to_run:
        model, variant, scenario, time_period = task['model'], task['variant'], task['scenario'], task['time_period']
        
        basins_to_process = BASINS
        
        for basin in basins_to_process:
            level0_basin_tasks.append({
                'model': model,
                'variant': variant,
                'scenario': scenario,
                'time_period': time_period,
                'basin': basin
            })
    
    print(f"\nLevel 0 basin tasks (draw status files): {len(level0_basin_tasks)}")
    if VERBOSE and level0_basin_tasks:
        print(f"First 10:")
        for i, task in enumerate(level0_basin_tasks[:10]):
            print(f"  {task['model']}/{task['variant']}/{task['scenario']}/{task['time_period']}/{task['basin']}")

# ============================================================================
# STAGE 4C: CREATE LEVEL 4 TASK ASSIGNMENTS WHEN RUNNING LEVEL 1
# ============================================================================

if STARTING_LEVEL <= 1 and ENDING_LEVEL >= 1:
    print("\n" + "=" * 80)
    print("STEP 4C: Creating Level 4 task assignments (full run for all combinations)")
    print("=" * 80)
    
    # When running Level 1 (creating folders), we create task assignments for ALL draws
    # for ALL basins for ALL combinations in tasks_to_run
    
    print(f"\nCreating task assignments for {len(tasks_to_run)} combinations")
    print(f"  Basins: {', '.join(BASINS)}")
    print(f"  Draws per combination: {NUM_DRAWS}")
    print(f"  Draws per batch: {DRAWS_PER_BATCH}")
    
    # Calculate batches
    num_batches_per_basin = (NUM_DRAWS + DRAWS_PER_BATCH - 1) // DRAWS_PER_BATCH
    
    assignments = []
    task_id = 1
    
    for task in tasks_to_run:
        model = task['model']
        variant = task['variant']
        scenario = task['scenario']
        time_period = task['time_period']
        
        for basin in BASINS:
            for batch_idx in range(num_batches_per_basin):
                batch_start = batch_idx * DRAWS_PER_BATCH
                batch_end = min(batch_start + DRAWS_PER_BATCH, NUM_DRAWS)
                
                # Add all draws in this batch
                for draw in range(batch_start, batch_end):
                    assignments.append({
                        'task_id': task_id,
                        'model': model,
                        'variant': variant,
                        'scenario': scenario,
                        'time_period': time_period,
                        'basin': basin,
                        'draw': draw
                    })
                
                task_id += 1
    
    # Save to file
    assignments_df = pd.DataFrame(assignments)
    output_file = TC_RISK_INPUT_PATH / DATA_SOURCE / "level_4_task_assignments.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    assignments_df.to_csv(output_file, index=False)
    
    total_tasks = assignments_df['task_id'].nunique()
    print(f"\n✅ Created task assignments file:")
    print(f"   File: {output_file}")
    print(f"   Total tasks: {total_tasks}")
    print(f"   Total rows: {len(assignments_df)}")
    print(f"   Format: {len(tasks_to_run)} combinations × {len(BASINS)} basins × {num_batches_per_basin} batches = {total_tasks} tasks")

# ============================================================================
# JOBMON SETUP & TEMPLATES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Setting up Jobmon workflow")
print("=" * 80)

user = getpass.getuser()

wf_uuid = uuid.uuid4()
tool_name = f"{package_name}_tc_risk_pipeline"
tool = Tool(name=tool_name)

workflow = tool.create_workflow(
    name=f"{tool_name}_workflow_{wf_uuid}",
    max_concurrently_running=5000,
)

# Set Default Compute Resources
workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={
        "memory": "2G", "cores": 1, "runtime": "10m",
        "queue": queue, "project": project,
    }
)

# --- Template 0-A: Create Draw Status File (Level 0) ---
draw_status_template = tool.get_task_template(
    template_name="create_draw_status_file",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "2G", "cores": 1, "runtime": "1h",
        "queue": queue, "project": project,
    },
    command_template=(
        "python {script_root}/00_create_draw_status_file.py "
        "--data_source {{data_source}} "
        "--model {{model}} "
        "--variant {{variant}} "
        "--scenario {{scenario}} "
        "--time_period {{time_period}} "
        "--basin {{basin}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["data_source", "model", "variant", "scenario", "time_period", "basin"],
)

# --- Template 0-B: Create Task Assignments (Level 0) ---
task_assignment_template = tool.get_task_template(
    template_name="create_level_4_task_assignments",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "2G", "cores": 1, "runtime": "5m",
        "queue": queue, "project": project,
    },
    command_template=(
        "python {script_root}/00_create_level_4_task_assignments.py "
        "--data_source {{data_source}} "
        "--draws_per_batch {{draws_per_batch}} "
        "--max_period_duration {{max_period_duration}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["data_source", "draws_per_batch", "max_period_duration"],
)

# --- Template 1: Folder Creation (Level 1) ---
# Note: Deletion is now handled at orchestrator level (top-level data_source dirs)
# so we don't pass --delete_destination_folder to individual tasks

folder_template = tool.get_task_template(
    template_name="create_folders",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "1G", "cores": 1, "runtime": "5m",
        "queue": queue, "project": project,
    },
    command_template=(
        "python {script_root}/01_create_folders.py "
        "--data_source {{data_source}} "
        "--model {{model}} "
        "--variant {{variant}} "
        "--scenario {{scenario}} "
        "--time_period {{time_period}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["data_source", "model", "variant", "scenario", "time_period"],
)

# --- Template 2: Process Raw to Time-Period (Level 2) ---
process_variable_template = tool.get_task_template(
    template_name="process_variable",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "16G", "cores": 4, "runtime": "1h",
        "queue": queue, "project": project,
    },
    command_template=(
        "python {script_root}/02_process_variable.py "
        "--data_source {{data_source}} "
        "--model {{model}} "
        "--variant {{variant}} "
        "--scenario {{scenario}} "
        "--time_period {{time_period}} "
        "--variable {{variable}} "
        "--grid {{grid}} "
        "--frequency {{frequency}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["data_source", "model", "variant", "scenario", "time_period", "variable", "grid", "frequency"],
)

# --- Template 3: Global TC-Risk Run ---
global_run_template = tool.get_task_template(
    template_name="run_global_tc_risk",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "40G", "cores": rfc.tc_risk_n_procs + 4, "runtime": "1h",
        "queue": queue, "project": project,
    },
    command_template=(
        "python {script_root}/04_run_global_tc_risk.py "
        "--data_source {{data_source}} "
        "--model {{model}} "
        "--variant {{variant}} "
        "--scenario {{scenario}} "
        "--time_period {{time_period}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["data_source", "model", "variant", "scenario", "time_period"],
)

# --- Template 4: Basin TC-Risk Run (task-based only) ---
basin_run_template = tool.get_task_template(
    template_name="run_basin_tc_risk",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "25G", 
        "cores": rfc.tc_risk_n_procs + 4, 
        "runtime": "4h",
        "queue": queue, 
        "project": project,
    },
    command_template=(
        "python {script_root}/05_run_basin_tc_risk.py "
        "--data_source {{data_source}} "
        "--task_id {{task_id}} "
        "--total_memory {{total_memory}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["data_source", "task_id", "total_memory"],
)

# ============================================================================
# TASK CREATION AND NESTED DEPENDENCIES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Creating Jobmon tasks with dependencies")
print("=" * 80)

all_tasks = []
dependencies = []

# ========== LEVEL 0 ==========
level0_draw_status_tasks = []
level0_task_assignment_task = None

if STARTING_LEVEL <= 0 and ENDING_LEVEL >= 0:
    print(f"Creating Level 0 tasks: {len(level0_basin_tasks)} draw status files")
    
    # Level 0-A: Create draw status file for each basin
    for task_info in level0_basin_tasks:
        draw_status_task = draw_status_template.create_task(
            data_source=DATA_SOURCE,
            model=task_info['model'],
            variant=task_info['variant'],
            scenario=task_info['scenario'],
            time_period=task_info['time_period'],
            basin=task_info['basin'],
        )
        all_tasks.append(draw_status_task)
        level0_draw_status_tasks.append(draw_status_task)
    
    # Level 0-B: Create task assignments (depends on all draw status files)
    level0_task_assignment_task = task_assignment_template.create_task(
        data_source=DATA_SOURCE,
        draws_per_batch=str(DRAWS_PER_BATCH),
        max_period_duration=str(MAX_PERIOD_DURATION) if MAX_PERIOD_DURATION else "None",
    )
    all_tasks.append(level0_task_assignment_task)
    
    # Dependencies: task assignment depends on all draw status files
    if ADD_DEPENDENCIES:
        for draw_status_task in level0_draw_status_tasks:
            dependencies.append((level0_task_assignment_task, draw_status_task))
    
    print(f"  Created {len(level0_draw_status_tasks)} draw status tasks")
    print(f"  Created 1 task assignment task")

# ========== DELETE TOP-LEVEL DIRECTORIES (BEFORE LEVEL 1) ==========
if DELETE_EXISTING_FOLDERS and STARTING_LEVEL <= 1 and ENDING_LEVEL >= 1:
    print("\n" + "=" * 80)
    print("DELETING existing data_source directories before Level 1")
    print("=" * 80)
    
    import subprocess
    import time
    
    # Define the three top-level directories to delete
    dirs_to_delete = [
        TC_RISK_INPUT_PATH / DATA_SOURCE,
        TC_RISK_OUTPUT_PATH / DATA_SOURCE,
        rfc.CLIMADA_INPUT_PATH / DATA_SOURCE,
    ]
    
    renamed_count = 0
    for dir_path in dirs_to_delete:
        if dir_path.exists():
            # Fast delete: rename with timestamp, then delete in background
            suffix = f'_DELETE_{int(time.time() * 1000000)}'
            renamed_path = dir_path.parent / (dir_path.name + suffix)
            
            print(f"\n  Renaming for async deletion: {dir_path}")
            print(f"    → {renamed_path.name}")
            dir_path.rename(renamed_path)
            renamed_count += 1
            
            # Launch background deletion with low I/O priority
            subprocess.Popen(
                ['ionice', '-c', '3', 'rm', '-rf', str(renamed_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            print(f"\n  Directory doesn't exist (skip): {dir_path}")
    
    print(f"\n✅ Renamed {renamed_count} directories for background deletion")
    if renamed_count > 0:
        print(f"   (Level 1 folder creation will now create fresh directories)")
    else:
        print(f"   (No directories needed deletion)")

# ========== LEVELS 1 & 2 ==========
level2_tasks_dict = {}
if STARTING_LEVEL <= 2:
    for task_info in tasks_to_run:
        model_name = task_info['model']
        variant_name = task_info['variant']
        scenario_name = task_info['scenario']
        time_period_str = task_info['time_period']

        # Define model/variant/scenario/time_period key
        mvst_key = (model_name, variant_name, scenario_name, time_period_str)

        # --- LEVEL 1: FOLDER CREATION ---
        if STARTING_LEVEL <= 1 and ENDING_LEVEL >= 1:
            folder_task = folder_template.create_task(
                data_source=DATA_SOURCE,
                model=model_name,
                variant=variant_name,
                scenario=scenario_name,
                time_period=time_period_str,
            )
            all_tasks.append(folder_task)
        else:
            folder_task = None

        # --- LEVEL 2: PROCESS RAW FILES TO TIME-PERIOD FILES ---
        process_tasks_for_this_mvst_combo = []
        
        if STARTING_LEVEL <= 2 and ENDING_LEVEL >= 2:
            for variable in rfc.VARIABLES[DATA_SOURCE]:
                detail_lookup_key = (model_name, variant_name, scenario_name, variable)
                details = variable_detail_map.get(detail_lookup_key)

                if details is None:
                    if VERBOSE:
                        print(f"  Warning: No details found for {detail_lookup_key}, skipping")
                    continue

                grid = details['grid']
                frequency = details['frequency']

                resources = get_level2_resources(time_period_str, variable, frequency)

                process_task = process_variable_template.create_task(
                    compute_resources={
                        "memory": resources["memory"],
                        "cores": resources["cores"],
                        "runtime": resources["runtime"],
                        "queue": queue,
                        "project": project,
                    },
                    data_source=DATA_SOURCE,
                    model=model_name,
                    variant=variant_name,
                    scenario=scenario_name,
                    time_period=time_period_str,
                    variable=variable,
                    grid=grid,
                    frequency=frequency,
                )
                all_tasks.append(process_task)
                process_tasks_for_this_mvst_combo.append(process_task)

                # Dependency: process_task depends on folder_task
                if ADD_DEPENDENCIES and folder_task is not None and STARTING_LEVEL <= 1:
                    dependencies.append((process_task, folder_task))

            # Store process tasks for this combo
            if process_tasks_for_this_mvst_combo:
                level2_tasks_dict[mvst_key] = process_tasks_for_this_mvst_combo

# ========== LEVEL 3 ==========
level3_tasks_dict = {}  # Map (model, variant, scenario, time_period) -> task for dependencies

if STARTING_LEVEL <= 3 and ENDING_LEVEL >= 3:
    if STARTING_LEVEL == 3:
        tasks_for_level3 = level3_tasks_to_create
    else:
        tasks_for_level3 = tasks_to_run
    
    for task_info in tasks_for_level3:
        model_name = task_info['model']
        variant_name = task_info['variant']
        scenario_name = task_info['scenario']
        time_period_str = task_info['time_period']
        mvst_key = (model_name, variant_name, scenario_name, time_period_str)

        resources = get_level3_resources(time_period_str)

        global_task = global_run_template.create_task(
            compute_resources={
                "memory": resources["memory"],
                "cores": resources["cores"],
                "runtime": resources["runtime"],
                "queue": queue,
                "project": project,
            },
            data_source=DATA_SOURCE,
            model=model_name,
            variant=variant_name,
            scenario=scenario_name,
            time_period=time_period_str,
        )
        all_tasks.append(global_task)
        level3_tasks_dict[mvst_key] = global_task

        # Dependency: global_task depends on process_tasks (if Level 2 ran)
        if ADD_DEPENDENCIES and STARTING_LEVEL <= 2:
            # Find the process tasks for this combo
            if mvst_key in level2_tasks_dict:
                for process_task in level2_tasks_dict[mvst_key]:
                    dependencies.append((global_task, process_task))

# ========== LEVEL 4 (TASK-BASED) ==========
if STARTING_LEVEL <= 4 and ENDING_LEVEL >= 4:
    # Check if level_4_task_assignments.csv exists
    level_4_task_assignments_path = rfc.CLIMADA_INPUT_PATH / DATA_SOURCE / "level_4_task_assignments.csv"
    
    if not level_4_task_assignments_path.exists():
        print(f"\n⚠️  ERROR: level_4_task_assignments.csv not found at {level_4_task_assignments_path}")
        print("\nLevel 4 now requires task assignments. You have two options:")
        print("  1. Run orchestrator with STARTING_LEVEL=0 first to create draw status files and task assignments")
        print("  2. Manually create level_4_task_assignments.csv with columns: task_id, basin, draw")
        print("\nFor a simple 'run all draws' setup, you can create uniform task assignments:")
        print("  - Task 1: all basins, draws 0-24")
        print("  - Task 2: all basins, draws 25-49")
        print("  - etc.")
        sys.exit(1)
    
    print(f"\nUsing task assignments from {level_4_task_assignments_path}")
    
    # Read task assignments CSV to determine resources per task
    df_assignments = pd.read_csv(level_4_task_assignments_path, keep_default_na=False)
    total_tasks = df_assignments['task_id'].nunique()
    print(f"Total task IDs in assignments: {total_tasks}")
    
    # Create one Jobmon task per task_id with appropriate resources
    for task_id in range(1, total_tasks + 1):
        # Get the rows for this task
        task_rows = df_assignments[df_assignments['task_id'] == task_id]
        
        if task_rows.empty:
            print(f"Warning: No assignments found for task_id {task_id}, skipping")
            continue
        
        # Extract combination details (should be same for all rows in this task)
        model = task_rows['model'].iloc[0]
        variant = task_rows['variant'].iloc[0]
        scenario = task_rows['scenario'].iloc[0]
        time_period = task_rows['time_period'].iloc[0]
        basin = task_rows['basin'].iloc[0]
        num_draws = len(task_rows)  # Number of draws in this task
        
        # Calculate resources based on actual combination
        # resources = get_level4_resources(
        #     model=model,
        #     variant=variant,
        #     scenario=scenario,
        #     time_period=time_period,
        #     basin=basin,
        #     draws_per_batch=num_draws,
        #     verbose=False
        # )

        resources = {
            'memory': hard_coded_level_4_memory,  # Override with hard-coded memory
            'cores': rfc.tc_risk_n_procs + 1,
            'runtime': '2h',
        }

        basin_task = basin_run_template.create_task(
            compute_resources={
                "memory": resources['memory'], 
                "cores": resources["cores"],
                "runtime": resources["runtime"],
                "queue": queue,
                "project": project,
            },
            data_source=DATA_SOURCE,
            task_id=str(task_id),
            total_memory=hard_coded_level_4_dask_memory,
        )
        all_tasks.append(basin_task)
        
        # Dependencies: basin task depends on its specific Level 3 task
        if ADD_DEPENDENCIES:
            if level0_task_assignment_task is not None:
                # Depend on task assignment creation (transitively depends on all draw status files)
                dependencies.append((basin_task, level0_task_assignment_task))
            
            elif STARTING_LEVEL <= 3:
                # Depend ONLY on the specific Level 3 task for this combination
                mvst_key = (model, variant, scenario, time_period)
                if mvst_key in level3_tasks_dict:
                    dependencies.append((basin_task, level3_tasks_dict[mvst_key]))
    
    print(f"Created {total_tasks} task-based Level 4 tasks")

print(f"\nTotal tasks created: {len(all_tasks)}")
print(f"Total dependencies: {len(dependencies)}")
print(f"Levels executed: {STARTING_LEVEL} through {ENDING_LEVEL}")

# --- ADD TASKS AND BIND DEPENDENCIES ---
workflow.add_tasks(all_tasks)

if dependencies:
    print(f"Adding {len(dependencies)} dependencies using add_upstream...")
    for child_task, parent_task in dependencies:
        child_task.add_upstream(parent_task)
    print("Dependencies successfully added.")
else:
    print("No dependencies to add (independent execution).")


# ============================================================================
# DRY RUN: PRINT TASKS WITHOUT SUBMITTING
# ============================================================================
if DRY_RUN:
    print("\n" + "=" * 80)
    print("DRY RUN: Tasks that would be executed")
    print("=" * 80)
    
    print(f"\nTotal Jobmon tasks to create: {len(all_tasks)}")
    print(f"Total dependencies: {len(dependencies)}")
    print(f"Levels: {STARTING_LEVEL} through {ENDING_LEVEL}")
    
    # Show Level 4 task assignment summary if executing Level 4
    if ENDING_LEVEL >= 4:
        level_4_task_assignments_path = rfc.CLIMADA_INPUT_PATH / DATA_SOURCE / "level_4_task_assignments.csv"
        if level_4_task_assignments_path.exists():
            df_assignments = pd.read_csv(level_4_task_assignments_path, keep_default_na=False)
            total_task_ids = df_assignments['task_id'].nunique()
            total_draws = len(df_assignments)
            basins = df_assignments['basin'].unique()
            
            print(f"\nLevel 4 task assignments:")
            print(f"  Task IDs: {total_task_ids}")
            print(f"  Total draws: {total_draws}")
            print(f"  Basins: {', '.join(sorted(basins))}")
            
            # Show first few task assignments
            print(f"\n  First 5 task assignments:")
            for task_id in sorted(df_assignments['task_id'].unique())[:5]:
                task_df = df_assignments[df_assignments['task_id'] == task_id]
                basins_list = task_df['basin'].unique()
                draws_list = sorted(task_df['draw'].unique())
                print(f"    Task {task_id}: {len(draws_list)} draws across {len(basins_list)} basins (draws {draws_list[0]}-{draws_list[-1]})")

    print("\n" + "=" * 80)
    print("To actually run the workflow, comment out or remove the sys.exit() below")
    print("=" * 80)

    # Comment out the line below to proceed with submission
    sys.exit(0)

# ============================================================================
# SUBMISSION
# ============================================================================
else:
    print("\n" + "=" * 80)
    print("STEP 8: Binding and running workflow")
    print("=" * 80)

    try:
        workflow.bind()
        print("Workflow successfully bound.")
        print(f"Running workflow with ID {workflow.workflow_id}.")
        print("For full information see the Jobmon GUI:")
        print(f"https://jobmon-gui.ihme.washington.edu/#/workflow/{workflow.workflow_id}")
    except Exception as e:
        print(f"Workflow binding failed: {e}")
        sys.exit(1)

    try:
        status = workflow.run(seconds_until_timeout=172800)
        print(f"Workflow {workflow.workflow_id} completed with status {status}.")
    except Exception as e:
        print(f"Workflow submission failed: {e}")
        sys.exit(1)