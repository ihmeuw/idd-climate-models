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
from idd_climate_models.dictionary_utils import parse_results
from idd_climate_models.io_compare_utils import compare_model_validation
from idd_climate_models.resource_functions import get_resource_info

# --- CONSTANT DEFINITIONS ---
repo_name = rfc.repo_name
package_name = rfc.package_name
RAW_DATA_PATH = rfc.RAW_DATA_PATH
PROCESSED_DATA_PATH = rfc.PROCESSED_DATA_PATH
TC_RISK_INPUT_PATH = rfc.TC_RISK_INPUT_PATH
TC_RISK_OUTPUT_PATH = rfc.TC_RISK_OUTPUT_PATH
TIME_BINS_DF_PATH = rfc.TIME_BINS_DF_PATH

# Script locations
SCRIPT_ROOT = rfc.REPO_ROOT / repo_name / "src" / package_name / "01_process_raw_through_tc_risk"

# Configuration
DATA_SOURCE = "cmip6"
NUM_DRAWS = rfc.NUM_DRAWS
TEST_RUN = 0  # Set to 0 for full run, >0 for test runs
STARTING_LEVEL = 4  # Level to start execution from (1, 2, 3, or 4)
ENDING_LEVEL = 4    # Level to end execution at (1, 2, 3, or 4)
ADD_DEPENDENCIES = True  # Set to False to run levels independently without upstream dependencies
BASINS = ['EP', 'NA', 'NI', 'SI', 'AU', 'SP', 'WP']
VERBOSE = False
DELETE_EXISTING_FOLDERS = True 
CLEAN_BASIN_FOLDERS_BEFORE_LEVEL4 = False  # Delete basin folders before running Level 4

DYNAMIC_NUM_DRAWS = True
if DYNAMIC_NUM_DRAWS:
    CLEAN_BASIN_FOLDERS_BEFORE_LEVEL4 = False 


DRY_RUN = False
# Validation configuration
INPUT_DATA_TYPE = "data"
INPUT_IO_TYPE = "raw"

# ============================================================================
# HELPER FUNCTIONS FOR OUTPUT CHECKING
# ============================================================================

def check_level3_output_finished(model, variant, scenario, time_period, data_source="cmip6"):
    """
    Check if Level 3 (global TC-risk) has finished successfully.
    Returns True if finished (has both env_wnd and thermo files), False otherwise.
    """
    output_path = TC_RISK_OUTPUT_PATH / data_source / model / variant / scenario / time_period
    if not output_path.exists():
        return False
    
    # Check for required Level 3 output files
    env_wnd_files = list(output_path.glob("env_wnd_*.nc"))
    thermo_files = list(output_path.glob("thermo_*.nc"))
    
    # Complete if both files exist
    return len(env_wnd_files) > 0 and len(thermo_files) > 0

def get_level4_basin_file_count(model, variant, scenario, time_period, basin, data_source="cmip6"):
    """
    Get the number of files in a Level 4 basin output folder.
    Returns the count, or -1 if folder doesn't exist.
    """
    output_path = TC_RISK_OUTPUT_PATH / data_source / model / variant / scenario / time_period / basin
    if not output_path.exists():
        return -1
    
    files = list(output_path.glob("*.nc"))
    return len(files)

def check_level4_output_complete(model, variant, scenario, time_period, basin, data_source="cmip6"):
    """
    Check if Level 4 (basin TC-risk) output is complete.
    Returns True if has NUM_DRAWS + 1 files, False otherwise.
    """
    file_count = get_level4_basin_file_count(model, variant, scenario, time_period, basin, data_source)
    return file_count >= NUM_DRAWS + 1

def clean_basin_folder(model, variant, scenario, time_period, basin, data_source="cmip6"):
    """
    Delete all files in a basin output folder before running Level 4.
    This ensures a clean run and prevents partial results from interfering.
    """
    output_path = TC_RISK_OUTPUT_PATH / data_source / model / variant / scenario / time_period / basin
    if output_path.exists():
        try:
            for file in output_path.glob("*.nc"):
                file.unlink()
            if VERBOSE:
                print(f"    Cleaned basin folder: {basin}")
        except Exception as e:
            print(f"    WARNING: Could not clean {output_path}: {e}")

# ============================================================================
# STAGE 1: LOAD TIME BINS (Model/Variant/Scenario-specific)
# ============================================================================

print("=" * 80)
print("STEP 1: Loading time bins from BayesPoisson changepoint detection")
print("=" * 80)

time_bins_df = pd.read_csv(TIME_BINS_DF_PATH)

# Filter to BayesPoisson method only
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
    # For levels 3+ only, build task list from time_bins without validation
    for (model_name, variant_name, scenario), time_periods in time_bins_lookup.items():
        for tp_info in time_periods:
            tasks_to_run.append({
                'model': model_name,
                'variant': variant_name,
                'scenario': scenario,
                'time_period': tp_info['time_period'],
                'start_year': tp_info['start_year'],
                'end_year': tp_info['end_year'],
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
if STARTING_LEVEL < 1 or ENDING_LEVEL > 4 or STARTING_LEVEL > ENDING_LEVEL:
    print(f"\nERROR: Invalid level configuration. STARTING_LEVEL={STARTING_LEVEL}, ENDING_LEVEL={ENDING_LEVEL}")
    print("Valid range: 1-4 and STARTING_LEVEL <= ENDING_LEVEL")
    sys.exit(1)

print(f"\nRunning Levels {STARTING_LEVEL} through {ENDING_LEVEL}")
print(f"Add dependencies between levels: {ADD_DEPENDENCIES}")

# ============================================================================
# STAGE 4: FILTER TASKS BASED ON STARTING LEVEL
# ============================================================================

if STARTING_LEVEL >= 3:
    print("\n" + "=" * 80)
    print("STEP 4: Filtering tasks based on current Level 3 and Level 4 status")
    print("=" * 80)
    
    # Identify which tasks have finished L3 and which haven't
    level3_finished = []
    level3_not_finished = []
    
    for task in tasks_to_run:
        if check_level3_output_finished(task['model'], task['variant'], task['scenario'], task['time_period'], DATA_SOURCE):
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
# STAGE 5: DETERMINE WHICH BASINS NEED TO RUN FOR LEVEL 4
# ============================================================================

if STARTING_LEVEL >= 3 and ENDING_LEVEL >= 4:
    print("\n" + "=" * 80)
    print("STEP 5: Checking Level 4 basin completeness")
    print("=" * 80)
    
    level4_basin_tasks = []  # List of (task, basin) pairs to run
    
    for task in level4_tasks_to_check:
        model, variant, scenario, time_period = task['model'], task['variant'], task['scenario'], task['time_period']
        incomplete_basins = []
        basin_file_counts = {}
        
        for basin in BASINS:
            file_count = get_level4_basin_file_count(model, variant, scenario, time_period, basin, DATA_SOURCE)
            basin_file_counts[basin] = file_count
            
            # Basin is incomplete if it has fewer than NUM_DRAWS + 1 files
            if file_count < NUM_DRAWS + 1:
                incomplete_basins.append(basin)
        
        if incomplete_basins:
            for basin in incomplete_basins:
                level4_basin_tasks.append({
                    'model': model,
                    'variant': variant,
                    'scenario': scenario,
                    'time_period': time_period,
                    'basin': basin,
                    'file_count': basin_file_counts[basin],
                    'level3_finished': task in level3_finished
                })
    
    print(f"\nLevel 4 basin tasks to create: {len(level4_basin_tasks)}")
    
    if VERBOSE and level4_basin_tasks:
        print(f"\nFirst 20 Level 4 basin tasks:")
        for i, task in enumerate(level4_basin_tasks[:20]):
            l3_status = "L3✓" if task['level3_finished'] else "L3✗"
            print(f"  {task['model']}/{task['variant']}/{task['scenario']}/{task['time_period']}/{task['basin']} ({task['file_count']} files, {l3_status})")
        if len(level4_basin_tasks) > 20:
            print(f"  ... and {len(level4_basin_tasks) - 20} more")
    
    # Clean basin folders before running if enabled
    if CLEAN_BASIN_FOLDERS_BEFORE_LEVEL4 and level4_basin_tasks:
        print(f"\nCleaning basin folders before Level 4 execution...")
        cleaned_count = 0
        for task in level4_basin_tasks:
            clean_basin_folder(task['model'], task['variant'], task['scenario'], task['time_period'], task['basin'], DATA_SOURCE)
            cleaned_count += 1
        print(f"Cleaned {cleaned_count} basin folders")

# ============================================================================
# JOBMON SETUP & TEMPLATES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Setting up Jobmon workflow")
print("=" * 80)

user = getpass.getuser()
project = "proj_rapidresponse"
queue = 'long.q'
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

# --- Template 1: Folder Creation (Level 1) ---
delete_flag = "--delete_destination_folder " if DELETE_EXISTING_FOLDERS else ""

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
        "{delete_flag}"
    ).format(script_root=SCRIPT_ROOT, delete_flag=delete_flag),
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
        "memory": "40G", "cores": rfc.tc_risk_n_procs + 1, "runtime": "1h",
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

# --- Template 4: Basin TC-Risk Run ---
basin_run_template = tool.get_task_template(
    template_name="run_basin_tc_risk",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "40G", "cores": rfc.tc_risk_n_procs + 1, "runtime": "2h",
        "queue": queue, "project": project,
    },
    command_template=(
        "python {script_root}/05_run_basin_tc_risk.py "
        "--data_source {{data_source}} "
        "--model {{model}} "
        "--variant {{variant}} "
        "--scenario {{scenario}} "
        "--time_period {{time_period}} "
        "--basin {{basin}} "
        "--num_draws {{num_draws}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["data_source", "model", "variant", "scenario", "time_period", "basin", "num_draws"],
)

# ============================================================================
# TASK CREATION AND NESTED DEPENDENCIES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Creating Jobmon tasks with dependencies")
print("=" * 80)

all_tasks = []
dependencies = []

# ========== LEVELS 1 & 2 ==========
if STARTING_LEVEL <= 2:
    for task_info in tasks_to_run:
        model_name = task_info['model']
        variant_name = task_info['variant']
        scenario_name = task_info['scenario']
        time_period_str = task_info['time_period']

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
        process_tasks_for_this_combo = []
        
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

                # Resource allocation based on frequency
                if 'day' in frequency:
                    mem = "120G"
                    runtime = "1h"
                    cores = 8
                else:
                    mem = "16G"
                    runtime = "10min"
                    cores = 8

                process_task = process_variable_template.create_task(
                    compute_resources={
                        "memory": mem,
                        "cores": cores,
                        "runtime": runtime,
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
                process_tasks_for_this_combo.append(process_task)

                # Dependency: process_task depends on folder_task
                if ADD_DEPENDENCIES and folder_task is not None and STARTING_LEVEL <= 1:
                    dependencies.append((process_task, folder_task))

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

        global_task = global_run_template.create_task(
            data_source=DATA_SOURCE,
            model=model_name,
            variant=variant_name,
            scenario=scenario_name,
            time_period=time_period_str,
        )
        all_tasks.append(global_task)
        level3_tasks_dict[(model_name, variant_name, scenario_name, time_period_str)] = global_task

        # Dependency: global_task depends on process_tasks (if Level 2 ran)
        if ADD_DEPENDENCIES and STARTING_LEVEL <= 2:
            # Find the process tasks for this combo
            for task in all_tasks:
                if hasattr(task, 'task_args'):
                    task_args = task.task_args
                    if (task_args.get('model') == model_name and 
                        task_args.get('variant') == variant_name and
                        task_args.get('scenario') == scenario_name and
                        task_args.get('time_period') == time_period_str and
                        'variable' in task_args):
                        dependencies.append((global_task, task))

# ========== LEVEL 4 ==========
if STARTING_LEVEL <= 4 and ENDING_LEVEL >= 4:
    for basin_task_info in level4_basin_tasks:
        model_name = basin_task_info['model']
        variant_name = basin_task_info['variant']
        scenario_name = basin_task_info['scenario']
        time_period_str = basin_task_info['time_period']
        basin = basin_task_info['basin']
        
        # Determine num_draws for this task
        if DYNAMIC_NUM_DRAWS:
            file_count = basin_task_info['file_count']
            num_draws_for_task = NUM_DRAWS if file_count < 0 else max(0, NUM_DRAWS - file_count)
        else:
            num_draws_for_task = NUM_DRAWS

        # Skip if no draws needed
        if num_draws_for_task == 0:
            if VERBOSE:
                print(f"  Skipping {model_name}/{variant_name}/{scenario_name}/{time_period_str}/{basin}: already has {NUM_DRAWS} draws")
            continue

        basin_task = basin_run_template.create_task(
            data_source=DATA_SOURCE,
            model=model_name,
            variant=variant_name,
            scenario=scenario_name,
            time_period=time_period_str,
            basin=basin,
            num_draws=num_draws_for_task,
        )
        all_tasks.append(basin_task)

        # Dependency: basin_task depends on global_task (if Level 3 is running/will run)
        if ADD_DEPENDENCIES and STARTING_LEVEL <= 3:
            level3_key = (model_name, variant_name, scenario_name, time_period_str)
            if level3_key in level3_tasks_dict:
                dependencies.append((basin_task, level3_tasks_dict[level3_key]))

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

    if level4_basin_tasks:
        print(f"\nTotal Level 4 basin tasks: {len(level4_basin_tasks)}\n")
        print("Task details:")
        for i, basin_task_info in enumerate(level4_basin_tasks):
            num_draws_for_task = max(0, NUM_DRAWS - basin_task_info['file_count']) if DYNAMIC_NUM_DRAWS else NUM_DRAWS
            print(f"{i+1}. {basin_task_info['model']}/{basin_task_info['variant']}/{basin_task_info['scenario']}/{basin_task_info['time_period']}/{basin_task_info['basin']}")
            print(f"   Current files: {basin_task_info['file_count']}, Num draws to run: {num_draws_for_task}")
    else:
        print("\nNo Level 4 basin tasks to run.")

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
        status = workflow.run()
        print(f"Workflow {workflow.workflow_id} completed with status {status}.")
    except Exception as e:
        print(f"Workflow submission failed: {e}")
        sys.exit(1)