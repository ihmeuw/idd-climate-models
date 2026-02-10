"""
Refactored Orchestrator for TC-risk pipeline.

This script creates a single Jobmon workflow with up to 4 levels:
  Level 1: Create folders (tc_risk input/output)
  Level 2: Process raw files directly to time-period files
  Level 3: Run global TC-risk
  Level 4: Run basin TC-risk (with dynamic draw counting)

Key features:
- Simplified configuration with fewer flags
- Dynamic draw counting always enabled (automatic resume)
- Optional cleaning steps between levels
- Automatic dependency management for multi-level runs
"""

import sys
import os
import getpass
import uuid
from pathlib import Path
import pandas as pd
from jobmon.client.tool import Tool

import idd_climate_models.constants as rfc
from idd_climate_models.dictionary_utils import parse_results
from idd_climate_models.io_compare_utils import compare_model_validation
from idd_climate_models.resource_functions import (
    get_level2_resources, 
    get_level3_resources, 
    get_level4_resources
)
import orchestrator_utils as utils

# ============================================================================
# CONFIGURATION
# ============================================================================

# --- Pipeline Control ---
STARTING_LEVEL = 3  # Level to start execution from (1-4)
ENDING_LEVEL = 4    # Level to end execution at (1-4)

# --- Cleaning Control ---
# Set to None to skip cleaning, or specify range of levels to clean
CLEAN_STARTING_LEVEL = None  # Start cleaning from this level (1-4 or None)
CLEAN_ENDING_LEVEL = None    # End cleaning at this level (1-4 or None)

# --- Data Source ---
DATA_SOURCE = "cmip6"

# --- Testing ---
TEST_RUN = 0  # Set to 0 for full run, >0 to limit number of model/variant/scenario/time_period combinations
DRY_RUN = False  # Set to True to print tasks without submitting

# --- Display ---
VERBOSE = False

# --- Paths ---
repo_name = rfc.repo_name
package_name = rfc.package_name
RAW_DATA_PATH = rfc.RAW_DATA_PATH
TC_RISK_INPUT_PATH = rfc.TC_RISK_INPUT_PATH
TC_RISK_OUTPUT_PATH = rfc.TC_RISK_OUTPUT_PATH
CLIMADA_INPUT_PATH = rfc.CLIMADA_INPUT_PATH
TIME_BINS_DF_PATH = rfc.TIME_BINS_DF_PATH

# Script locations
WORKER_SCRIPT_ROOT = rfc.REPO_ROOT / repo_name / "src" / package_name / "01_process_raw_through_climada_input"

# --- Constants ---
NUM_DRAWS = rfc.NUM_DRAWS
DRAWS_PER_BATCH = 10
BASINS = ['EP', 'NA', 'NI', 'SI', 'AU', 'SP', 'WP']

# ============================================================================
# VALIDATION
# ============================================================================

if STARTING_LEVEL < 1 or ENDING_LEVEL > 4 or STARTING_LEVEL > ENDING_LEVEL:
    print(f"ERROR: Invalid level configuration. STARTING_LEVEL={STARTING_LEVEL}, ENDING_LEVEL={ENDING_LEVEL}")
    print("Valid range: 1-4 and STARTING_LEVEL <= ENDING_LEVEL")
    sys.exit(1)

if CLEAN_STARTING_LEVEL is not None and CLEAN_ENDING_LEVEL is not None:
    if CLEAN_STARTING_LEVEL < 1 or CLEAN_ENDING_LEVEL > 4 or CLEAN_STARTING_LEVEL > CLEAN_ENDING_LEVEL:
        print(f"ERROR: Invalid cleaning configuration. CLEAN_STARTING_LEVEL={CLEAN_STARTING_LEVEL}, CLEAN_ENDING_LEVEL={CLEAN_ENDING_LEVEL}")
        sys.exit(1)

# ============================================================================
# LOAD TIME BINS
# ============================================================================

print("=" * 80)
print("STEP 1: Loading time bins")
print("=" * 80)

time_bins_df, time_bins_lookup = utils.load_time_bins(TIME_BINS_DF_PATH)

print(f"Loaded {len(time_bins_df)} time bin rows")
print(f"Found {len(time_bins_lookup)} unique (model, variant, scenario) combinations")

# ============================================================================
# VALIDATE RAW DATA (if needed for Level 1 or 2)
# ============================================================================

variable_detail_map = {}

if STARTING_LEVEL <= 2:
    print("\n" + "=" * 80)
    print("STEP 2: Validating raw data")
    print("=" * 80)

    validation_info = compare_model_validation(
        input_data_type="data",
        input_io_type="raw",
        output_data_type="tc_risk",
        output_io_type="input",
        data_source=DATA_SOURCE,
        verbose=VERBOSE,
        rerun_all=True,
    )

    input_complete_model_variants = validation_info["input_complete_model_variants"]
    print(f"Found {len(input_complete_model_variants)} complete model/variant combinations")

    # Build variable detail map
    full_path_list = parse_results(validation_info["input_validation_dict"], 'all')
    for item in full_path_list:
        key = (item['model'], item['variant'], item['scenario'], item['variable'])
        raw_dir = RAW_DATA_PATH / DATA_SOURCE / item['model'] / item['variant'] / item['scenario'] / item['variable'] / item['grid'] / item['frequency']
        raw_files = sorted(raw_dir.glob("*.nc")) if raw_dir.exists() else []
        variable_detail_map[key] = {
            'grid': item['grid'],
            'frequency': item['frequency'],
            'raw_files': raw_files
        }
else:
    input_complete_model_variants = []

# ============================================================================
# BUILD TASK LIST
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Building task list")
print("=" * 80)

tasks_to_run = []

if STARTING_LEVEL <= 2:
    # Build from validated raw data
    for model_variant_tuple in input_complete_model_variants:
        model_name, variant_name = model_variant_tuple
        for scenario in rfc.SCENARIOS:
            key = (model_name, variant_name, scenario)
            if key not in time_bins_lookup:
                if VERBOSE:
                    print(f"  Skipping {model_name}/{variant_name}/{scenario}: no time bins")
                continue
            for tp_info in time_bins_lookup[key]:
                tasks_to_run.append({
                    'model': model_name,
                    'variant': variant_name,
                    'scenario': scenario,
                    'time_period': tp_info['time_period'],
                    'start_year': tp_info['start_year'],
                    'end_year': tp_info['end_year'],
                })
else:
    # Build from time bins only
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

print(f"Total tasks: {len(tasks_to_run)} (model/variant/scenario/time_period combinations)")

if not tasks_to_run:
    print("No tasks to run. Exiting.")
    sys.exit(0)

# Apply test run limit
if TEST_RUN > 0:
    print(f"TEST MODE: Limiting to {TEST_RUN} tasks")
    tasks_to_run = tasks_to_run[:TEST_RUN]

# ============================================================================
# CHECK LEVEL 3 STATUS (if starting at Level 3+)
# ============================================================================

level3_finished = []
level3_not_finished = []

if STARTING_LEVEL >= 3:
    print("\n" + "=" * 80)
    print("STEP 4: Checking Level 3 status")
    print("=" * 80)
    
    for task in tasks_to_run:
        if utils.check_level3_output_finished(
            task['model'], task['variant'], task['scenario'], task['time_period'],
            TC_RISK_OUTPUT_PATH / DATA_SOURCE
        ):
            level3_finished.append(task)
        else:
            level3_not_finished.append(task)
    
    print(f"Level 3 finished: {len(level3_finished)}")
    print(f"Level 3 not finished: {len(level3_not_finished)}")
    
    if STARTING_LEVEL == 3:
        level3_tasks_to_create = level3_not_finished
        level4_tasks_to_check = tasks_to_run  # Check all
    else:  # STARTING_LEVEL == 4
        level3_tasks_to_create = []
        level4_tasks_to_check = level3_finished  # Only check finished L3
else:
    level3_tasks_to_create = tasks_to_run if ENDING_LEVEL >= 3 else []
    level4_tasks_to_check = tasks_to_run if ENDING_LEVEL >= 4 else []

# ============================================================================
# DETERMINE LEVEL 4 BASIN TASKS (with dynamic draw counting)
# ============================================================================

level4_basin_tasks = []

if ENDING_LEVEL >= 4:
    print("\n" + "=" * 80)
    print("STEP 5: Checking Level 4 basin status (dynamic draw counting)")
    print("=" * 80)
    
    for task in level4_tasks_to_check:
        model, variant, scenario, time_period = task['model'], task['variant'], task['scenario'], task['time_period']
        
        for basin in BASINS:
            # Get existing draws
            existing_draws = utils.get_existing_draw_numbers(
                model, variant, scenario, time_period, basin,
                TC_RISK_OUTPUT_PATH / DATA_SOURCE
            )
            
            # Determine missing draw batches
            missing_batches = utils.determine_missing_draw_batches(
                existing_draws, NUM_DRAWS, DRAWS_PER_BATCH
            )
            
            if missing_batches:
                for batch_start, batch_end in missing_batches:
                    level4_basin_tasks.append({
                        'model': model,
                        'variant': variant,
                        'scenario': scenario,
                        'time_period': time_period,
                        'basin': basin,
                        'draw_start': batch_start,
                        'draw_end': batch_end,
                        'existing_draws': len(existing_draws),
                        'level3_finished': task in level3_finished if STARTING_LEVEL >= 3 else True
                    })
    
    print(f"Level 4 basin tasks to create: {len(level4_basin_tasks)}")
    if VERBOSE and level4_basin_tasks:
        print(f"\nFirst 10 tasks:")
        for task in level4_basin_tasks[:10]:
            print(f"  {task['model']}/{task['variant']}/{task['scenario']}/{task['time_period']}/{task['basin']} "
                  f"draws {task['draw_start']}-{task['draw_end']} ({task['existing_draws']} existing)")

# ============================================================================
# JOBMON SETUP
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Setting up Jobmon workflow")
print("=" * 80)

user = getpass.getuser()
project = "proj_rapidresponse"
queue = 'long.q'
wf_uuid = uuid.uuid4()
tool_name = f"{package_name}_tc_risk_pipeline_refactored"
tool = Tool(name=tool_name)

workflow = tool.create_workflow(
    name=f"{tool_name}_{wf_uuid}",
    max_concurrently_running=5000,
)

workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={
        "memory": "2G", "cores": 1, "runtime": "10m",
        "queue": queue, "project": project,
    }
)

# ============================================================================
# TASK TEMPLATES
# ============================================================================

# --- Cleaning Templates ---

# Level 1 Cleaning: Rename paths (fast)
level1_rename_template = tool.get_task_template(
    template_name="clean_level1_rename",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "1G", "cores": 1, "runtime": "5m",
        "queue": queue, "project": project,
    },
    command_template=(
        "python -c \""
        "from pathlib import Path; "
        "import sys; "
        "tc_path = Path('{tc_risk_input_path}') / '{data_source}'; "
        "climada_path = Path('{climada_input_path}') / '{data_source}'; "
        "for p in [tc_path, climada_path]: "
        "    if p.exists(): "
        "        p.rename(p.parent / (p.name + '_DELETE')); "
        "        print(f'Renamed {{p}} for deletion'); "
        "print('Level 1 rename complete')"
        "\""
    ),
    node_args=["tc_risk_input_path", "climada_input_path", "data_source"],
)

# Level 1 Cleaning: Delete renamed paths (async, slow)
level1_delete_template = tool.get_task_template(
    template_name="clean_level1_delete",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "1G", "cores": 1, "runtime": "2h",
        "queue": queue, "project": project,
    },
    command_template=(
        "python -c \""
        "from pathlib import Path; "
        "import subprocess; "
        "tc_path = Path('{tc_risk_input_path}') / '{data_source}_DELETE'; "
        "climada_path = Path('{climada_input_path}') / '{data_source}_DELETE'; "
        "for p in [tc_path, climada_path]: "
        "    if p.exists(): "
        "        subprocess.run(['rm', '-rf', str(p)], check=False); "
        "        print(f'Deleted {{p}}'); "
        "print('Level 1 delete complete')"
        "\""
    ),
    node_args=["tc_risk_input_path", "climada_input_path", "data_source"],
)

# Level 2 Cleaning: Delete processed variables
level2_clean_template = tool.get_task_template(
    template_name="clean_level2",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "2G", "cores": 1, "runtime": "30m",
        "queue": queue, "project": project,
    },
    command_template=(
        "python -c \""
        "from pathlib import Path; "
        "import subprocess; "
        "path = Path('{tc_risk_input_path}') / '{data_source}' / '{model}' / '{variant}' / '{scenario}' / '{time_period}'; "
        "if path.exists(): "
        "    subprocess.run(['rm', '-rf', str(path)], check=False); "
        "    print(f'Cleaned {{path}}'); "
        "else: "
        "    print(f'Path does not exist: {{path}}'); "
        "print('Level 2 clean complete')"
        "\""
    ),
    node_args=["tc_risk_input_path", "data_source", "model", "variant", "scenario", "time_period"],
)

# Level 3 Cleaning: Delete global TC-risk outputs
level3_clean_template = tool.get_task_template(
    template_name="clean_level3",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "2G", "cores": 1, "runtime": "30m",
        "queue": queue, "project": project,
    },
    command_template=(
        "python -c \""
        "from pathlib import Path; "
        "import subprocess; "
        "path = Path('{tc_risk_output_path}') / '{data_source}' / '{model}' / '{variant}' / '{scenario}' / '{time_period}'; "
        "if path.exists(): "
        "    env_wnd = list(path.glob('env_wnd_*.nc')); "
        "    thermo = list(path.glob('thermo_*.nc')); "
        "    for f in env_wnd + thermo: f.unlink(); "
        "    print(f'Cleaned {{len(env_wnd) + len(thermo)}} Level 3 files from {{path}}'); "
        "else: "
        "    print(f'Path does not exist: {{path}}'); "
        "print('Level 3 clean complete')"
        "\""
    ),
    node_args=["tc_risk_output_path", "data_source", "model", "variant", "scenario", "time_period"],
)

# Level 4 Cleaning: Delete basin outputs
level4_clean_template = tool.get_task_template(
    template_name="clean_level4",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "2G", "cores": 1, "runtime": "1h",
        "queue": queue, "project": project,
    },
    command_template=(
        "python -c \""
        "from pathlib import Path; "
        "import subprocess; "
        "tc_path = Path('{tc_risk_output_path}') / '{data_source}' / '{model}' / '{variant}' / '{scenario}' / '{time_period}' / '{basin}'; "
        "climada_path = Path('{climada_input_path}') / '{data_source}' / '{model}' / '{variant}' / '{scenario}' / '{time_period}' / '{basin}'; "
        "for p in [tc_path, climada_path]: "
        "    if p.exists(): "
        "        subprocess.run(['timeout', '60', 'rm', '-rf', str(p)], check=False); "
        "        p.mkdir(parents=True, exist_ok=True); "
        "        print(f'Cleaned {{p}}'); "
        "print('Level 4 clean complete')"
        "\""
    ),
    node_args=["tc_risk_output_path", "climada_input_path", "data_source", "model", "variant", "scenario", "time_period", "basin"],
)

# --- Level 1: Folder Creation ---
folder_template = tool.get_task_template(
    template_name="create_folders",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "1G", "cores": 1, "runtime": "5m",
        "queue": queue, "project": project,
    },
    command_template=(
        f"python {WORKER_SCRIPT_ROOT}/01_create_folders.py "
        "--data_source {data_source} "
        "--model {model} "
        "--variant {variant} "
        "--scenario {scenario} "
        "--time_period {time_period}"
    ),
    node_args=["data_source", "model", "variant", "scenario", "time_period"],
)

# --- Level 2: Process Variable ---
process_variable_template = tool.get_task_template(
    template_name="process_variable",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "16G", "cores": 4, "runtime": "1h",
        "queue": queue, "project": project,
    },
    command_template=(
        f"python {WORKER_SCRIPT_ROOT}/02_process_variable.py "
        "--data_source {data_source} "
        "--model {model} "
        "--variant {variant} "
        "--scenario {scenario} "
        "--time_period {time_period} "
        "--variable {variable} "
        "--grid {grid} "
        "--frequency {frequency}"
    ),
    node_args=["data_source", "model", "variant", "scenario", "time_period", "variable", "grid", "frequency"],
)

# --- Level 3: Global TC-Risk ---
global_run_template = tool.get_task_template(
    template_name="run_global_tc_risk",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "40G", "cores": rfc.tc_risk_n_procs + 1, "runtime": "1h",
        "queue": queue, "project": project,
    },
    command_template=(
        f"python {WORKER_SCRIPT_ROOT}/04_run_global_tc_risk.py "
        "--data_source {data_source} "
        "--model {model} "
        "--variant {variant} "
        "--scenario {scenario} "
        "--time_period {time_period}"
    ),
    node_args=["data_source", "model", "variant", "scenario", "time_period"],
)

# --- Level 4: Basin TC-Risk ---
basin_run_template = tool.get_task_template(
    template_name="run_basin_tc_risk",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "25G", 
        "cores": rfc.tc_risk_n_procs + 1, 
        "runtime": "4h",
        "queue": queue, 
        "project": project,
    },
    command_template=(
        f"python {WORKER_SCRIPT_ROOT}/05_run_basin_tc_risk.py "
        "--data_source {data_source} "
        "--model {model} "
        "--variant {variant} "
        "--scenario {scenario} "
        "--time_period {time_period} "
        "--basin {basin} "
        "--draw_start {draw_start} "
        "--draw_end {draw_end}"
    ),
    node_args=["data_source", "model", "variant", "scenario", "time_period", "basin", "draw_start", "draw_end"],
)

# ============================================================================
# CREATE TASKS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Creating Jobmon tasks")
print("=" * 80)

all_tasks = []
dependencies = []

# Track tasks for dependency chaining
level1_tasks_dict = {}  # (model, variant, scenario, time_period) -> task
level2_tasks_dict = {}  # (model, variant, scenario, time_period) -> [tasks]
level3_tasks_dict = {}  # (model, variant, scenario, time_period) -> task

# --- LEVEL 1: FOLDER CREATION ---
if STARTING_LEVEL <= 1 and ENDING_LEVEL >= 1:
    print(f"Creating Level 1 tasks: {len(tasks_to_run)} folder creation tasks")
    
    for task_info in tasks_to_run:
        folder_task = folder_template.create_task(
            data_source=DATA_SOURCE,
            model=task_info['model'],
            variant=task_info['variant'],
            scenario=task_info['scenario'],
            time_period=task_info['time_period'],
        )
        all_tasks.append(folder_task)
        
        key = (task_info['model'], task_info['variant'], task_info['scenario'], task_info['time_period'])
        level1_tasks_dict[key] = folder_task

# --- LEVEL 2: PROCESS VARIABLES ---
if STARTING_LEVEL <= 2 and ENDING_LEVEL >= 2:
    print(f"Creating Level 2 tasks...")
    
    level2_count = 0
    for task_info in tasks_to_run:
        key = (task_info['model'], task_info['variant'], task_info['scenario'], task_info['time_period'])
        process_tasks = []
        
        for variable in rfc.VARIABLES[DATA_SOURCE]:
            detail_key = (task_info['model'], task_info['variant'], task_info['scenario'], variable)
            details = variable_detail_map.get(detail_key)
            
            if details is None:
                if VERBOSE:
                    print(f"  Warning: No details for {detail_key}")
                continue
            
            resources = get_level2_resources(task_info['time_period'], variable, details['frequency'])
            
            process_task = process_variable_template.create_task(
                compute_resources={
                    "memory": resources["memory"],
                    "cores": resources["cores"],
                    "runtime": resources["runtime"],
                    "queue": queue,
                    "project": project,
                },
                data_source=DATA_SOURCE,
                model=task_info['model'],
                variant=task_info['variant'],
                scenario=task_info['scenario'],
                time_period=task_info['time_period'],
                variable=variable,
                grid=details['grid'],
                frequency=details['frequency'],
            )
            all_tasks.append(process_task)
            process_tasks.append(process_task)
            level2_count += 1
            
            # Dependency: process_task depends on folder_task (if Level 1 ran)
            if STARTING_LEVEL <= 1 and key in level1_tasks_dict:
                dependencies.append((process_task, level1_tasks_dict[key]))
        
        if process_tasks:
            level2_tasks_dict[key] = process_tasks
    
    print(f"Created {level2_count} Level 2 tasks")

# --- LEVEL 3: GLOBAL TC-RISK ---
if STARTING_LEVEL <= 3 and ENDING_LEVEL >= 3:
    print(f"Creating Level 3 tasks: {len(level3_tasks_to_create)} global TC-risk tasks")
    
    for task_info in level3_tasks_to_create:
        key = (task_info['model'], task_info['variant'], task_info['scenario'], task_info['time_period'])
        
        resources = get_level3_resources(task_info['time_period'])
        
        global_task = global_run_template.create_task(
            compute_resources={
                "memory": resources["memory"],
                "cores": resources["cores"],
                "runtime": resources["runtime"],
                "queue": queue,
                "project": project,
            },
            data_source=DATA_SOURCE,
            model=task_info['model'],
            variant=task_info['variant'],
            scenario=task_info['scenario'],
            time_period=task_info['time_period'],
        )
        all_tasks.append(global_task)
        level3_tasks_dict[key] = global_task
        
        # Dependency: global_task depends on all process_tasks (if Level 2 ran)
        if STARTING_LEVEL <= 2 and key in level2_tasks_dict:
            for process_task in level2_tasks_dict[key]:
                dependencies.append((global_task, process_task))

# --- LEVEL 4: BASIN TC-RISK ---
if STARTING_LEVEL <= 4 and ENDING_LEVEL >= 4:
    print(f"Creating Level 4 tasks: {len(level4_basin_tasks)} basin tasks (with dynamic draws)")
    
    for basin_task_info in level4_basin_tasks:
        key = (basin_task_info['model'], basin_task_info['variant'], 
               basin_task_info['scenario'], basin_task_info['time_period'])
        
        resources = get_level4_resources(
            basin_task_info['model'], basin_task_info['variant'], 
            basin_task_info['scenario'], basin_task_info['time_period'],
            basin_task_info['basin'], DRAWS_PER_BATCH,
            verbose=VERBOSE
        )
        
        basin_task = basin_run_template.create_task(
            compute_resources={
                "memory": resources["memory"],
                "cores": resources["cores"],
                "runtime": resources["runtime"],
                "queue": queue,
                "project": project,
            },
            data_source=DATA_SOURCE,
            model=basin_task_info['model'],
            variant=basin_task_info['variant'],
            scenario=basin_task_info['scenario'],
            time_period=basin_task_info['time_period'],
            basin=basin_task_info['basin'],
            draw_start=basin_task_info['draw_start'],
            draw_end=basin_task_info['draw_end'],
        )
        all_tasks.append(basin_task)
        
        # Dependency: basin_task depends on global_task (if Level 3 is running)
        if STARTING_LEVEL <= 3 and key in level3_tasks_dict:
            dependencies.append((basin_task, level3_tasks_dict[key]))

print(f"\nTotal tasks created: {len(all_tasks)}")
print(f"Total dependencies: {len(dependencies)}")

# ============================================================================
# ADD TASKS AND DEPENDENCIES
# ============================================================================

workflow.add_tasks(all_tasks)

if dependencies:
    print(f"Adding {len(dependencies)} dependencies...")
    for child_task, parent_task in dependencies:
        child_task.add_upstream(parent_task)
    print("Dependencies added.")

# ============================================================================
# DRY RUN OR SUBMIT
# ============================================================================

if DRY_RUN:
    print("\n" + "=" * 80)
    print("DRY RUN: Tasks that would be executed")
    print("=" * 80)
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Levels: {STARTING_LEVEL} to {ENDING_LEVEL}")
    print("\nTo actually run, set DRY_RUN = False")
    sys.exit(0)
else:
    print("\n" + "=" * 80)
    print("STEP 8: Binding and running workflow")
    print("=" * 80)
    
    try:
        workflow.bind()
        print(f"Workflow bound with ID {workflow.workflow_id}")
        print(f"View at: https://jobmon-gui.ihme.washington.edu/#/workflow/{workflow.workflow_id}")
    except Exception as e:
        print(f"Workflow binding failed: {e}")
        sys.exit(1)
    
    try:
        status = workflow.run()
        print(f"Workflow {workflow.workflow_id} completed with status {status}")
    except Exception as e:
        print(f"Workflow execution failed: {e}")
        sys.exit(1)
