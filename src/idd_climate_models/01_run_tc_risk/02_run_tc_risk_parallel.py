import importlib
import sys
import getpass
import uuid
import os
import copy
from jobmon.client.tool import Tool 
from pathlib import Path

# NOTE: Imports required for the execution logic
import idd_climate_models.constants as rfc
from idd_climate_models.dictionary_utils import parse_results
from idd_climate_models.validation_functions import create_validation_dict, validate_all_models_in_source 
# Assuming a constant for strict_grid_check_flag is defined elsewhere or set to False
strict_grid_check_flag = False 
# ----------------------------------------------------------------------------

# --- CONSTANT DEFINITIONS ---
repo_name = rfc.repo_name
package_name = rfc.package_name
TC_RISK_INPUT_PATH = rfc.TC_RISK_INPUT_PATH
TC_RISK_OUTPUT_PATH = rfc.TC_RISK_OUTPUT_PATH
SCRIPT_ROOT = rfc.REPO_ROOT / repo_name / "src" / package_name / "01_run_tc_risk"

# Configuration
DATA_SOURCE = "cmip6"
NUM_DRAWS = rfc.NUM_DRAWS
TEST_RUN = 0  # Set to 0 for full run, >0 for test runs
BASINS = ['EP', 'NA', 'NI', 'SI', 'SP', 'WP'] 

INPUT_DATA_TYPE = "tc_risk"
INPUT_IO_TYPE = "input"
OUTPUT_DATA_TYPE = "tc_risk"
OUTPUT_IO_TYPE = "output"

# ============================================================================
# STAGE 1: DATA SETUP & VALIDATION (Identify Complete Inputs)
# ============================================================================

# 1. Validate TC-Risk input folder structure
input_validation_dict = create_validation_dict(
    INPUT_DATA_TYPE, 
    INPUT_IO_TYPE, 
    DATA_SOURCE,
    strict_grid_check=False
)
input_validation_dict = validate_all_models_in_source(
    validation_dict=input_validation_dict,
    verbose=False,
    strict_grid_check=strict_grid_check_flag 
)

# 2. Get the list of all *complete* time_period paths to drive task creation
validation_results_for_tasks = parse_results(
    input_validation_dict, 
    detail='time_period'
)

# ============================================================================
# JOBMON SETUP & TEMPLATES
# ============================================================================

user = getpass.getuser()
project = "proj_rapidresponse"
queue = 'all.q'
wf_uuid = uuid.uuid4()
tool_name = f"{package_name}_tc_risk_execution_tool"
tool = Tool(name=tool_name)

workflow = tool.create_workflow(
    name=f"{tool_name}_workflow_{wf_uuid}",
    max_concurrently_running=5000,
)

# Set Default Compute Resources 
workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={
        "memory": "1G", "cores": 1, "runtime": "1m", 
        "queue": queue, "project": project,
    }
)

# --- Template 1: Folder Creation (2a) ---
folder_template = tool.get_task_template(
    template_name="create_output_folders",
    default_cluster_name="slurm", 
    command_template=(
        "python {script_root}/2a_create_tc_risk_output_folders.py "
        "--data_source {{data_source}} " 
        "--model {{model}} "
        "--variant {{variant}} " 
        "--scenario {{scenario}} "
        "--time_period {{time_period}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["data_source", "model", "variant", "scenario", "time_period"],
)

# --- Template 2: Global Run (2b) ---
global_run_template = tool.get_task_template(
    template_name="run_global_tc_risk",
    default_cluster_name="slurm", 
    default_compute_resources={ 
        "memory": "40G", "cores": rfc.tc_risk_n_procs + 1, "runtime": "1h", # core: 17, max runtime: 20m in test, max mem: 6.1G
        "queue": 'long.q', "project": project,
    },
    command_template=(
        "python {script_root}/2b_run_global_tc_risk.py "
        "--data_source {{data_source}} "
        "--model {{model}} "
        "--variant {{variant}} "
        "--scenario {{scenario}} "
        "--time_period {{time_period}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["data_source", "model", "variant", "scenario", "time_period"],
)

# --- Template 3: Basin Runs (2c) ---
basin_run_template = tool.get_task_template(
    template_name="run_basin_tc_risk",
    default_cluster_name="slurm",
    default_compute_resources={ 
        "memory": "40G", "cores": rfc.tc_risk_n_procs + 1, "runtime": "2h", # core: 17, max runtime: 58m in test, max mem: 7.4G
        "queue": queue, "project": project,
    },
    command_template=(
        "python {script_root}/2c_run_basin_tc_risk.py "
        "--data_source {{data_source}} "
        "--model {{model}} "
        "--variant {{variant}} "
        "--scenario {{scenario}} "
        "--time_period {{time_period}} "
        "--basin {{basin}} "
        f"--num_draws {NUM_DRAWS} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["data_source", "model", "variant", "scenario", "time_period", "basin"],
)

# ============================================================================
# TASK CREATION AND NESTED DEPENDENCIES (Using child_task.add_upstream)
# ============================================================================

all_tasks = []
dependencies = []

print("\n" + "=" * 80)
if TEST_RUN > 0:
    print(f"STEP 2: Creating Jobmon tasks for {TEST_RUN} model/time-bin combinations (Test Run).")
    items_to_process = validation_results_for_tasks[:TEST_RUN]
else:
    print(f"STEP 2: Creating Jobmon tasks for all valid model/time-bin combinations.")
    items_to_process = validation_results_for_tasks
print("=" * 80)

for item in items_to_process:
    # Extract the unique components of the path
    model_name = item['model']
    variant_name = item['variant']
    scenario_name = item['scenario']
    time_period_str = item['time_period'] 

    # --- LEVEL 1: FOLDER CREATION (Parent Task) ---
    folder_task = folder_template.create_task(
        data_source=DATA_SOURCE,
        model=model_name,
        variant=variant_name,
        scenario=scenario_name,
        time_period=time_period_str,
    )
    all_tasks.append(folder_task)


    # --- LEVEL 2: GLOBAL RUN (Child of Folder Task) ---
    global_task = global_run_template.create_task(
        data_source=DATA_SOURCE,
        model=model_name,
        variant=variant_name,
        scenario=scenario_name,
        time_period=time_period_str,
    )
    all_tasks.append(global_task)
    dependencies.append((global_task, folder_task)) 


    # --- LEVEL 3: BASIN RUNS (Child of Global Run Task) ---
    for basin in BASINS:
        basin_task = basin_run_template.create_task(
            data_source=DATA_SOURCE,
            model=model_name,
            variant=variant_name,
            scenario=scenario_name,
            time_period=time_period_str,
            basin=basin,
        )
        all_tasks.append(basin_task)
        dependencies.append((basin_task, global_task))


# --- ADD TASKS AND BIND DEPENDENCIES ---
workflow.add_tasks(all_tasks)
print(f"Total tasks created: {len(all_tasks)}")

print(f"Adding {len(dependencies)} dependencies using add_upstream...")
for child_task, parent_task in dependencies:
    child_task.add_upstream(parent_task) 
print("✅ Dependencies successfully added.")

# ============================================================================
# SUBMISSION
# ============================================================================

try:
    workflow.bind()
    print("✅ Workflow successfully bound.")
    print(f"Running workflow with ID {workflow.workflow_id}.")
    print("For full information see the Jobmon GUI:")
    print(f"https://jobmon-gui.ihme.washington.edu/#/workflow/{workflow.workflow_id}")
except Exception as e:
    print(f"❌ Workflow binding failed: {e}")
    exit(1)

try:
    status = workflow.run()
    print(f"Workflow {workflow.workflow_id} completed with status {status}.")

except Exception as e:
    print(f"❌ Workflow submission failed: {e}")
    exit(1)
    exit(1)