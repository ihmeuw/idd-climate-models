import importlib
import sys
import getpass
import uuid
import os
import copy
import pandas as pd
from jobmon.client.tool import Tool 
from pathlib import Path

# NOTE: Imports required for the execution logic
import idd_climate_models.constants as rfc
from idd_climate_models.dictionary_utils import parse_results
from idd_climate_models.validation_functions import create_validation_dict, validate_all_models_in_source 

# ----------------------------------------------------------------------------

# --- CONSTANT DEFINITIONS ---
repo_name = rfc.repo_name
package_name = rfc.package_name
SCRIPT_ROOT = rfc.REPO_ROOT / repo_name / "src" / package_name / "02_run_climada"

NUM_DRAWS = rfc.NUM_DRAWS

# Configuration
DATA_SOURCE = "cmip6"
INPUT_DATA_TYPE = "tc_risk"
INPUT_IO_DATA_TYPE = "output"
OUTPUT_DATA_TYPE = "climada"
OUTPUT_IO_DATA_TYPE = "input"
BASINS = ['EP', 'NA', 'NI', 'SI', 'AU', 'SP', 'WP']
TEST_RUN = 0  # Set to 0 for full run, >0 for test runs

# ============================================================================
# STAGE 1: LOAD TC-RISK REGISTRY
# ============================================================================

print("=" * 80)
print("STEP 1: Loading TC-risk registry to identify model/variant/scenario/time_period combinations")
print("=" * 80)

# Path to the TC-risk registry created by 01_create_folders.py
tc_registry_path = rfc.TC_RISK_INPUT_PATH / DATA_SOURCE / "folder_paths_registry.csv"

if not tc_registry_path.exists():
    print(f"❌ ERROR: TC-risk registry file not found at {tc_registry_path}")
    print("Please run the TC-risk pipeline first to generate the registry.")
    sys.exit(1)

tc_registry_df = pd.read_csv(tc_registry_path)
print(f"Loaded {len(tc_registry_df)} model/variant/scenario/time_period combinations from TC-risk registry")

# Get unique combinations from the TC-risk registry
unique_combos = tc_registry_df[['model', 'variant', 'scenario', 'time_period']].drop_duplicates()

# Apply test run limit if specified
if TEST_RUN > 0:
    print(f"\n⚠️  TEST MODE: Limiting to {TEST_RUN} combinations")
    unique_combos = unique_combos.iloc[:TEST_RUN]

print(f"Will process {len(unique_combos)} combinations\n")

# ============================================================================
# STAGE 2: CREATE CLIMADA REGISTRY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Creating CLIMADA registry with paths for all combinations")
print("=" * 80)

climada_registry_rows = []

for _, row in unique_combos.iterrows():
    model_name = row['model']
    variant_name = row['variant']
    scenario_name = row['scenario']
    time_period_str = row['time_period']
    
    # Get basins for this combination
    key = (model_name, variant_name, scenario_name, time_period_str)

    for basin in BASINS:
        # Get TC-risk output path from TC registry
        tc_output_path = rfc.TC_RISK_OUTPUT_PATH / DATA_SOURCE / model_name / variant_name / scenario_name / time_period_str
        
        # Get CLIMADA input path
        climada_input_path = rfc.CLIMADA_INPUT_PATH / DATA_SOURCE / model_name / variant_name / scenario_name / time_period_str / basin
        
        climada_registry_rows.append({
            'model': model_name,
            'variant': variant_name,
            'scenario': scenario_name,
            'time_period': time_period_str,
            'basin': basin,
            'tc_risk_output_path': str(tc_output_path),
            'climada_input_path': str(climada_input_path),
        })

climada_registry_df = pd.DataFrame(climada_registry_rows)

# Save CLIMADA registry
climada_registry_path = rfc.CLIMADA_INPUT_PATH / DATA_SOURCE / "folder_paths_registry.csv"
climada_registry_path.parent.mkdir(parents=True, exist_ok=True)
climada_registry_df.to_csv(climada_registry_path, index=False)

print(f"Created CLIMADA registry with {len(climada_registry_df)} rows")
print(f"Saved to: {climada_registry_path}")

# ============================================================================
# JOBMON SETUP & TEMPLATES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Setting up Jobmon workflow")
print("=" * 80)

user = getpass.getuser()
project = "proj_rapidresponse"
queue = 'all.q'
wf_uuid = uuid.uuid4()
tool_name = f"{package_name}_climada_data_prep"
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

# --- Template 1: Folder Creation (1a) ---
folder_template = tool.get_task_template(
    template_name="create_input_folders",
    default_cluster_name="slurm", 
    command_template=(
        "python {script_root}/1a_create_climada_input_folder.py "
        "--input_data_type {{input_data_type}} "
        "--input_io_data_type {{input_io_data_type}} "
        "--output_data_type {{output_data_type}} "
        "--output_io_data_type {{output_io_data_type}} "
        "--data_source {{data_source}} " 
        "--model {{model}} "
        "--variant {{variant}} " 
        "--scenario {{scenario}} "
        "--time_period {{time_period}} "
        "--basin {{basin}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["input_data_type", "input_io_data_type", "output_data_type", "output_io_data_type", "data_source", "model", "variant", "scenario", "time_period", "basin"],
)

# --- Template 2: Data Prep (1b) ---
data_prep_template = tool.get_task_template(
    template_name="create_climada_input_data",
    default_cluster_name="slurm", 
    default_compute_resources={ 
        "memory": "40G", "cores": rfc.tc_risk_n_procs + 1, "runtime": "1h", # core: 17, max runtime: 20m in test, max mem: 6.1G
        "queue": 'long.q', "project": project,
    },
    command_template=(
        "python {script_root}/1b_process_basin_draw.py "
        "--input_data_type {{input_data_type}} "
        "--input_io_data_type {{input_io_data_type}} "
        "--output_data_type {{output_data_type}} "
        "--output_io_data_type {{output_io_data_type}} "
        "--data_source {{data_source}} "
        "--model {{model}} "
        "--variant {{variant}} "
        "--scenario {{scenario}} "
        "--time_period {{time_period}} "
        "--basin {{basin}} "
        "--draw {{draw}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["input_data_type", "input_io_data_type", "output_data_type", "output_io_data_type", "data_source", "model", "variant", "scenario", "time_period", "basin", "draw"],
)

# ============================================================================
# TASK CREATION AND NESTED DEPENDENCIES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Creating Jobmon tasks with dependencies")
print("=" * 80)

all_tasks = []
dependencies = []

for _, row in unique_combos.iterrows():
    model_name = row['model']
    variant_name = row['variant']
    scenario_name = row['scenario']
    time_period_str = row['time_period']
    
    for basin in BASINS:
        # --- LEVEL 1: FOLDER CREATION (Parent Task) ---
        folder_task = folder_template.create_task(
            input_data_type=INPUT_DATA_TYPE,
            input_io_data_type=INPUT_IO_DATA_TYPE,
            output_data_type=OUTPUT_DATA_TYPE,
            output_io_data_type=OUTPUT_IO_DATA_TYPE,
            data_source=DATA_SOURCE,
            model=model_name,
            variant=variant_name,
            scenario=scenario_name,
            time_period=time_period_str,
            basin=basin,
        )
        all_tasks.append(folder_task)

        for draw in range(NUM_DRAWS):
            # --- LEVEL 2: DATA PREP (Child of Folder Task) ---
            data_prep_task = data_prep_template.create_task(
                input_data_type=INPUT_DATA_TYPE,
                input_io_data_type=INPUT_IO_DATA_TYPE,
                output_data_type=OUTPUT_DATA_TYPE,
                output_io_data_type=OUTPUT_IO_DATA_TYPE,
                data_source=DATA_SOURCE,
                model=model_name,
                variant=variant_name,
                scenario=scenario_name,
                time_period=time_period_str,
                basin=basin,
                draw=draw,
            )
            all_tasks.append(data_prep_task)
            dependencies.append((data_prep_task, folder_task))


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