import importlib
import sys
import getpass
import uuid
import os
from jobmon.client.tool import Tool 
from pathlib import Path

# NOTE: These imports rely on external module definitions (constants, io_compare_utils, etc.)
import idd_climate_models.constants as rfc
from idd_climate_models.io_compare_utils import compare_model_validation
from idd_climate_models.dictionary_utils import parse_results
from idd_climate_models.resource_functions import get_rep_file_size_gb, get_resource_info

# --- CONSTANT DEFINITIONS (from rfc) ---
repo_name = rfc.repo_name
package_name = rfc.package_name
DATA_DIR = rfc.RAW_DATA_PATH
PROCESSED_DATA_PATH = rfc.PROCESSED_DATA_PATH
TC_RISK_INPUT_PATH = rfc.TC_RISK_INPUT_PATH
SCRIPT_ROOT = rfc.REPO_ROOT / repo_name / "src" / package_name / "01_run_tc_risk"

# Configuration

DATA_SOURCE = "cmip6"
BIN_SIZE_YEARS = 20
VERBOSE = True


INPUT_DATA_TYPE = "data"
INPUT_IO_TYPE = "processed"
OUTPUT_DATA_TYPE = "tc_risk"
OUTPUT_IO_TYPE = "input"



# ============================================================================
# DATA SETUP & VALIDATION
# ============================================================================

# Use the unified function for validation and comparison
validation_info = compare_model_validation(
    input_data_type=INPUT_DATA_TYPE,
    input_io_type=INPUT_IO_TYPE,
    output_data_type=OUTPUT_DATA_TYPE,
    output_io_type=OUTPUT_IO_TYPE,
    data_source=DATA_SOURCE,
    verbose=False
)

models_to_process = validation_info["models_to_process"]
model_variants_to_run = parse_results(validation_info["models_to_process_dict"], 'variant')

if not model_variants_to_run:
    print("✅ All jobs are processed. No tasks to run.")
    sys.exit(0)
    
# Get the full hierarchy list to build the variable detail map
full_path_list = parse_results(validation_info["models_to_process_dict"], 'all')
variable_detail_map = {}

# Build the map: {(model, variant, scenario, variable): {'grid': 'gn', 'frequency': 'day'}}
for item in full_path_list:
    key = (
        item['model'],
        item['variant'],
        item['scenario'],
        item['variable']
    )
    # Store the unique grid and frequency needed to build the source_dir
    variable_detail_map[key] = {
        'grid': item['grid'],
        'frequency': item['frequency']
    }

def get_time_bins(scenario_name, bin_size_years):
    date_ranges = rfc.VALIDATION_RULES['tc_risk']['time-period']['date_ranges']
    if scenario_name not in date_ranges:
        print(f"Warning: No date range found for scenario '{scenario_name}'")
        return []
    start_year, end_year = date_ranges[scenario_name]
    return [(y, min(y + bin_size_years - 1, end_year)) for y in range(start_year, end_year + 1, bin_size_years)]

TIME_BINS = {
    scenario: get_time_bins(scenario, BIN_SIZE_YEARS)
    for scenario in rfc.SCENARIOS
}

# ============================================================================
# JOBMON SETUP
# ============================================================================

user = getpass.getuser()

log_dir = Path("/mnt/team/idd/pub/")
log_dir.mkdir(parents=True, exist_ok=True)
stdout_dir = log_dir / "stdout"
stderr_dir = log_dir / "stderr" 
stdout_dir.mkdir(parents=True, exist_ok=True)
stderr_dir.mkdir(parents=True, exist_ok=True)

project = "proj_rapidresponse"
queue = 'all.q' # Switched to 'all.q' for general use, but kept long.q settings in template

wf_uuid = uuid.uuid4()
tool_name = f"{package_name}_tc_risk_reorganization_tool"
tool = Tool(name=tool_name)

workflow = tool.create_workflow(
    name=f"{tool_name}_workflow_{wf_uuid}",
    max_concurrently_running=5000,
)

# Default compute resources for small tasks (Folder Creation)
workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={
        "memory": "5G",
        "cores": 1,
        "runtime": "10m",
        "queue": queue,
        "project": project,
    }
)

# LEVEL 1: Folder creation task template
folder_template = tool.get_task_template(
    template_name="create_folders",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "1G",
        "cores": 1,
        "runtime": "5m",
        "queue": queue,
        "project": project,
    },
    command_template=(
        "python {script_root}/1a_create_tc_risk_input_folder.py "
        "--data_source {{data_source}} " 
        "--model {{model}} "
        "--variant {{variant}} " 
        "--scenario {{scenario}} "
        "--time_bin {{time_bin}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["data_source", "model", "variant", "scenario", "time_bin"],
    task_args=[],
    op_args=[],
)

# LEVEL 2: Processing task template (Dynamic Resources must be applied here)
process_template = tool.get_task_template(
    template_name="process_variable_frequency",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "15G",
        "cores": 4, # Changed from 1 to match expected need for xarray ops
        "runtime": "1h",
        "queue": 'long.q', # Default to a longer queue for processing
        "project": project,
    },
    command_template=(
        "python {script_root}/1b_process_time_chunk.py "
        "--data_source {{data_source}} "
        "--model {{model}} "
        "--variant {{variant}} "
        "--scenario {{scenario}} "
        "--time_bin {{time_bin}} "
        "--variable {{variable}} "
        "--grid {{grid}} "
        "--frequency {{frequency}} "
        "--needs_regridding_str {{needs_regridding_str}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["data_source", "model", "variant", "scenario", "time_bin", "variable", "grid", "frequency", "needs_regridding_str"],
    task_args=[],
    op_args=[],
)

# ============================================================================
# TASK CREATION AND DEPENDENCY SETUP (Single Loop)
# ============================================================================

folder_task_map = {}
folder_tasks = []
process_tasks = []
dependencies = []

print("\n" + "=" * 80)
print("STEP 3: Creating Jobmon tasks (Dynamic Resources)")
print("=" * 80)

for mv_info in model_variants_to_run:
    model_name = mv_info['model']
    variant_name = mv_info['variant']
    
    for scenario in rfc.SCENARIOS:
        for time_bin_tuple in TIME_BINS[scenario]:
            time_bin_str = f"{time_bin_tuple[0]}-{time_bin_tuple[1]}"
            
            parent_key = (model_name, variant_name, scenario, time_bin_str)
            
            # --- LEVEL 1: CREATE PARENT (FOLDER) TASK ---
            folder_task = folder_template.create_task(
                data_source = DATA_SOURCE,
                model = model_name,
                variant = variant_name,
                scenario = scenario,
                time_bin = time_bin_str,
            )
            
            folder_tasks.append(folder_task)
            folder_task_map[parent_key] = folder_task
            
            # --- LEVEL 2: CREATE CHILD (PROCESSING) TASKS AND DEPENDENCIES ---
            for variable in rfc.VARIABLES[DATA_SOURCE]:
                
                detail_lookup_key = (model_name, variant_name, scenario, variable)
                details = variable_detail_map.get(detail_lookup_key)
                
                if not details:
                    continue 

                # --- DYNAMIC RESOURCE CALCULATION ---
                
                # 1. CONSTRUCT PATH TO A REPRESENTATIVE INPUT FILE
                source_file_dir = PROCESSED_DATA_PATH / DATA_SOURCE / model_name / variant_name / scenario / variable / details['grid'] / details['frequency']
                resource_request, needs_regridding = get_resource_info(file_path=source_file_dir, representative='first', num_files = BIN_SIZE_YEARS)
                if VERBOSE:
                    print(f"Variable: {variable} | Model: {model_name} | Variant: {variant_name} | Scenario: {scenario} | Time Bin: {time_bin_str} ->" )
                    print(f"        Requesting Mem: {resource_request['memory']}, Run: {resource_request['runtime']}, Cores: {resource_request['cores']}, Regridding: {needs_regridding}")
                # --- TASK CREATION ---
                process_task = process_template.create_task(
                    # Inject dynamic resources here
                    compute_resources={
                        "memory": resource_request["memory"],
                        "cores": resource_request["cores"],
                        "runtime": resource_request["runtime"],
                        "queue": queue, # Use the default queue if specific queue not needed
                        "project": project,
                    },
                    data_source = DATA_SOURCE,
                    model = model_name,
                    variant = variant_name,
                    scenario = scenario,
                    time_bin = time_bin_str,
                    variable = variable,
                    grid = details['grid'],
                    frequency = details['frequency'],
                    needs_regridding_str=str(needs_regridding),
                )
                
                process_tasks.append(process_task)
                
                # Add dependency: process_task depends on folder_task
                dependencies.append((process_task, folder_task))


# Add all tasks to workflow
all_tasks = folder_tasks + process_tasks

print(f"\nTotal tasks: {len(all_tasks)} ({len(folder_tasks)} folder + {len(process_tasks)} processing)")

workflow.add_tasks(all_tasks)
print("✅ Tasks successfully added to workflow.")

# Add dependencies by calling add_dependency on each child task
print(f"Adding {len(dependencies)} dependencies...")
for child_task, parent_task in dependencies:
    # Use the method shown in the Burdenator example
    child_task.add_upstream(parent_task) 
print("✅ Dependencies successfully added.")

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