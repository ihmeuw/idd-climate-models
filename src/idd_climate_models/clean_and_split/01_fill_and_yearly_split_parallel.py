import sys
import getpass
import uuid
import os
from jobmon.client.tool import Tool # type: ignore
from pathlib import Path
from typing import Dict, Any

import idd_climate_models.constants as rfc
from idd_climate_models.dictionary_utils import parse_results
from idd_climate_models.io_compare_utils import compare_model_validation

repo_name = rfc.repo_name
package_name = rfc.package_name
DATA_DIR = rfc.RAW_DATA_PATH 
PROCESSED_DATA_PATH = rfc.PROCESSED_DATA_PATH 
SCRIPT_ROOT = rfc.REPO_ROOT / repo_name / "src" / package_name / "clean_and_split"

INPUT_DATA_TYPE = "data"
INPUT_IO_TYPE = "raw"
OUTPUT_DATA_TYPE = "data"
OUTPUT_IO_TYPE = "processed" 
DATA_SOURCE = "cmip6" 

TEST_MODE = False

def get_file_size_gb(file_path: Path) -> float:
    return os.path.getsize(file_path) / (1024**3)

def get_resource_tier(file_size_gb: float, REQUIRED_MEM_FACTOR: float = 4.0,
                      MIN_MEM_GB: float = 8.0, MAX_MEM_GB: float = 64.0) -> Dict[str, Any]:
    required_mem_gb = int(file_size_gb * REQUIRED_MEM_FACTOR) + 2 
    memory = f"{min(MAX_MEM_GB, max(MIN_MEM_GB, required_mem_gb))}G"
    if file_size_gb < 1.0:
        runtime = "10m"
        cores = 4
    elif file_size_gb < 5.0: 
        runtime = "20m" 
        cores = 4
    else: 
        runtime = "30m"
        cores = 8 
    return {
        "memory": memory,
        "cores": cores,
        "runtime": runtime
    }

# ================================================================================

# Use the unified function for validation and comparison
validation_info = compare_model_validation(
    input_data_type=INPUT_DATA_TYPE,
    input_io_type=INPUT_IO_TYPE,
    output_data_type=OUTPUT_DATA_TYPE,
    output_io_type=OUTPUT_IO_TYPE,
    data_source=DATA_SOURCE,
    verbose=True
)

models_to_process = validation_info["models_to_process"]
input_results_for_tasks = validation_info["models_to_process_dict"]["validation_results"]
input_complete_models = validation_info["input_complete_models"]
output_complete_models = validation_info["output_complete_models"]


tasks_to_run = parse_results(
    validation_dict = validation_info["models_to_process_dict"],
    detail='all',
)


processed_only_models = output_complete_models - input_complete_models

if processed_only_models:
    print("\n⚠️ WARNING: Found PROCESSED models that are NOT complete in RAW data:")
    print(f"   {processed_only_models}")
    print("   These models may have been processed from incomplete source data.")

print("\n" + "=" * 80)
print(f"SUMMARY: {len(input_complete_models)} complete RAW models found.")
print(f"         {len(output_complete_models)} complete PROCESSED models found.")
print(f"         {len(models_to_process)} unique models require processing.")
print(f"         Resulting in {len(tasks_to_run)} files to process.")
print("=" * 80)

if not tasks_to_run:
    print("\n✅ Execution halted: All required files have either been processed or are incomplete in the raw data.")
    exit(0)

user = getpass.getuser()
log_dir = Path("/mnt/team/idd/pub/")
log_dir.mkdir(parents=True, exist_ok=True)
stdout_dir = log_dir / "stdout"
stderr_dir = log_dir / "stderr"
stdout_dir.mkdir(parents=True, exist_ok=True)
stderr_dir.mkdir(parents=True, exist_ok=True)

project = "proj_rapidresponse"
queue = 'all.q'

wf_uuid = uuid.uuid4()
tool_name = f"{package_name}_fill_and_split_tool"
tool = Tool(name=tool_name)

workflow = tool.create_workflow(
    name=f"{tool_name}_workflow_{wf_uuid}",
    max_concurrently_running=10000,
)

workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={
        "memory": "15G", 
        "cores": 1,
        "runtime": "5m",
        "queue": queue,
        "project": project,
    }
)

task_template = tool.get_task_template(
    template_name="fill_and_yearly_split",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "8G",
        "cores": 1,
        "runtime": "10m",
        "queue": queue,
        "project": project,
    },
    command_template=(
        "python {script_root}/fill_and_yearly_split.py "
        "--data_source {{data_source}} "
        "--model {{model}} "
        "--variant {{variant}} "
        "--scenario {{scenario}} "
        "--variable {{variable}} "
        "--grid {{grid}} "
        "--frequency {{frequency}} "
        "--file_path {{file_path}} "
        "--fill_required {{fill_required}}"
    ).format(script_root=SCRIPT_ROOT), 
    node_args=[
        "data_source", "model", "variant", "scenario", 
        "variable", "grid", "frequency", "file_path", 
        "fill_required"
    ],
    task_args=[], 
    op_args=[],
)

print("\n" + "=" * 80)
print("STEP 3: Creating Jobmon tasks (Dynamic Resources)")
print("=" * 80)
tasks = []

for file_data in tasks_to_run:
    full_file_path = Path(file_data['file_path'])
    try:
        file_size_gb = get_file_size_gb(full_file_path)
    except FileNotFoundError:
        print(f"Skipping task: File not found at {full_file_path}")
        continue
    resource_request = get_resource_tier(file_size_gb)
    print(f"File: {full_file_path.name} ({file_size_gb:.2f} GB) -> Requesting Mem: {resource_request['memory']}, Run: {resource_request['runtime']}")
    task = task_template.create_task(
        compute_resources={
            "memory": resource_request["memory"],
            "cores": resource_request["cores"],
            "runtime": resource_request["runtime"],
            "queue": queue,
            "project": project,
        },
        data_source=DATA_SOURCE,
        model=file_data['model'],
        variant=file_data['variant'],
        scenario=file_data['scenario'],
        variable=file_data['variable'],
        grid=file_data['grid'],
        frequency=file_data['frequency'],
        file_path=file_data['file_path'],
        fill_required=file_data['fill_required']
    )
    tasks.append(task)

print(f"Number of tasks to run: {len(tasks)}")

if TEST_MODE:
    print("\n✅ TEST MODE is ON. Exiting before running the workflow.")
    exit(0)
if tasks:
    workflow.add_tasks(tasks)
    print("✅ Tasks successfully added to workflow.")
else:
    print("⚠️ No tasks to run. All files may already be processed.")
    exit(0)

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