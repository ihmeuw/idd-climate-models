import importlib
import sys
import getpass
import uuid
from jobmon.client.tool import Tool  # type: ignore
from pathlib import Path

import idd_climate_models.constants as rfc
from idd_climate_models.validate_model_functions import *

repo_name = rfc.repo_name
package_name = rfc.package_name
DATA_DIR = rfc.DATA_PATH
PROCESSED_DATA_PATH = rfc.PROCESSED_DATA_PATH
SCRIPT_ROOT = rfc.REPO_ROOT / repo_name / "src" / package_name / "clean_and_split"

# Configuration
DATA_SOURCE = "cmip6"  # Data source name (used in log filename)
RERUN = False  # Set to True to reprocess everything, False to skip already processed files

# Step 1: Find unprocessed models using the processing log
# This only validates models/variants not already marked as complete
print("=" * 80)
print("STEP 1: Finding unprocessed models")
print("=" * 80)
results = find_unprocessed_models(
    DATA_DIR, 
    PROCESSED_DATA_PATH,
    data_source=DATA_SOURCE, 
    verbose=True
)

if not results:
    print("\n✅ All models have been processed! Nothing to do.")
    exit(0)

# Step 2: Filter out files that are already processed
# This does detailed file-level checking only for unprocessed variants
print("\n" + "=" * 80)
print("STEP 2: Checking for already processed files")
print("=" * 80)
results = filter_already_processed(
    results, 
    PROCESSED_DATA_PATH,
    data_source=DATA_SOURCE,
    rerun=RERUN,
    verbose=True
)

if not results:
    print("\n✅ All remaining files have been processed! Nothing to do.")
    exit(0)

complete_models = get_complete_models(results)
print(f"\nComplete models to process: {len(complete_models)}")
print(complete_models)

# Jobmon setup
user = getpass.getuser()

log_dir = Path("/mnt/team/idd/pub/")
log_dir.mkdir(parents=True, exist_ok=True)
stdout_dir = log_dir / "stdout"
stderr_dir = log_dir / "stderr"
stdout_dir.mkdir(parents=True, exist_ok=True)
stderr_dir.mkdir(parents=True, exist_ok=True)

# Project
project = "proj_rapidresponse"
queue = 'all.q'

wf_uuid = uuid.uuid4()
tool_name = f"{package_name}_fill_and_split_tool"
tool = Tool(name=tool_name)

# Create a workflow
workflow = tool.create_workflow(
    name=f"{tool_name}_workflow_{wf_uuid}",
    max_concurrently_running=10000,
)

# Compute resources
workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={
        "memory": "15G",
        "cores": 1,
        "runtime": "5m",
        "queue": queue,
        "project": project,
        "stdout": str(stdout_dir),
        "stderr": str(stderr_dir),
    }
)

# Define the task template
task_template = tool.get_task_template(
    template_name="malaria_as_calculation",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "30G",
        "cores": 4,
        "runtime": "5m",
        "queue": queue,
        "project": project,
        "stdout": str(stdout_dir),
        "stderr": str(stderr_dir),
    },
    command_template=(
        "python {script_root}/fill_and_yearly_split.py "
        "--data_source {{data_source}} "
        "--model {{model}} "
        "--variant {{variant}} "
        "--scenario {{scenario}} "
        "--variable {{variable}} "
        "--grid {{grid}} "
        "--time_period {{time_period}} "
        "--file_path {{file_path}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["data_source", "model", "variant", "scenario", "variable", "grid", "time_period", "file_path"],
    task_args=[],
    op_args=[],
)

# Add tasks
print("\n" + "=" * 80)
print("STEP 3: Creating Jobmon tasks")
print("=" * 80)
tasks = []
for model_name in complete_models:
    model_data = results[model_name]
    # Use the iterator to simplify looping
    for variant, scenario, variable, grid, time_period, file_path, _ in iterate_model_files(model_data):
        task = task_template.create_task(
            data_source=DATA_SOURCE,
            model=model_name,
            variant=variant,
            scenario=scenario,
            variable=variable,
            grid=grid,
            time_period=time_period,
            file_path=file_path,
        )
        tasks.append(task)

print(f"Number of tasks to run: {len(tasks)}")

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