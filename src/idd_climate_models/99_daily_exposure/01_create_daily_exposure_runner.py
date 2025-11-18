import getpass
import uuid
from jobmon.client.status_commands import workflow_tasks, task_status # type: ignore
from jobmon.client.tool import Tool # type: ignore
from pathlib import Path

"""
Run from 'idd-climate-models/src/idd_climate_models/99_daily_exposure' directory
"""

# Script directory
SCRIPT_ROOT = Path.cwd()

ROOT_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/tc_risk/output/cmip6") # tc_risk model outputs
# get models from root path folders
MODELS = [p.name for p in ROOT_PATH.iterdir() if p.is_dir()]
VARIANT = "r1i1p1f1"
SCENARIOS = ["ssp126", "ssp245", "ssp585"]
BATCH_YEARS = [
    "2015-2034",
    "2035-2054",
    "2055-2074",
    "2075-2094",
    "2095-2100",
    ]
BASINS = ["EP", "NA", "NI", "SI", "AU", "SP", "WP", "GL"]
DRAWS = list(range(100))  # 0 to 100
RESOULUTION = 0.1  # degrees
OUTPUT_DIR = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/daily_exposure")  # Output directory for daily exposure rasters

# Jobmon setup
user = getpass.getuser()

# Project
project = "proj_rapidresponse"  # Adjust this to your project name if needed

# create jobmon jobs
user = getpass.getuser()
wf_uuid = uuid.uuid4()

# Create a tool
tool = Tool(name="CLIMADA_daily_exposure_generation")


# Create a workflow, and set the executor
workflow = tool.create_workflow(
    name=f"CLIMADA_daily_exposure_generation_{wf_uuid}",
    max_concurrently_running = 500,
)

# Set resources on the workflow
workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={
        "memory": "100G",
        "cores": 1,
        "runtime": "30m",
        "constraints": "archive",
        "queue": "all.q",
        "project": project,  # Ensure the project is set correctly
    }
)


# Define the task template for processing each year batch
task_template = tool.get_task_template(
    template_name="flood_model_standardization_task",
    default_cluster_name="slurm",
    default_compute_resources={
        "queue": "all.q",
        "cores": 1,
        "memory": "100G",
        "runtime": "30m",
        "queue": "all.q",
        "project": project,  # Ensure the project is set correctly
    },
    command_template=(
        "python {script_root}/01_create_daily_exposure.py "
        "--root_path {{root_path}} "
        "--model {{model}} "
        "--variant {{variant}} "
        "--scenario {{scenario}} "
        "--batch_year {{batch_year}} "
        "--basin {{basin}} "
        "--draw {{draw}} "
        "--resolution {{resolution}} "
        "--output_dir {{output_dir}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["root_path", "model", "variant", "scenario", "batch_year", "basin", "draw", "resolution", "output_dir"],  # 👈 Include years in node_args
    task_args=[],
    op_args=[],
)

tasks = []
for model in MODELS:
    for scenario in SCENARIOS:
        for batch_year in BATCH_YEARS:
            for basin in BASINS:
                for draw in DRAWS:
                    task = task_template.create_task(
                        root_path=ROOT_PATH,
                        model=model,
                        variant=VARIANT,
                        scenario=scenario,
                        batch_year=batch_year,
                        basin=basin,
                        draw=draw,
                        resolution=RESOULUTION,
                        output_dir=OUTPUT_DIR,
                    )
                    tasks.append(task)

    
print(f"Number of tasks: {len(tasks)}")

if tasks:
    workflow.add_tasks(tasks)
    print("✅ Tasks successfully added to workflow.")
else:
    print("⚠️ No tasks added to workflow. Check task generation.")

try:
    workflow.bind()
    print("✅ Workflow successfully bound.")
    print(f"Running workflow with ID {workflow.workflow_id}.")
    print("For full information see the Jobmon GUI:")
    print(f"https://jobmon-gui.ihme.washington.edu/#/workflow/{workflow.workflow_id}")
except Exception as e:
    print(f"❌ Workflow binding failed: {e}")

try:
    status = workflow.run()
    print(f"Workflow {workflow.workflow_id} completed with status {status}.")
except Exception as e:
    print(f"❌ Workflow submission failed: {e}")
