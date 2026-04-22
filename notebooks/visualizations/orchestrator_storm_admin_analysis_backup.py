"""
Orchestrator for storm-admin0 impact analysis.

Parallelizes over the same draw batches as the main TC-risk pipeline,
using the level_4_task_assignments.csv file.

Each task processes multiple draws for a specific basin/model/variant/scenario/time_period.
"""

import sys
import getpass
import uuid
import pandas as pd
from pathlib import Path
from jobmon.client.tool import Tool

import idd_climate_models.constants as rfc

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_SOURCE = "cmip6"
PROJECT = "proj_rapidresponse"
QUEUE = 'long.q'

# Resource configuration per task
TASK_MEMORY = "1G"
TASK_CORES = 2
TASK_RUNTIME = "5m"
MAX_CONCURRENT_TASKS = 10000

# Test mode - set to N to only process first N tasks (0 = all tasks)
TEST_RUN = 0

# Scripts
ANALYSIS_SCRIPT = Path(__file__).parent / "run_storm_admin_analysis.py"  # Level 1: draw processing
NOTEBOOK_SCRIPT = Path(__file__).parent / "run_analysis_notebook.py"  # Level 2: notebook execution

# Task assignments file (created by main orchestrator Level 0)
TASK_ASSIGNMENTS_FILE = rfc.CLIMADA_INPUT_PATH / DATA_SOURCE / "level_4_task_assignments.csv"

# Output directory for results
OUTPUT_DIR = Path("/ihme/homes/bcreiner/repos/idd-climate-models/outputs/storm_admin_impacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Level 2 resources (notebook execution)
NOTEBOOK_MEMORY = "10G"
NOTEBOOK_CORES = 4
NOTEBOOK_RUNTIME = "30m"

# ============================================================================
# VALIDATE INPUTS
# ============================================================================

print("=" * 80)
print("Storm-Admin0 Impact Analysis Orchestrator")
print("=" * 80)

if not ANALYSIS_SCRIPT.exists():
    print(f"\n❌ ERROR: Analysis script not found: {ANALYSIS_SCRIPT}")
    print("\nYou need to create this script. It should:")
    print("  - Accept --task_id argument")
    print("  - Read task assignments from level_4_task_assignments.csv")
    print("  - Process all draws for that task_id")
    print("  - Save results to output directory")
    sys.exit(1)

if not NOTEBOOK_SCRIPT.exists():
    print(f"\n❌ ERROR: Notebook script not found: {NOTEBOOK_SCRIPT}")
    print("\nYou need to create this script.")
    sys.exit(1)

if not TASK_ASSIGNMENTS_FILE.exists():
    print(f"Level 1 script: {ANALYSIS_SCRIPT}")
print(f"✓ Level 2 script: {NOTEBOOKs file not found: {TASK_ASSIGNMENTS_FILE}")
    print("\nRun the main orchestrator with STARTING_LEVEL=0 to create task assignments:")
    print("  python src/idd_climate_models/01_process_raw_through_climada_input/00_orchestrator.py")
    sys.exit(1)

print(f"\n✓ Analysis script: {ANALYSIS_SCRIPT}")
print(f"✓ Task assignments: {TASK_ASSIGNMENTS_FILE}")
print(f"✓ Output directory: {OUTPUT_DIR}")

# ============================================================================
# LOAD TASK ASSIGNMENTS
# ============================================================================

print("\n" + "=" * 80)
print("Loading task assignments")
print("=" * 80)

df_assignments = pd.read_csv(TASK_ASSIGNMENTS_FILE, keep_default_na=False)
total_tasks = df_assignments['task_id'].nunique()

print(f"\nTask assignments summary:")
print(f"  Total tasks: {total_tasks}")
print(f"  Total draw-basin combinations: {len(df_assignments)}")

# Show breakdown by basin
basin_counts = df_assignments.groupby('basin').size()
print(f"\n  Draws per basin:")
for basin, count in basin_counts.items():
    print(f"    {basin}: {count}")

# Show breakdown by combination
combo_counts = df_assignments.groupby(['model', 'variant', 'scenario', 'time_period']).size()
print(f"\n  Unique combinations: {len(combo_counts)}")
print(f"  First few:")
for i, (combo, count) in enumerate(combo_counts.head().items()):
    model, variant, scenario, time_period = combo
    print(f"    {model}/{variant}/{scenario}/{time_period}: {count} draws")

# Apply test mode if enabled
if TEST_RUN > 0:
    print(f"\n⚠️  TEST MODE: Limiting to first {TEST_RUN} tasks")
    task_ids_to_run = list(range(1, min(TEST_RUN + 1, total_tasks + 1)))
else:
    task_ids_to_run = list(range(1, total_tasks + 1))

print(f"\nWill create {len(task_ids_to_run)} Jobmon tasks")

# ============================================================================
# CREATE JOBMON WORKFLOW
# ============================================================================

print("\n" + "=" * 80)
print("Setting up Jobmon workflow")
print("=" * 80)

user = getpass.getuser()
wf_uuid = uuid.uuid4()
tool_name = "storm_admin_impact_analysis"

tool = Tool(name=tool_name)

workflow = tool.create_workflow(
    name=f"{tool_name}_{wf_uuid}",
    max_concurrently_running=MAX_CONCURRENT_TASKS,
)

# Set default compute resources
workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={
        "memory": TASK_MEMORY,
        "cores": TASK_CORES,
        "runtime": TASK_RUNTIME,
        "queue": QUEUE,
        "project": PROJECT,
        "constraints": "archive",
    }
)
s
# Level 1: Process draws (one task per task_id)
level1_template = tool.get_task_template(
    template_name="storm_admin_analysis_level1",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": TASK_MEMORY,
        "cores": TASK_CORES,
        "runtime": TASK_RUNTIME,
        "queue": QUEUE,
        "project": PROJECT,
        "constraints": "archive",
    },
    command_template=(
        f"python {ANALYSIS_SCRIPT} "
        "--data_source {data_source} "
        "--task_id {task_id} "
        "--output_dir {output_dir}"
    ),
    node_args=["data_source", "task_id", "output_dir"],
)

# Level 2: Run analysis notebook (one task per model/variant/scenario)
level2_teLEVEL 1 TASKS (Draw Processing)
# ============================================================================

print("\n" + "=" * 80)
print("Creating Level 1 tasks (draw processing)")
print("=" * 80)

level1_tasks = []
task_id_to_task = {}  # For dependency tracking

for task_id in task_ids_to_run:
    task = level1_template.create_task(
        data_source=DATA_SOURCE,
        task_id=str(task_id),
        output_dir=str(OUTPUT_DIR),
    )
    level1_tasks.append(task)
    task_id_to_task[task_id] = task

print(f"\n✓ Created {len(level1_tasks)} Level 1 tasks")

# ============================================================================
# CREATE LEVEL 2 TASKS (Notebook Execution)
# ============================================================================

print("\n" + "=" * 80)
print("Creating Level 2 tasks (notebook execution)")
print("=" * 80)

# Get unique model/variant/scenario combinations
combinations = df_assignments[['model', 'variant', 'scenario']].drop_duplicates()
print(f"\nFound {len(combinations)} unique model/variant/scenario combinations")

level2_tasks = []
dependencies_to_add = []

for _, row in combinations.iterrows():
    model = row['model']
    variant = row['variant']
    scenario = row['scenario']
    
    # Create Level 2 task
    task = level2_template.create_task(
        data_source=DATA_SOURCE,
        model=model,
        variant=variant,
        scenario=scenario,
    )
    leveLevel 1 tasks: {len(level1_tasks)}")
print(f"  Resources: {TASK_MEMORY}, {TASK_CORES} cores, {TASK_RUNTIME}")
print(f"Level 2 tasks: {len(level2_tasks)}")
print(f"  Resources: {NOTEBOOK_MEMORY}, {NOTEBOOK_CORES} cores, {NOTEBOOK_RUNTIME}")
print(f"Total tasks: {len(all_tasks)}")
print(f"Dependencies: {len(dependencies_to_add)
    # Find all Level 1 tasks for this combination
    matching_task_ids = df_assignments[
        (df_assignments['model'] == model) &
        (df_assignments['variant'] == variant) &
        (df_assignments['scenario'] == scenario)
    ]['task_id'].unique()
    
    # Track dependencies
    for task_id in matching_task_ids:
        if task_id in task_id_to_task:
            dependencies_to_add.append((task, task_id_to_task[task_id]))
    
    print(f"  {model}/{variant}/{scenario}: depends on {len(matching_task_ids)} Level 1 tasks")

print(f"\n✓ Created {len(level2_tasks)} Level 2 tasks")
print(f"✓ Will add {len(dependencies_to_add)} dependencies")

# ============================================================================
# ADD ALL TASKS TO WORKFLOW
# ============================================================================

all_tasks = level1_tasks + level2_tasks
workflow.add_tasks(all_tasks)

# ============================================================================
# ADD DEPENDENCIES
# ============================================================================

print("\n" + "=" * 80)
print("Adding dependencies")
print("=" * 80)

for level2_task, level1_task in dependencies_to_add:
    level2_task.add_upstream(level1_task)

print(f"✓ Added {len(dependencies_to_add)} dependencies using add_upstream"

# ============================================================================
# CREATE TASKS
# ============================================================================

print("\n" + "=" * 80)
print("Creating Jobmon tasks")
print("=" * 80)

all_tasks = []

for task_id in task_ids_to_run:
    task = task_template.create_task(
        data_source=DATA_SOURCE,
        task_id=str(task_id),
        output_dir=str(OUTPUT_DIR),
    )
    all_tasks.append(task)

print(f"\n✓ Created {len(all_tasks)} tasks")

# Add tasks to workflow
workflow.add_tasks(all_tasks)

print("\n" + "=" * 80)
print("Workflow Summary")
print("=" * 80)
print(f"Tasks: {len(all_tasks)}")
print(f"Resources per task: {TASK_MEMORY}, {TASK_CORES} cores, {TASK_RUNTIME}")
print(f"Max concurrent: {workflow.max_concurrently_running}")

# ============================================================================
# BIND AND RUN WORKFLOW
# ============================================================================

print("\n" + "=" * 80)
print("Binding and submitting workflow")
print("=" * 80)

try:
    workflow.bind()
    print("✓ Workflow successfully bound")
    print(f"✓ Workflow ID: {workflow.workflow_id}")
    print(f"\n🔗 Jobmon GUI:")
    print(f"   https://jobmon-gui.ihme.washington.edu/#/workflow/{workflow.workflow_id}")
except Exception as e:
    print(f"❌ Workflow binding failed: {e}")
    sys.exit(1)

try:
    print(f"\n🚀 Running workflow...")
    status = workflow.run(seconds_until_timeout=86400)  # 24 hour timeout
    print(f"\n✅ Workflow {workflow.workflow_id} completed with status: {status}")
except Exception as e:
    print(f"\n❌ Workflow execution failed: {e}")
    sys.exit(1)
