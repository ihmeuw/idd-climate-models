"""
Orchestrator for CLIMADA intensity raster → admin impact analysis.

Two-level workflow:
- Level 1: Process draws (one task per task_id from level_4_task_assignments.csv)
           Runs run_climada_storm_analysis.py
- Level 2: Run summary aggregation (one task per model/variant/scenario combination)
           Runs run_climada_summary.py

Level 2 tasks depend on all Level 1 tasks for their model/variant/scenario.

Output: Zarr/NetCDF files with storm counts by year/month/country/severity
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

# Level 1 resources (draw processing - more memory for raster operations)
TASK_MEMORY = "8G"
TASK_CORES = 2
TASK_RUNTIME = "30m"

# Level 2 resources (summary aggregation)
SUMMARY_MEMORY = "4G"
SUMMARY_CORES = 2
SUMMARY_RUNTIME = "15m"

MAX_CONCURRENT_TASKS = 500

# Test mode - set to N to only process first N tasks (0 = all tasks)
TEST_RUN = 0

# Level control - set to skip levels
SKIP_LEVEL_1 = False  # Set to True to only run Level 2 (summary aggregation)
SKIP_LEVEL_2 = False  # Set to True to only run Level 1 (draw processing)

# Admin level (0 = country, 1 = state/province, 2 = district)
ADMIN_LEVEL = 0

# Use simplified shapefiles (faster, smaller)
USE_SIMPLIFIED = True

# Save storm-level data (Level 1/2 - can be large)
SAVE_STORM_LEVEL = False

# Scripts
ANALYSIS_SCRIPT = Path(__file__).parent / "run_climada_storm_analysis.py"
SUMMARY_SCRIPT = Path(__file__).parent / "run_climada_summary.py"

# Task assignments file (created by main orchestrator Level 0)
TASK_ASSIGNMENTS_FILE = rfc.CLIMADA_INPUT_PATH / DATA_SOURCE / "level_4_task_assignments.csv"

# Output directory for summary results
OUTPUT_DIR = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada_admin_summaries")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# VALIDATE INPUTS
# ============================================================================

print("=" * 80)
print("CLIMADA Intensity → Admin Impact Analysis Orchestrator (2-Level)")
print("=" * 80)

if SKIP_LEVEL_1 and SKIP_LEVEL_2:
    print("\n❌ ERROR: Cannot skip both Level 1 and Level 2!")
    sys.exit(1)

if not ANALYSIS_SCRIPT.exists():
    print(f"\n❌ ERROR: Level 1 script not found: {ANALYSIS_SCRIPT}")
    sys.exit(1)

if not SUMMARY_SCRIPT.exists():
    print(f"\n❌ ERROR: Level 2 script not found: {SUMMARY_SCRIPT}")
    sys.exit(1)

if not TASK_ASSIGNMENTS_FILE.exists():
    print(f"\n❌ ERROR: Task assignments file not found: {TASK_ASSIGNMENTS_FILE}")
    print("\nRun the main orchestrator with STARTING_LEVEL=0 to create task assignments:")
    print("  python src/idd_climate_models/01_process_raw_through_climada_input/00_orchestrator.py")
    sys.exit(1)

print(f"\n✓ Level 1 script: {ANALYSIS_SCRIPT}")
print(f"✓ Level 2 script: {SUMMARY_SCRIPT}")
print(f"✓ Task assignments: {TASK_ASSIGNMENTS_FILE}")
print(f"✓ Output directory: {OUTPUT_DIR}")
print(f"✓ Admin level: {ADMIN_LEVEL}")
print(f"✓ Simplified shapefile: {USE_SIMPLIFIED}")
print(f"✓ Save storm-level data: {SAVE_STORM_LEVEL}")

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

# Show breakdown by combination
combo_counts = df_assignments.groupby(['model', 'variant', 'scenario', 'time_period']).size()
print(f"\n  Unique model/variant/scenario/time_period combinations: {len(combo_counts)}")

# Show model/variant/scenario combinations (for Level 2)
level2_combinations = df_assignments[['model', 'variant', 'scenario']].drop_duplicates()
print(f"  Unique model/variant/scenario combinations: {len(level2_combinations)}")

# Apply test mode if enabled
if TEST_RUN > 0:
    print(f"\n⚠️  TEST MODE: Limiting to first {TEST_RUN} tasks")
    task_ids_to_run = list(range(1, min(TEST_RUN + 1, total_tasks + 1)))
    
    # Also limit Level 2 combinations to those with tasks in test set
    tasks_in_test = df_assignments[df_assignments['task_id'].isin(task_ids_to_run)]
    level2_combinations = tasks_in_test[['model', 'variant', 'scenario']].drop_duplicates()
else:
    task_ids_to_run = list(range(1, total_tasks + 1))

print(f"\nWill create {len(task_ids_to_run)} Level 1 tasks")
print(f"Will create {len(level2_combinations)} Level 2 tasks")

if SKIP_LEVEL_1:
    print(f"⚠️  SKIP_LEVEL_1 is True - only running Level 2 (summary aggregation)")
if SKIP_LEVEL_2:
    print(f"⚠️  SKIP_LEVEL_2 is True - only running Level 1 (draw processing)")

# ============================================================================
# CREATE JOBMON WORKFLOW
# ============================================================================

print("\n" + "=" * 80)
print("Setting up Jobmon workflow")
print("=" * 80)

user = getpass.getuser()
wf_uuid = uuid.uuid4()
tool_name = "climada_admin_impact_analysis"

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

# ============================================================================
# CREATE TASK TEMPLATES
# ============================================================================

# Build command template arguments
simplified_flag = "--simplified" if USE_SIMPLIFIED else ""
storm_level_flag = "--save_storm_level" if SAVE_STORM_LEVEL else ""

# Level 1: Process draws (one task per task_id)
level1_template = tool.get_task_template(
    template_name="climada_admin_analysis_level1",
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
        "--output_dir {output_dir} "
        f"--admin_level {ADMIN_LEVEL} "
        f"{simplified_flag} "
        f"{storm_level_flag}"
    ),
    node_args=["data_source", "task_id", "output_dir"],
)

# Level 2: Run summary aggregation (one task per model/variant/scenario)
level2_template = tool.get_task_template(
    template_name="climada_admin_analysis_level2",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": SUMMARY_MEMORY,
        "cores": SUMMARY_CORES,
        "runtime": SUMMARY_RUNTIME,
        "queue": QUEUE,
        "project": PROJECT,
        "constraints": "archive",
    },
    command_template=(
        f"python {SUMMARY_SCRIPT} "
        "--data_source {data_source} "
        "--model {model} "
        "--variant {variant} "
        "--scenario {scenario} "
        f"--admin_level {ADMIN_LEVEL} "
        "--output_dir {output_dir}"
    ),
    node_args=["data_source", "model", "variant", "scenario", "output_dir"],
)

# ============================================================================
# CREATE LEVEL 1 TASKS (Draw Processing)
# ============================================================================

level1_tasks = []
task_id_to_task = {}  # For dependency tracking

if not SKIP_LEVEL_1:
    print("\n" + "=" * 80)
    print("Creating Level 1 tasks (draw processing)")
    print("=" * 80)

    for task_id in task_ids_to_run:
        task = level1_template.create_task(
            data_source=DATA_SOURCE,
            task_id=str(task_id),
            output_dir=str(OUTPUT_DIR),
        )
        level1_tasks.append(task)
        task_id_to_task[task_id] = task

    print(f"\n✓ Created {len(level1_tasks)} Level 1 tasks")
else:
    print("\n⚠️  Skipping Level 1 task creation")

# ============================================================================
# CREATE LEVEL 2 TASKS (Summary Aggregation)
# ============================================================================

level2_tasks = []
dependencies_to_add = []

if not SKIP_LEVEL_2:
    print("\n" + "=" * 80)
    print("Creating Level 2 tasks (summary aggregation)")
    print("=" * 80)

    print(f"\nFound {len(level2_combinations)} unique model/variant/scenario combinations:")

    for _, row in level2_combinations.iterrows():
        model = row['model']
        variant = row['variant']
        scenario = row['scenario']
        
        # Create Level 2 task
        task = level2_template.create_task(
            data_source=DATA_SOURCE,
            model=model,
            variant=variant,
            scenario=scenario,
            output_dir=str(OUTPUT_DIR),
        )
        level2_tasks.append(task)
        
        # Track dependencies only if Level 1 tasks exist
        if not SKIP_LEVEL_1:
            # Find all Level 1 tasks for this combination (across all time_periods and basins)
            matching_task_ids = df_assignments[
                (df_assignments['model'] == model) &
                (df_assignments['variant'] == variant) &
                (df_assignments['scenario'] == scenario)
            ]['task_id'].unique()
            
            # Only include tasks that are in our run set
            matching_task_ids = [tid for tid in matching_task_ids if tid in task_id_to_task]
            
            # Track dependencies: Level 2 task depends on all Level 1 tasks for this combo
            for task_id in matching_task_ids:
                dependencies_to_add.append((task, task_id_to_task[task_id]))
            
            print(f"  {model}/{variant}/{scenario}: depends on {len(matching_task_ids)} Level 1 tasks")
        else:
            print(f"  {model}/{variant}/{scenario}: no dependencies (Level 1 skipped)")

    print(f"\n✓ Created {len(level2_tasks)} Level 2 tasks")
    if dependencies_to_add:
        print(f"✓ Will add {len(dependencies_to_add)} dependencies")
else:
    print("\n⚠️  Skipping Level 2 task creation")

# ============================================================================
# ADD ALL TASKS TO WORKFLOW
# ============================================================================

print("\n" + "=" * 80)
print("Adding tasks to workflow")
print("=" * 80)

all_tasks = level1_tasks + level2_tasks

if not all_tasks:
    print("\n❌ ERROR: No tasks to run!")
    sys.exit(1)

workflow.add_tasks(all_tasks)

print(f"✓ Added {len(all_tasks)} total tasks to workflow")

# ============================================================================
# ADD DEPENDENCIES
# ============================================================================

if dependencies_to_add:
    print("\n" + "=" * 80)
    print("Adding dependencies")
    print("=" * 80)

    for level2_task, level1_task in dependencies_to_add:
        level2_task.add_upstream(level1_task)

    print(f"✓ Added {len(dependencies_to_add)} dependencies using add_upstream")
else:
    print("\n⚠️  No dependencies to add (single level run or Level 1 skipped)")

# ============================================================================
# WORKFLOW SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("Workflow Summary")
print("=" * 80)
print(f"Level 1 tasks (draw processing): {len(level1_tasks)}")
print(f"  Resources: {TASK_MEMORY}, {TASK_CORES} cores, {TASK_RUNTIME}")
print(f"Level 2 tasks (summary aggregation): {len(level2_tasks)}")
print(f"  Resources: {SUMMARY_MEMORY}, {SUMMARY_CORES} cores, {SUMMARY_RUNTIME}")
print(f"Total tasks: {len(all_tasks)}")
print(f"Total dependencies: {len(dependencies_to_add)}")
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
