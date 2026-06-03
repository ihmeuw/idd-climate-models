import uuid
import pandas as pd  # type: ignore
from jobmon.client.tool import Tool  # type: ignore
from pathlib import Path



ADMIN_LEVEL = 0
# Draw batches of 1 for testing
DRAW_BATCHES = [f"{i}-{i}" for i in range(100)]

SAVE_ROOT = Path(f"/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4a_metadata_admin{ADMIN_LEVEL}")


def _stage4a_draw_metadata_path(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
) -> Path:
    """Path to a single draw's admin-level metadata parquet. Must match the
    format written by `save_draw_dataframe` in 04_admin_level_exposure_a_main.py."""
    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    start_year, end_year = batch_year.split("-")
    return (
        SAVE_ROOT
        / source_id / variant_label / experiment_id / batch_year / basin
        / f"tc_risk_draw_{draw}" / "admin_level_metadata"
        / (
            f"admin_level_metadata_{basin}_{source_id}_{variant_label}_"
            f"{experiment_id}_{start_year}01_{end_year}12{draw_text}.parquet"
        )
    )


def task_is_complete(row) -> bool:
    """A stage-4A task is complete iff every draw in its draw_batch has an
    admin-level metadata parquet on disk that's at least 1 KB. Mirrors the
    existence + size check in `check_if_draw_is_complete`; skips the parquet
    header validity test (the main script's resume logic will catch and
    rebuild invalid files at runtime)."""
    start_draw, end_draw = map(int, row["draw_batch"].split("-"))
    for draw in range(start_draw, end_draw + 1):
        path = _stage4a_draw_metadata_path(
            source_id=row["source_id"],
            variant_label=row["variant_label"],
            experiment_id=row["experiment_id"],
            batch_year=row["batch_year"],
            basin=row["basin"],
            draw=draw,
        )
        if not path.exists() or path.stat().st_size < 1024:
            return False
    return True


# Read in paths
meta_df = pd.read_csv("/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/level_4_task_assignments.csv")
meta_df = meta_df.drop(columns=["task_id", "draw"]).drop_duplicates()


# replace nan basin with "NA"
meta_df["basin"] = meta_df["basin"].fillna("NA")

# Normalize column names
meta_df = meta_df.rename(columns={
    "model": "source_id",
    "variant": "variant_label",
    "scenario": "experiment_id",
    "time_period": "batch_year",
})

# drop 1965-1969 batch
meta_df = meta_df[meta_df["batch_year"] != "1965-1969"].reset_index(drop=True)

# read in resource requirements df
resource_df = pd.read_parquet("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4_resource_requirements.parquet")
resource_df = resource_df.drop(columns=["year", "num_admin0_first_year", "num_years_in_batch", "estimated_admin0_total", "draw_batch"], errors="ignore")


meta_df = meta_df.merge(resource_df, on=["source_id", "variant_label", "experiment_id", "batch_year", "basin"], how="left")

# fill any missing required run time with 5.0 and missing memory with 16.0
meta_df["req_runtime_min"] = meta_df["req_runtime_min"].fillna(5.0)
meta_df["req_mem_gb_rounded"] = meta_df["req_mem_gb_rounded"].fillna(16.0)

meta_df["num_cores"] = 1

# Memory: bump the resource_df estimate by 25% for headroom, then format
# as a SLURM-friendly "{int}G" string. (Vectorized; was a row-wise apply.)
meta_df["memory_req"] = (meta_df["req_mem_gb_rounded"] * 1.25).astype(int).astype(str) + "G"

full_tasks = (meta_df
    .assign(key=1)
    .merge(pd.DataFrame({"draw_batch": DRAW_BATCHES, "key": 1}), on="key")
    .drop(columns=["key"])
)


# Production overrides — single-draw batches (above) plus these per-task
# resource constants. Stage 4A is the metadata-estimation pass; the heavy
# compute that needs the resource_df budget is 4B.
full_tasks["req_runtime_min"] = 3.0
full_tasks["memory_req"] = "2G"

############################################################################################
# Completion derivation (filesystem-based, not Jobmon-based).
#
# A stage-4A task writes one admin_level_metadata parquet per draw in the
# draw_batch (via save_draw_dataframe). A task is complete iff every draw
# in its batch has the parquet on disk and ≥ 1 KB. We mirror that check
# here per row and submit only rows whose batch isn't already fully written.
completed_mask = full_tasks.apply(task_is_complete, axis=1)
remaining_meta = full_tasks[~completed_mask].copy()
print(
    f"Completion scan: {completed_mask.sum()} / {len(full_tasks)} tasks "
    f"already done on disk; {len(remaining_meta)} tasks to submit."
)
full_tasks = remaining_meta


#######################################################################
project = "proj_rapidresponse"
wf_uuid = uuid.uuid4()

# Paths to the worker scripts (resolved relative to this launcher so the
# repo can be moved without breaking the command).
MAIN_SCRIPT = Path(__file__).resolve().parent / "04_admin_level_exposure_a_main.py"
RESOURCE_ASSIGNMENT_SCRIPT = (
    Path(__file__).resolve().parent / "04_admin_level_exposure_a_resource_assignment.py"
)

tool = Tool(name="CLIMADA_stage4")


# Create a workflow, and set the executor
workflow = tool.create_workflow(
    name=f"CLIMADA_stage4_{wf_uuid}",
    # max_concurrently_running = 100,
)


# Set resources on the workflow
workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={
        "memory": "5G",
        "cores": 2,
        "runtime": "5m",
        "constraints": "archive",
        "queue": "all.q",
        "project": project,  # Ensure the project is set correctly
    }
)


# Get unique combinations of runtime, cores, and memory
unique_configs = full_tasks[['req_runtime_min', 'num_cores', 'memory_req']].drop_duplicates()

# Create task templates for each unique configuration
task_templates = {}
for _, config in unique_configs.iterrows():
    config_key = f"{config['req_runtime_min']}_{config['num_cores']}_{config['memory_req']}"
    
    task_templates[config_key] = tool.get_task_template(
        template_name=f"CLIMADA_stage4_{config_key}",
        default_cluster_name="slurm",
        default_compute_resources={
            "queue": "all.q",
            "cores": config['num_cores'],
            "memory": config['memory_req'],
            "runtime": f"{int(config['req_runtime_min'])}m",
            "project": project,
        },
        default_resource_scales={
            "memory": lambda x: int(x*1.5),  # scale memory by 50%
            "runtime": lambda x: int(x*3),  # scale runtime by 300%
        },
        max_attempts=5,
        command_template=(
            f"python {MAIN_SCRIPT} "
            "--source_id {source_id} "
            "--variant_label {variant_label} "
            "--experiment_id {experiment_id} "
            "--batch_year {batch_year} "
            "--basin {basin} "
            "--draw_batch {draw_batch} "
            "--admin_level {admin_level} "
            "--num_cores {num_cores}"
        ),
        node_args=["source_id", "variant_label", "experiment_id", "batch_year", "basin", "draw_batch", "admin_level", "num_cores"],
    )


# Create tasks using the appropriate template
tasks = []
for row in full_tasks.itertuples():
    config_key = f"{row.req_runtime_min}_{row.num_cores}_{row.memory_req}"
    template = task_templates[config_key]

    task = template.create_task(
        name=(
            f"CLIMADA_stage4_"
            f"src{row.source_id}_"
            f"var{row.variant_label}_"
            f"exp{row.experiment_id}_"
            f"yr{row.batch_year}_"
            f"{row.basin}_"
            f"admin{ADMIN_LEVEL}_"
            f"rt{row.req_runtime_min}m_"
            f"mem{row.memory_req}_"
            f"cores{row.num_cores}_"
            f"db{row.draw_batch}"
        ),
        source_id=row.source_id,
        variant_label=row.variant_label,
        experiment_id=row.experiment_id,
        batch_year=row.batch_year,
        basin=row.basin,
        draw_batch=row.draw_batch,
        admin_level=ADMIN_LEVEL,
        num_cores=row.num_cores,
    )

    tasks.append(task)

print(f"Number of tasks: {len(tasks)}")
print(f"Number of task templates created: {len(task_templates)}")


# Resource-assignment downstream task: a single task that walks the
# completed 4A output tree, concats every (storm × admin) row, bins by
# n_pixels, and writes the launcher-ready parquet 4B consumes. Depends on
# every per-(combo × draw) task above so it only fires once 4A is done.
# When resume filters tasks down to 0, this still runs (upstream_tasks=[]
# is just "no dependencies").
resource_assignment_template = tool.get_task_template(
    template_name="CLIMADA_stage4a_resource_assignment",
    default_cluster_name="slurm",
    default_compute_resources={
        "queue": "all.q",
        "cores": 20,
        "memory": "20G",
        "runtime": "60m",
        "project": project,
    },
    default_resource_scales={
        "memory": lambda x: int(x * 1.5),
        "runtime": lambda x: int(x * 2),
    },
    max_attempts=2,
    command_template=f"python {RESOURCE_ASSIGNMENT_SCRIPT}",
    node_args=[],
)

resource_assignment_task = resource_assignment_template.create_task(
    name="CLIMADA_stage4a_resource_assignment",
    upstream_tasks=tasks,
)


if tasks:
    workflow.add_tasks(tasks)
    print(f"✅ {len(tasks)} per-(combo × draw) tasks added to workflow.")
else:
    print(
        "ℹ️ No per-(combo × draw) tasks to submit (all complete on disk); "
        "resource-assignment will fire immediately."
    )

workflow.add_tasks([resource_assignment_task])
print("✅ Resource-assignment downstream task added to workflow.")

try:
    workflow.bind()
except Exception as e:
    print(f"❌ Workflow binding failed: {e}")
    raise SystemExit(1)

print("✅ Workflow successfully bound.")
print(f"Running workflow with ID {workflow.workflow_id}.")
print("For full information see the Jobmon GUI:")
print(f"https://jobmon-gui.ihme.washington.edu/#/workflow/{workflow.workflow_id}")

try:
    status = workflow.run()
    print(f"Workflow {workflow.workflow_id} completed with status {status}.")
except Exception as e:
    print(f"❌ Workflow submission failed: {e}")
