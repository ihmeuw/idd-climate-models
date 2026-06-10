import uuid
import numpy as np
import pandas as pd  # type: ignore
from jobmon.client.tool import Tool  # type: ignore
from pathlib import Path


ADMIN_LEVEL = 0
DRAW_BATCHES = [f"{i}-{i}" for i in range(100)]

SAVE_ROOT = Path(
    f"/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4a_metadata_admin{ADMIN_LEVEL}"
)

TASK_RUNTIME_MIN = 5
TASK_MEMORY_GB = 5
MAX_GROUP_RUNTIME_MIN = 60
GROUPED_PARQUET_DIR = SAVE_ROOT / "_workflow_grouped_tasks"


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
    """Return True iff the draw's admin-level metadata parquet exists and is ≥ 1 KB."""
    path = _stage4a_draw_metadata_path(
        source_id=row["source_id"],
        variant_label=row["variant_label"],
        experiment_id=row["experiment_id"],
        batch_year=row["batch_year"],
        basin=row["basin"],
        draw=int(row["draw"]),
    )
    return path.exists() and path.stat().st_size >= 1024


def assign_groups(
    meta_df: pd.DataFrame, max_runtime_min: int = MAX_GROUP_RUNTIME_MIN
) -> pd.DataFrame:
    """
    Greedy bin-pack tasks into groups whose total serial runtime ≤ max_runtime_min.
    All stage-4A tasks share the same resource class (TASK_RUNTIME_MIN /
    TASK_MEMORY_GB), so each group is simply a consecutive batch of up to
    max_runtime_min // TASK_RUNTIME_MIN draws.
    """
    tasks_per_group = max_runtime_min // TASK_RUNTIME_MIN
    group_ids = [i // tasks_per_group for i in range(len(meta_df))]
    meta_df = meta_df.copy()
    meta_df["group_id"] = group_ids
    return meta_df


# ---------------------------------------------------------------------------
# Build full task list
# ---------------------------------------------------------------------------
meta_df = pd.read_csv(
    "/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/level_4_task_assignments.csv"
)
meta_df = meta_df.drop(columns=["task_id", "draw"]).drop_duplicates()
meta_df["basin"] = meta_df["basin"].fillna("NA")
meta_df = meta_df.rename(columns={
    "model": "source_id",
    "variant": "variant_label",
    "scenario": "experiment_id",
    "time_period": "batch_year",
})
meta_df = meta_df[meta_df["batch_year"] != "1965-1969"].reset_index(drop=True)

# Cross-join combos × draws and normalise draw to an integer column.
full_tasks = (
    meta_df
    .assign(key=1)
    .merge(pd.DataFrame({"draw_batch": DRAW_BATCHES, "key": 1}), on="key")
    .drop(columns=["key"])
)
full_tasks["draw"] = full_tasks["draw_batch"].str.split("-").str[0].astype(int)
full_tasks = full_tasks.drop(columns=["draw_batch"])

############################################################################################
# Completion scan — filesystem-based, not Jobmon-based.
completed_mask = full_tasks.apply(task_is_complete, axis=1)
remaining_tasks = full_tasks[~completed_mask].copy().reset_index(drop=True)
print(
    f"Completion scan: {completed_mask.sum()} / {len(full_tasks)} tasks "
    f"already done on disk; {len(remaining_tasks)} tasks to submit."
)

#######################################################################
project = "proj_rapidresponse"
wf_uuid = uuid.uuid4()

MAIN_SCRIPT = Path(__file__).resolve().parent / "04_admin_level_exposure_a_main.py"
RESOURCE_ASSIGNMENT_SCRIPT = (
    Path(__file__).resolve().parent / "04_admin_level_exposure_a_resource_assignment.py"
)

tool = Tool(name="CLIMADA_stage4")

workflow = tool.create_workflow(name=f"CLIMADA_stage4_{wf_uuid}")

workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={
        "memory": "5G",
        "cores": 1,
        "runtime": "5m",
        "constraints": "archive",
        "queue": "all.q",
        "project": project,
    }
)

# --- Group assignment and per-group task creation ---
tasks = []

if not remaining_tasks.empty:
    grouped_df = assign_groups(remaining_tasks, MAX_GROUP_RUNTIME_MIN)

    GROUPED_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    grouped_parquet_path = GROUPED_PARQUET_DIR / f"grouped_tasks_{wf_uuid}.parquet"
    grouped_df.to_parquet(grouped_parquet_path, index=False)
    print(f"✅ Grouped task list saved → {grouped_parquet_path}")

    n_groups = grouped_df["group_id"].nunique()
    avg_tasks = len(grouped_df) / n_groups
    print(
        f"\nGroup summary: {n_groups:,} groups from {len(grouped_df):,} tasks "
        f"(avg {avg_tasks:.1f} tasks/group, "
        f"max {MAX_GROUP_RUNTIME_MIN // TASK_RUNTIME_MIN} draws/group)"
    )

    # All stage-4A tasks share one resource class — a single template suffices.
    runtime_str = f"{int(np.ceil(MAX_GROUP_RUNTIME_MIN / 10) * 10)}m"
    memory_str = f"{TASK_MEMORY_GB}G"

    task_template = tool.get_task_template(
        template_name=f"CLIMADA_stage4a_{memory_str}_{runtime_str}",
        default_cluster_name="slurm",
        default_compute_resources={
            "queue": "all.q",
            "cores": 1,
            "memory": memory_str,
            "runtime": runtime_str,
            "project": project,
        },
        default_resource_scales={
            "memory": lambda x: int(x * 1.5),
            "runtime": lambda x: int(x * 3),
        },
        max_attempts=5,
        command_template=(
            f"python {MAIN_SCRIPT} "
            f"--grouped_tasks_parquet {grouped_parquet_path} "
            "--group_id {group_id}"
        ),
        node_args=["group_id"],
    )

    for group_id in range(n_groups):
        task = task_template.create_task(
            name=f"CLIMADA_stage4a_group_{group_id}",
            group_id=group_id,
        )
        tasks.append(task)

    print(f"Number of group tasks: {len(tasks)}")


# Resource-assignment downstream task fires once all per-(combo × draw) groups
# are done. When resume filters tasks to empty it still runs (no upstreams =
# fires immediately).
resource_assignment_template = tool.get_task_template(
    template_name="CLIMADA_stage4a_resource_assignment",
    default_cluster_name="slurm",
    default_compute_resources={
        "queue": "all.q",
        "cores": 40,
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
    print(f"✅ {len(tasks)} group tasks added to workflow.")
else:
    print(
        "ℹ️ No group tasks to submit (all complete on disk); "
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
