import uuid
import numpy as np
import pandas as pd  # type: ignore
from jobmon.client.tool import Tool  # type: ignore
from pathlib import Path


SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4b_v2/")
MAX_GROUP_RUNTIME_MIN = 360 # 6 hours per group is a reasonable balance of SLURM efficiency vs. risk of mid-run failures.
GROUPED_PARQUET_DIR = Path(
    "/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4a_metadata_admin0"
    "/_workflow_grouped_tasks_4b"
)


def assign_groups(
    meta_df: pd.DataFrame, max_runtime_min: int = MAX_GROUP_RUNTIME_MIN
) -> pd.DataFrame:
    """
    Greedy bin-pack tasks into groups whose total serial runtime ≤ max_runtime_min
    minutes. Tasks are sorted by memory_gb desc then runtime_min_binned desc so
    that tasks with the same resource class cluster together — all tasks in a
    group share the same memory_gb, avoiding memory over-allocation on the SLURM
    job.

    Each memory_gb value maps to a single runtime_min_binned, so grouping within
    a class reduces to consecutive batching. Fully vectorised — no Python row loop.

    Returns the same DataFrame (reordered) with a new integer `group_id` column.
    """
    df = meta_df.sort_values(
        ["memory_gb", "runtime_min_binned"], ascending=[False, False]
    ).reset_index(drop=True)

    # 0-based rank of each task within its memory class.
    df["_rank"] = df.groupby("memory_gb").cumcount()
    # Tasks that fit in one group for this memory class.
    df["_tpg"] = max_runtime_min // df["runtime_min_binned"]
    # Local group ID within the memory class.
    df["_local_group"] = df["_rank"] // df["_tpg"]

    # Global offset: cumulative group count of all higher-memory classes
    # (they appear first in the sort order).
    class_n_groups = (
        df.groupby("memory_gb")["_local_group"]
        .max()
        .add(1)
        .sort_index(ascending=False)
    )
    class_offset = class_n_groups.cumsum().shift(1, fill_value=0).astype(int)
    df["group_id"] = (df["_local_group"] + df["memory_gb"].map(class_offset)).astype(int)
    df = df.drop(columns=["_rank", "_tpg", "_local_group"])
    return df


# Read resource parquet — keep integer columns numeric for grouping.
meta_df = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/tropical-storms/climada/output/"
    "stage4a_metadata_admin0/resource_estimation_all_storms.parquet"
)

meta_df["basin"] = meta_df["basin"].fillna("NA")
meta_df["num_cores"] = 1

print(f"Total tasks to group: {len(meta_df):,}")

#######################################################################
project = "proj_rapidresponse"
wf_uuid = uuid.uuid4()

MAIN_SCRIPT = Path(__file__).resolve().parent / "04_admin_level_exposure_b_main.py"
POST_PROCESSING_SCRIPT = (
    Path(__file__).resolve().parent / "04_admin_level_exposure_b_post_processing.py"
)

tool = Tool(name="CLIMADA_stage4")

workflow = tool.create_workflow(
    name=f"CLIMADA_stage4_{wf_uuid}",
)

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
# No launcher-level completion scan: each worker's main() checks _is_task_complete
# per row and skips already-done tasks. This avoids an O(n_files) NFS walk that
# becomes prohibitively slow once millions of output files exist.
meta_df = assign_groups(meta_df, MAX_GROUP_RUNTIME_MIN)

GROUPED_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
grouped_parquet_path = GROUPED_PARQUET_DIR / f"grouped_tasks_{wf_uuid}.parquet"
meta_df.to_parquet(grouped_parquet_path, index=False)
print(f"✅ Grouped task list saved → {grouped_parquet_path}")

# Compute per-group SLURM resources.
# All tasks in a group share the same memory_gb (by construction).
# Round group runtime up to the next 10-minute slot, capped at the budget.
group_meta = (
    meta_df
    .groupby("group_id", as_index=False)
    .agg(
        memory_gb=("memory_gb", "max"),
        total_runtime_min=("runtime_min_binned", "sum"),
        n_tasks=("storm_id", "count"),
    )
)
group_meta["runtime_slot_min"] = (
    (np.ceil(group_meta["total_runtime_min"] / 10) * 10)
    .astype(int)
    .clip(upper=MAX_GROUP_RUNTIME_MIN)
)
group_meta["memory_str"] = group_meta["memory_gb"].astype(str) + "G"
group_meta["runtime_str"] = group_meta["runtime_slot_min"].astype(str) + "m"

print(
    f"\nGroup summary: {len(group_meta):,} groups from {len(meta_df):,} tasks "
    f"(avg {len(meta_df) / len(group_meta):.1f} tasks/group)"
)
print("\n=== Tasks per group runtime slot ===")
print(
    group_meta.groupby("runtime_str")["n_tasks"]
    .agg(["count", "sum"])
    .rename(columns={"count": "n_groups", "sum": "n_tasks"})
    .sort_index()
)

# One Jobmon task template per unique (memory, runtime) resource combination.
unique_configs = group_meta[["runtime_str", "memory_str"]].drop_duplicates()
tasks = []
task_templates: dict = {}
for _, config in unique_configs.iterrows():
    config_key = f"{config['runtime_str']}_1_{config['memory_str']}"
    task_templates[config_key] = tool.get_task_template(
        template_name=f"CLIMADA_stage4_{config_key}",
        default_cluster_name="slurm",
        default_compute_resources={
            "queue": "all.q",
            "cores": 1,
            "memory": config["memory_str"],
            "runtime": config["runtime_str"],
            "project": project,
        },
        default_resource_scales={
            "memory": 2.0,
            "runtime": 1.5,
        },
        max_attempts=3,
        command_template=(
            f"python {MAIN_SCRIPT} "
            f"--grouped_tasks_parquet {grouped_parquet_path} "
            "--group_id {group_id}"
        ),
        node_args=["group_id"],
    )

# One Jobmon task per group.
for _, grp in group_meta.iterrows():
    config_key = f"{grp['runtime_str']}_1_{grp['memory_str']}"
    task = task_templates[config_key].create_task(
        name=f"CLIMADA_stage4_group_{int(grp['group_id'])}",
        group_id=int(grp["group_id"]),
    )
    tasks.append(task)

print(f"\nNumber of group tasks: {len(tasks)}")
print(f"Number of task templates created: {len(task_templates)}")


# Post-processing downstream task: a single task that builds the expected
# 4B output paths from 4A's compiled metadata, parallel-reads every
# per-(storm × loc) parquet, and writes one consolidated parquet.
# Depends on every per-group task so it only fires once 4B is done.
post_processing_template = tool.get_task_template(
    template_name="CLIMADA_stage4b_post_processing",
    default_cluster_name="slurm",
    default_compute_resources={
        "queue": "all.q",
        "cores": 20,
        "memory": "100G",
        "runtime": "240m",
        "project": project,
    },
    default_resource_scales={
        "memory": 1.5,
        "runtime": 2.0,
    },
    max_attempts=2,
    command_template=f"python {POST_PROCESSING_SCRIPT}",
    node_args=[],
)

post_processing_task = post_processing_template.create_task(
    name="CLIMADA_stage4b_post_processing",
    upstream_tasks=tasks,
)

workflow.add_tasks(tasks)
workflow.add_tasks([post_processing_task])
print(f"✅ {len(tasks)} group tasks + post-processing task added to workflow.")

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
