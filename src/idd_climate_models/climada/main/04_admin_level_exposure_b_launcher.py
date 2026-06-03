import re
import uuid
import pandas as pd  # type: ignore
from jobmon.client.tool import Tool  # type: ignore
from pathlib import Path


SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4b_v2/")


def gather_completed_tasks(meta_df: pd.DataFrame) -> set:
    """
    Walk the stage 4B output tree once and return a set of 8-tuples
    `(source_id, variant_label, experiment_id, batch_year, basin, draw,
    storm_id, location_id)` representing completed per-(storm, loc) parquets.

    Stage 4B's save function writes each task's output as
    `storm_<storm_id>_loc_<location_id>_<basin>_..._<draw_text>.parquet`
    under `tc_risk_draw_<draw>/storm_exposure/<year>/`. The year subdir is
    derived from the storm's start_date inside the worker, so the launcher
    doesn't know it up-front — we glob across year subdirs and parse
    storm_id + location_id out of the filename.

    Files smaller than 1 KB are treated as incomplete (the main script's
    save uses `to_parquet` which would otherwise produce a header-only
    file on a mid-write crash).
    """
    completed: set = set()
    combo_cols = [
        "source_id", "variant_label", "experiment_id",
        "batch_year", "basin", "draw",
    ]
    filename_re = re.compile(r"^storm_(\d+)_loc_(\d+)_")

    for combo in meta_df[combo_cols].drop_duplicates().itertuples(index=False):
        storm_exposure_dir = (
            SAVE_ROOT
            / combo.source_id / combo.variant_label / combo.experiment_id
            / combo.batch_year / combo.basin
            / f"tc_risk_draw_{int(combo.draw)}" / "storm_exposure"
        )
        if not storm_exposure_dir.exists():
            continue
        for parquet in storm_exposure_dir.rglob("storm_*_loc_*.parquet"):
            if parquet.stat().st_size < 1024:
                continue
            m = filename_re.match(parquet.name)
            if not m:
                continue
            completed.add((
                combo.source_id, combo.variant_label, combo.experiment_id,
                combo.batch_year, combo.basin, int(combo.draw),
                int(m.group(1)), int(m.group(2)),
            ))
    return completed


# Read in paths
meta_df = pd.read_parquet("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4a_metadata_admin0/resource_estimation_all_storms.parquet")


# replace nan basin with "NA"
meta_df["basin"] = meta_df["basin"].fillna("NA")

# assign single num_core
meta_df["num_cores"] = 1

# Format memory + runtime as SLURM-friendly strings. (Vectorized; was a
# per-row apply lambda.)
meta_df["memory_gb"] = meta_df["memory_gb"].astype(int).astype(str) + "G"
meta_df["runtime_min_binned"] = meta_df["runtime_min_binned"].astype(int).astype(str) + "m"


############################################################################################
# Completion derivation (filesystem-based, not Jobmon-based).
#
# A stage-4B task writes one per-(storm, location) parquet. We pre-walk the
# output tree once per unique (source/variant/exp/batch_year/basin/draw)
# combo, build a set of done (storm_id, location_id, ...) keys, then filter
# meta_df with a single set-membership pass. Vastly cheaper than per-row
# globs at the 100K+-task scale of this stage.
_completed_keys = gather_completed_tasks(meta_df)
_meta_keys = list(zip(
    meta_df["source_id"],
    meta_df["variant_label"],
    meta_df["experiment_id"],
    meta_df["batch_year"],
    meta_df["basin"],
    meta_df["draw"].astype(int),
    meta_df["storm_id"].astype(int),
    meta_df["location_id"].astype(int),
))
_completed_mask = pd.Series([k in _completed_keys for k in _meta_keys], index=meta_df.index)
print(
    f"Completion scan: {_completed_mask.sum()} / {len(meta_df)} tasks "
    f"already done on disk; {(~_completed_mask).sum()} tasks to submit."
)
meta_df = meta_df[~_completed_mask].copy()


#######################################################################
project = "proj_rapidresponse"
wf_uuid = uuid.uuid4()

# Paths to the worker scripts (resolved relative to this launcher so the
# repo can be moved without breaking the command).
MAIN_SCRIPT = Path(__file__).resolve().parent / "04_admin_level_exposure_b_main.py"
POST_PROCESSING_SCRIPT = (
    Path(__file__).resolve().parent / "04_admin_level_exposure_b_post_processing.py"
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
unique_configs = meta_df[['runtime_min_binned', 'num_cores', 'memory_gb']].drop_duplicates()

# Create task templates for each unique configuration
task_templates = {}
for _, config in unique_configs.iterrows():
    config_key = f"{config['runtime_min_binned']}_{config['num_cores']}_{config['memory_gb']}"
    
    task_templates[config_key] = tool.get_task_template(
        template_name=f"CLIMADA_stage4_{config_key}",
        default_cluster_name="slurm",
        default_compute_resources={
            "queue": "all.q",
            "cores": config['num_cores'],
            "memory": config['memory_gb'],
            "runtime": config['runtime_min_binned'],
            "project": project,
        },
        default_resource_scales={
            "memory": lambda x: int(x*2),  # scale memory by 100%
            "runtime": lambda x: int(x*1.5),  # scale runtime by 50%
        },
        max_attempts=3,
        command_template=(
            f"python {MAIN_SCRIPT} "
            "--source_id {source_id} "
            "--variant_label {variant_label} "
            "--experiment_id {experiment_id} "
            "--batch_year {batch_year} "
            "--basin {basin} "
            "--draw {draw} "
            "--storm_id {storm_id} "
            "--location_id {location_id} "
            "--num_cores {num_cores}"
        ),
        node_args=["source_id", "variant_label", "experiment_id", "batch_year", "basin", "draw", "storm_id", "location_id", "num_cores"],
    )


# Create tasks using the appropriate template
tasks = []
for row in meta_df.itertuples():
    config_key = f"{row.runtime_min_binned}_{row.num_cores}_{row.memory_gb}"
    template = task_templates[config_key]

    task = template.create_task(
        name=(
            f"CLIMADA_stage4_"
            f"{row.source_id}_"
            f"{row.variant_label}_"
            f"{row.experiment_id}_"
            f"{row.batch_year}_"
            f"{row.basin}_"
            f"{row.draw}_"
            f"{row.storm_id}_"
            f"{row.location_id}_"
            f"rt{row.runtime_min_binned}_"
            f"mem{row.memory_gb}_"
            f"cores{row.num_cores}_"

        ),
        source_id=row.source_id,
        variant_label=row.variant_label,
        experiment_id=row.experiment_id,
        batch_year=row.batch_year,
        basin=row.basin,
        draw=row.draw,
        storm_id=row.storm_id,
        location_id=row.location_id,
        num_cores=row.num_cores,
    )

    tasks.append(task)

print(f"Number of tasks: {len(tasks)}")
print(f"Number of task templates created: {len(task_templates)}")


# Post-processing downstream task: a single task that builds the expected
# 4B output paths from 4A's compiled metadata, parallel-reads every
# per-(storm × loc) parquet, and writes one consolidated parquet
# (stage4b_v2/_consolidated/storm_exposure_all.parquet). Depends on every
# per-(storm × loc) task above so it only fires once 4B is done. When
# resume filters `tasks` to empty, this still runs (upstream_tasks=[] is
# just "no dependencies").
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
        "memory": lambda x: int(x * 1.5),
        "runtime": lambda x: int(x * 2),
    },
    max_attempts=2,
    command_template=f"python {POST_PROCESSING_SCRIPT}",
    node_args=[],
)

post_processing_task = post_processing_template.create_task(
    name="CLIMADA_stage4b_post_processing",
    upstream_tasks=tasks,
)


if tasks:
    workflow.add_tasks(tasks)
    print(f"✅ {len(tasks)} per-(storm × loc) tasks added to workflow.")
else:
    print(
        "ℹ️ No per-(storm × loc) tasks to submit (all complete on disk); "
        "post-processing will fire immediately."
    )

workflow.add_tasks([post_processing_task])
print("✅ Post-processing downstream task added to workflow.")

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
