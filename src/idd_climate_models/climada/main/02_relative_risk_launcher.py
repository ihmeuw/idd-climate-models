import argparse
import uuid
from concurrent.futures import ThreadPoolExecutor
import pandas as pd  # type: ignore
from jobmon.client.tool import Tool  # type: ignore
from pathlib import Path

from stage1_run_config import require_stage1_complete

parser = argparse.ArgumentParser(description="Launch CLIMADA stage 2 (one task per combo)")
parser.add_argument("--pilot", type=int, default=0,
                    help="submit only ~N combos, spread over basins and models, to check "
                         "resources; no count tasks")
# One core and one worker per task: with more workers each draw's per-job
# sums (up to ~360 MiB for a 58-job EP combo) are pickled back to the parent,
# which was far slower than running the draws in one process.
parser.add_argument("--cores", type=int, default=1, help="SLURM cores per task")
parser.add_argument("--workers", type=int, default=None,
                    help="parallel stage 1 draws per task (default: --cores)")
parser.add_argument("--memory_gb", type=int, default=4, help="memory per task (x1.5 on retry)")
parser.add_argument("--runtime_min", type=int, default=60, help="runtime per task (x1.5 on retry)")
parser.add_argument("--max_concurrent", type=int, default=300,
                    help="max tasks running at once (caps the load on the shared filesystem)")
parser.add_argument("--combo", action="append", default=[],
                    help="only this combo, as SOURCE/VARIANT/EXPERIMENT/BATCH_YEAR/BASIN "
                         "(repeatable); skips the completion scan of the others")
parser.add_argument("--dry_run", action="store_true",
                    help="build the task list and print it; do not bind or submit anything")
args = parser.parse_args()
WORKERS = args.workers or args.cores

# One task per combo (source_id, variant_label, experiment_id, batch_year,
# basin): it reads each stage 1 storm once and writes the rasters of every
# storm draw x cause of that combo. Before 2026-10-13 there was one task per
# (storm draw, cause), each re-reading all 100 stage 1 draws of its combo
# (~25 reads of every storm), which flooded the filesystem with lookups.

RELATIVE_RISKS = ["indirect_resp_draw", "indirect_cvd_draw"]

ROOT_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/")
SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage2_v3/")


def _stage2_paf_path(
    storm_draw: str,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    relative_risk: str,
    sample_name: str,
    year: int,
) -> Path:
    """Path to a single year's raw_paf GeoTIFF. Must match the format
    written by `save_raster` in 02_relative_risk_main.py."""
    start_year, end_year = batch_year.split("-")
    return (
        SAVE_ROOT
        / storm_draw / source_id / variant_label / experiment_id
        / batch_year / str(year) / basin / "raw_paf"
        / (
            f"draw_mean_raw_paf_{storm_draw}_{relative_risk}_{sample_name}_"
            f"{basin}_{source_id}_{experiment_id}_{variant_label}_"
            f"{start_year}01_{end_year}12_{year}.tif"
        )
    )


def combo_is_complete(rows: pd.DataFrame) -> bool:
    """A combo is complete iff every (storm draw, cause) row has a raw_paf
    GeoTIFF of at least 1 KB for every year in batch_year. Stops at the first
    missing file. Skips the load-validity check (the main script's per-job
    resume logic catches and rebuilds corrupt files)."""
    for row in rows.itertuples(index=False):
        start_year, end_year = map(int, row.batch_year.split("-"))
        for year in range(start_year, end_year + 1):
            path = _stage2_paf_path(
                storm_draw=row.storm_draw,
                source_id=row.source_id,
                variant_label=row.variant_label,
                experiment_id=row.experiment_id,
                batch_year=row.batch_year,
                basin=row.basin,
                relative_risk=row.relative_risk,
                sample_name=row.sample_name,
                year=year,
            )
            if not path.exists() or path.stat().st_size < 1024:
                return False
    return True

# Read in paths
meta_df = pd.read_csv("/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/level_4_task_assignments.csv")
meta_df = meta_df.drop(columns=["task_id", "draw"]).drop_duplicates()
# replace nan with NA
meta_df = meta_df.fillna("NA")

# Normalize column names
meta_df = meta_df.rename(columns={
    "model": "source_id",
    "variant": "variant_label",
    "scenario": "experiment_id",
    "time_period": "batch_year",
})

# Exclude the historical-baseline batch (1965-1969). Stage 2 runs on the
# projection batches only; the baseline outputs come from a separate workflow.
meta_df = meta_df[meta_df["batch_year"] != "1965-1969"]

# read in storm draws
storm_draw_df = pd.read_csv("/mnt/team/rapidresponse/pub/tropical-storms/storm_draw_table.csv")

complete_df = meta_df.merge(
    storm_draw_df, 
    on=["source_id", "variant_label"],
    how="inner",
)

# replace storm_draw as storm_draw_XXXX
complete_df["storm_draw"] = complete_df["storm_draw"].apply(lambda x: f"storm_draw_{x:04d}")

# Ordered list of high-priority storm_draws. These are submitted first when
# PRIORITY_MODE == "priority"; submitted at the end of the queue when
# PRIORITY_MODE == "non_priority"; and shipped in priority order when
# PRIORITY_MODE == "all".
PRIORITY_DRAWS = [
    "storm_draw_0002",
    "storm_draw_0004",
    "storm_draw_0005",
    "storm_draw_0007",
    "storm_draw_0008",
    "storm_draw_0003",
    "storm_draw_0006",
    "storm_draw_0001",
]

# Which subset of storm_draws to submit. Change to flip the workflow.
#   "non_priority" — submit everything EXCEPT PRIORITY_DRAWS (default state)
#   "priority"     — submit ONLY PRIORITY_DRAWS (for testing / smoke runs)
#   "all"          — submit every storm_draw, priority ones first
PRIORITY_MODE = "all"

priority_map = {v: i for i, v in enumerate(PRIORITY_DRAWS)}

df = complete_df.copy()
df["draw_order"] = df["storm_draw"].map(priority_map).fillna(len(PRIORITY_DRAWS))
df = df.sort_values(
    ["draw_order", "storm_draw", "source_id", "variant_label",
     "experiment_id", "batch_year", "basin"]
).drop(columns="draw_order")

if PRIORITY_MODE == "non_priority":
    df = df[~df["storm_draw"].isin(PRIORITY_DRAWS)]
elif PRIORITY_MODE == "priority":
    df = df[df["storm_draw"].isin(PRIORITY_DRAWS)]
elif PRIORITY_MODE == "all":
    pass  # keep everything, priority-ordered
else:
    raise ValueError(
        f"Invalid PRIORITY_MODE={PRIORITY_MODE!r}; "
        f"expected one of: 'non_priority', 'priority', 'all'."
    )

#########################################################################################
# Fan out (indirect_resp_draw, indirect_cvd_draw): one row per (storm draw, cause).
rr_cols = ["indirect_cvd_draw", "indirect_resp_draw"]
COMBO_COLS = ["source_id", "variant_label", "experiment_id", "batch_year", "basin"]

final_long = df.melt(
    id_vars=COMBO_COLS + ["storm_draw"],
    value_vars=rr_cols,
    var_name="relative_risk",
    value_name="sample_name",
)

############################################################################################
# Completion derivation (filesystem-based, not Jobmon-based): submit a combo
# if any of its (storm draw, cause, year) rasters is missing. The worker only
# computes the missing ones (get_year_status per job).
if args.combo:
    # Repair runs: check and submit only the named combos.
    wanted = {tuple(c.split("/")) for c in args.combo}
    bad = [c for c in wanted if len(c) != len(COMBO_COLS)]
    if bad:
        raise SystemExit(f"--combo must be SOURCE/VARIANT/EXPERIMENT/BATCH_YEAR/BASIN, got {bad}")
    keys = final_long[COMBO_COLS].apply(tuple, axis=1)
    unknown = wanted - set(keys)
    if unknown:
        raise SystemExit(f"--combo not in the task list: {sorted(unknown)}")
    final_long = final_long[keys.isin(wanted)]

combos = final_long.groupby(COMBO_COLS, sort=False)
# Each check is one NFS round trip (~15 ms), and a complete combo checks all
# of its rasters (up to 290), so the full scan runs in threads.
with ThreadPoolExecutor(max_workers=32) as pool:
    done_flags = list(pool.map(lambda kv: combo_is_complete(kv[1]), combos))
incomplete = [key for (key, _), done in zip(combos, done_flags) if not done]
remaining = pd.DataFrame(incomplete, columns=COMBO_COLS)
print(f"Completion scan: {combos.ngroups - len(remaining)} / {combos.ngroups} combos "
      f"already done on disk; {len(remaining)} combos to submit.")

if args.pilot:
    # A resource check: about N combos spread over basins and models (the
    # models differ in storm draws per combo: 1-29).
    per_basin = max(1, -(-args.pilot // remaining["basin"].nunique()))
    remaining = (
        remaining.groupby("basin", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_basin), random_state=0))
        .reset_index(drop=True)
    )
    print(f"PILOT: {len(remaining)} combos (by basin: {remaining['basin'].value_counts().to_dict()}; "
          f"by model: {remaining['source_id'].value_counts().to_dict()})")

# Each task reads all 100 stage 1 draws of its combo (N_DRAWS in the main),
# and a missing stage 1 store reads as "no storms", so refuse to submit until
# stage 1 has finished every one of them.
require_stage1_complete({
    (*combo, draw)
    for combo in remaining[COMBO_COLS].itertuples(index=False, name=None)
    for draw in range(100)
})


############################################################################################

project = "proj_rapidresponse"
wf_uuid = uuid.uuid4()

# Path to the worker script (resolved relative to this launcher so the repo
# can be moved without breaking the command).
MAIN_SCRIPT = Path(__file__).resolve().parent / "02_relative_risk_main.py"

tool = Tool(name="CLIMADA_stage2")

workflow = tool.create_workflow(
    name=f"CLIMADA_stage2_combo_{'pilot_' if args.pilot else ''}{wf_uuid}",
    max_concurrently_running=args.max_concurrent,
)

workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={
        "memory": "10G",
        "cores": 1,
        "runtime": "60m",
        "constraints": "archive",
        "queue": "all.q",
        "project": project,  # Ensure the project is set correctly
    }
)

task_template = tool.get_task_template(
    template_name=f"CLIMADA_stage2_combo_{args.runtime_min}m_{WORKERS}w_{args.cores}c_{args.memory_gb}G",
    default_cluster_name="slurm",
    default_compute_resources={
        "queue": "all.q",
        "cores": args.cores,
        "memory": f"{args.memory_gb}G",
        "runtime": f"{args.runtime_min}m",
        "project": project,
    },
    default_resource_scales={
        "memory": 1.5,
        "runtime": 1.5,
    },
    command_template=(
        f"python {MAIN_SCRIPT} "
        "--source_id {source_id} "
        "--variant_label {variant_label} "
        "--experiment_id {experiment_id} "
        "--batch_year {batch_year} "
        "--basin {basin} "
        "--num_cores {num_cores}"
    ),
    node_args=["source_id", "variant_label", "experiment_id", "batch_year", "basin", "num_cores"],
    task_args=[],
    op_args=[],
)

tasks = [
    task_template.create_task(
        name=(
            f"CLIMADA_stage2_{row.source_id}_{row.variant_label}_{row.experiment_id}_"
            f"{row.batch_year}_{row.basin}"
        ),
        source_id=row.source_id,
        variant_label=row.variant_label,
        experiment_id=row.experiment_id,
        batch_year=row.batch_year,
        basin=row.basin,
        num_cores=WORKERS,
    )
    for row in remaining.itertuples(index=False)
]

print(f"Number of tasks: {len(tasks)} ({args.cores} cores, {WORKERS} workers, "
      f"{args.memory_gb}G, {args.runtime_min} min, at most {args.max_concurrent} at once)")

if tasks:
    workflow.add_tasks(tasks)
    print("✅ Tasks successfully added to workflow.")
else:
    print("⚠️ No tasks added to workflow. Check task generation.")

############################################################################################
# Downstream: the Stage 3 count parquets (storm_draw_admin{0,1}_count_v3).
# Each generator reads the first-year CVD rasters of 8 representative storm
# draws plus the draw-0 Stage 1 stores, so it runs once every Stage 2 task in
# this workflow has succeeded (immediately if none were submitted). Added
# only when its parquet is missing (delete it to regenerate), and only for
# PRIORITY_MODE "all": the representative draws are not all priority draws.
PREP_DIR = Path(__file__).resolve().parent.parent / "prep"
COUNT_JOBS = {
    level: (
        PREP_DIR / f"generate_storm_draw_admin{level}_count.py",
        Path(
            "/mnt/team/rapidresponse/pub/tropical-storms/climada/output/"
            f"storm_draw_admin{level}_count_v3.parquet"
        ),
    )
    for level in (0, 1)
}

count_template = tool.get_task_template(
    template_name="CLIMADA_stage2_admin_counts",
    default_cluster_name="slurm",
    default_compute_resources={
        "queue": "all.q",
        "cores": 64,  # the generators run_parallel over 64 cores
        "memory": "64G",
        "runtime": "120m",
        "project": project,
    },
    default_resource_scales={
        "memory": 1.5,
        "runtime": 1.5,
    },
    max_attempts=2,
    command_template="python {count_script}",
    node_args=["count_script"],
)

count_tasks = []
for level, (count_script, count_parquet) in COUNT_JOBS.items():
    if args.pilot:
        print(f"PILOT: not adding the admin{level} count task.")
    elif PRIORITY_MODE != "all":
        print(f"⚠️ PRIORITY_MODE={PRIORITY_MODE!r}: not adding the admin{level} count task.")
    elif count_parquet.exists():
        print(f"Admin{level} counts already exist at {count_parquet}; not regenerating.")
    else:
        count_tasks.append(count_template.create_task(
            name=f"CLIMADA_stage2_admin{level}_counts",
            upstream_tasks=tasks,
            count_script=str(count_script),
        ))

if count_tasks:
    workflow.add_tasks(count_tasks)
    print(f"✅ {len(count_tasks)} Stage 3 count task(s) added downstream of the Stage 2 tasks.")

if args.dry_run:
    print("DRY RUN: not binding or submitting. Combos that would run:")
    print(remaining.groupby(["basin", "source_id"]).size().unstack(fill_value=0).to_string())
    raise SystemExit(0)

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
    # 5 days; Jobmon's default (10 h) stopped the first stage 1 run mid-way.
    status = workflow.run(seconds_until_timeout=5 * 24 * 3600)
    print(f"Workflow {workflow.workflow_id} completed with status {status}.")
except Exception as e:
    print(f"❌ Workflow submission failed: {e}")
