import argparse
import uuid
from concurrent.futures import ThreadPoolExecutor
import pandas as pd  # type: ignore
from jobmon.client.tool import Tool  # type: ignore
from pathlib import Path

parser = argparse.ArgumentParser(description="Launch CLIMADA stage 3 (one task per combo)")
parser.add_argument("--admin_level", type=int, required=True, choices=[0, 1])
parser.add_argument("--pilot", type=int, default=0,
                    help="submit only ~N combos, spread over basins and models, to check resources")
parser.add_argument("--combo", action="append", default=[],
                    help="only this combo, as SOURCE/VARIANT/EXPERIMENT/BATCH_YEAR/BASIN (repeatable)")
parser.add_argument("--cores", type=int, default=1, help="SLURM cores per task = years run in parallel")
parser.add_argument("--runtime_factor", type=float, default=1.5,
                    help="safety factor on the estimated (combo, year) runtime (x1.75 on retry)")
parser.add_argument("--max_concurrent", type=int, default=300,
                    help="max tasks running at once (caps the load on the shared filesystem)")
parser.add_argument("--dry_run", action="store_true",
                    help="build the task list and print it; do not bind or submit anything")
args = parser.parse_args()
ADMIN_LEVEL = args.admin_level

RELATIVE_RISKS = ["indirect_resp_draw", "indirect_cvd_draw"]

STORM_DRAW_TABLE_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/storm_draw_table.csv")

if ADMIN_LEVEL == 0:
    SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage3_v3_admin0/")
    META_PARQUET = Path(
        "/mnt/team/rapidresponse/pub/tropical-storms/climada/output/storm_draw_admin0_count_v3.parquet"
    )
    GENERATE_SCRIPT = Path(__file__).resolve().parent.parent / "prep" / "generate_storm_draw_admin0_count.py"
    # basin: (memory GB, per-job runtime min). v2 admin 0 peak memory / longest task
    # (runtime summed over attempts), workflows 584230/584480/584688:
    # AU 37.4G/46m, EP 87.8G/54m (every sampled EP task OOMed at 50G), NA 46.0G/49m,
    # NI 16.5G/33m, SI 23.6G/22m, SP 32.4G/19m, WP 28.3G/44m. Memory is set by the
    # largest 100 m population window, which a combo task holds one at a time,
    # as a per-job task did.
    _BASIN_RESOURCES = {
        "AU": (43, 55), "EP": (101, 65), "NA": (53, 60), "NI": (19, 40),
        "SI": (28, 30), "SP": (38, 25), "WP": (33, 55),
    }
else:
    SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage3_v3_admin1/")
    META_PARQUET = Path(
        "/mnt/team/rapidresponse/pub/tropical-storms/climada/output/storm_draw_admin1_count_v3.parquet"
    )
    GENERATE_SCRIPT = Path(__file__).resolve().parent.parent / "prep" / "generate_storm_draw_admin1_count.py"
    # Same, v2 admin 1 workflows 589680/589842: AU 2.2G/5m, EP 6.2G/24m,
    # NA 3.0G/16m and NI 10.0G/16m (both pinned at their 3G/10G limits),
    # SI 1.1G/0.5m, SP 1.0G/0.4m, WP 6.0G/14m.
    _BASIN_RESOURCES = {
        "AU": (3, 6), "EP": (8, 29), "NA": (4, 20), "NI": (12, 20),
        "SI": (2, 5), "SP": (2, 5), "WP": (7, 17),
    }

UNIQUE_KEYS = ["source_id", "variant_label", "experiment_id", "batch_year", "basin"]

if not META_PARQUET.exists():
    # The generator runs on 64 cores; it normally runs as the downstream task of
    # the stage 2 workflow. Don't start it from wherever this launcher runs.
    raise SystemExit(
        f"Count parquet not found: {META_PARQUET}. It is written by the stage 2 workflow's "
        f"count task ({GENERATE_SCRIPT.name}); run that first."
    )


def _stage3_paf_path(
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
    """Path to a single year's per-admin PAF parquet. Must match the format
    written by `save_batch_paf_dataframe` in 03_admin_level_paf_main.py."""
    start_year, end_year = batch_year.split("-")
    return (
        SAVE_ROOT
        / storm_draw / source_id / variant_label / experiment_id
        / batch_year / basin / str(year) / "paf_df"
        / (
            f"paf_{storm_draw}_{relative_risk}_{sample_name}_{basin}_"
            f"{source_id}_{experiment_id}_{variant_label}_"
            f"{start_year}01_{end_year}12_{year}.parquet"
        )
    )


def combo_year_is_complete(rows: pd.DataFrame, year: int) -> bool:
    """A (combo, year) is complete iff every (storm draw, cause) row has a PAF
    parquet of at least 1 KB or a .no_intersection marker for that year. Stops
    at the first missing file. Skips the parquet validity check (the main
    script's resume logic catches and rebuilds invalid files)."""
    for row in rows.itertuples(index=False):
        path = _stage3_paf_path(
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
        if path.with_suffix(".no_intersection").exists():
            continue  # no admin intersects this year (see the main script)
        if not path.exists() or path.stat().st_size < 1024:
            return False
    return True


# One task per (combo, year), combo = (source_id, variant_label, experiment_id,
# batch_year, basin): it reads each 100 m population window once for all
# storm draw x cause jobs of the combo (they share the stage 1 storms), instead
# of once per job. Before 2026-10-13 there was one task per (storm draw, cause)
# covering all years.
combos_df = pd.read_parquet(META_PARQUET)[UNIQUE_KEYS].drop_duplicates()
storm_draw_df = pd.read_csv(STORM_DRAW_TABLE_PATH)
storm_draw_df["storm_draw"] = storm_draw_df["storm_draw"].apply(lambda x: f"storm_draw_{x:04d}")
final_long = (
    storm_draw_df
    .merge(combos_df, on=["source_id", "variant_label"], how="inner")
    .melt(id_vars=UNIQUE_KEYS + ["storm_draw"], value_vars=RELATIVE_RISKS,
          var_name="relative_risk", value_name="sample_name")
)

if args.combo:
    # Repair runs: check and submit only the named combos.
    wanted = {tuple(c.split("/")) for c in args.combo}
    bad = [c for c in wanted if len(c) != len(UNIQUE_KEYS)]
    if bad:
        raise SystemExit(f"--combo must be SOURCE/VARIANT/EXPERIMENT/BATCH_YEAR/BASIN, got {bad}")
    keys = final_long[UNIQUE_KEYS].apply(tuple, axis=1)
    unknown = wanted - set(keys)
    if unknown:
        raise SystemExit(f"--combo not in the task list: {sorted(unknown)}")
    final_long = final_long[keys.isin(wanted)]

############################################################################################
# Completion derivation (filesystem-based): submit a (combo, year) if any of its
# (storm draw, cause) outputs is missing; the worker computes only those.
# Each check is one NFS round trip (~15 ms), so the scan runs in threads.
combos = final_long.groupby(UNIQUE_KEYS, sort=False)
combo_years = [
    (key, rows, year)
    for key, rows in combos
    for year in range(int(key[3][:4]), int(key[3][5:]) + 1)
]
with ThreadPoolExecutor(max_workers=32) as pool:
    done_flags = list(pool.map(lambda kry: combo_year_is_complete(kry[1], kry[2]), combo_years))
remaining = pd.DataFrame(
    [(*key, year, len(rows)) for (key, rows, year), done in zip(combo_years, done_flags) if not done],
    columns=UNIQUE_KEYS + ["year", "n_jobs"],
)
print(f"Completion scan: {len(combo_years) - len(remaining)} / {len(combo_years)} (combo, year) pairs "
      f"already done on disk; {len(remaining)} to submit ({remaining[UNIQUE_KEYS].drop_duplicates().shape[0]} combos).")

if args.pilot:
    per_basin = max(1, -(-args.pilot // max(remaining["basin"].nunique(), 1)))
    remaining = (
        remaining.groupby("basin", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_basin), random_state=0))
        .reset_index(drop=True)
    )
    print(f"PILOT: {len(remaining)} (combo, year) tasks (by basin: {remaining['basin'].value_counts().to_dict()}; "
          f"by model: {remaining['source_id'].value_counts().to_dict()})")

unknown = set(remaining["basin"]) - set(_BASIN_RESOURCES)
if unknown:
    raise ValueError(f"No stage 3 resources for basin(s) {sorted(unknown)}")
remaining["memory_gb"] = remaining["basin"].map(lambda b: _BASIN_RESOURCES[b][0]).astype(int)
# Runtime of a (combo, year): a shared part (reading and masking the population
# windows, ~2/3 of a v2 per-job year) plus a per-job part (each job's PAF
# resampled to the 100 m grid, ~1/3), from profiling an SP admin 0 combo-year
# (6 jobs: 200 s shared + ~95 s per job). v2 per-job runtime covered 5 years.
per_job_year = remaining["basin"].map(lambda b: _BASIN_RESOURCES[b][1] / 5)
remaining["runtime_min"] = (
    (args.runtime_factor * per_job_year * (0.67 + 0.33 * remaining["n_jobs"]))
    .clip(lower=10).round(-1).astype(int)
)

###############################################################
project = "proj_rapidresponse"
wf_uuid = uuid.uuid4()

# Path to the worker script (resolved relative to this launcher so the repo
# can be moved without breaking the command).
MAIN_SCRIPT = Path(__file__).resolve().parent / "03_admin_level_paf_main.py"

tool = Tool(name=f"CLIMADA_stage3_admin{ADMIN_LEVEL}")

workflow = tool.create_workflow(
    name=f"CLIMADA_stage3_admin{ADMIN_LEVEL}_combo_{'pilot_' if args.pilot else ''}{wf_uuid}",
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

task_templates = {}
for (memory_gb, runtime_min), _ in remaining.groupby(["memory_gb", "runtime_min"]):
    task_templates[(memory_gb, runtime_min)] = tool.get_task_template(
        template_name=f"CLIMADA_stage3_admin{ADMIN_LEVEL}_combo_{runtime_min}m_{args.cores}c_{memory_gb}G",
        default_cluster_name="slurm",
        default_compute_resources={
            "queue": "all.q",
            "cores": args.cores,
            "memory": f"{memory_gb}G",
            "runtime": f"{runtime_min}m",
            "project": project,
        },
        default_resource_scales={
            "memory": 1.25,
            "runtime": 1.75,
        },
        max_attempts=5,
        command_template=(
            f"python {MAIN_SCRIPT} "
            "--source_id {source_id} "
            "--variant_label {variant_label} "
            "--experiment_id {experiment_id} "
            "--batch_year {batch_year} "
            "--basin {basin} "
            "--year {year} "
            "--num_cores {num_cores} "
            f"--admin_level {ADMIN_LEVEL}"
        ),
        node_args=["source_id", "variant_label", "experiment_id", "batch_year", "basin", "year", "num_cores"],
        task_args=[],
        op_args=[],
    )

tasks = [
    task_templates[(row.memory_gb, row.runtime_min)].create_task(
        name=(
            f"CLIMADA_stage3_a{ADMIN_LEVEL}_{row.source_id}_{row.variant_label}_"
            f"{row.experiment_id}_{row.batch_year}_{row.basin}_{row.year}"
        ),
        source_id=row.source_id,
        variant_label=row.variant_label,
        experiment_id=row.experiment_id,
        batch_year=row.batch_year,
        basin=row.basin,
        year=row.year,
        num_cores=args.cores,
    )
    for row in remaining.itertuples(index=False)
]

print(f"Number of tasks: {len(tasks)} ({args.cores} core(s), at most {args.max_concurrent} at once); "
      f"memory GB and runtime min (min-max) by basin:")
print(remaining.groupby("basin").agg(memory_gb=("memory_gb", "first"), runtime_min=("runtime_min", "min"),
                                     runtime_max=("runtime_min", "max"), tasks=("year", "size")).to_string())

if args.dry_run:
    print("DRY RUN: not binding or submitting. (combo, year) tasks that would run:")
    print(remaining.groupby(["basin", "source_id"]).size().unstack(fill_value=0).to_string())
    raise SystemExit(0)

if tasks:
    workflow.add_tasks(tasks)
    print("✅ Tasks successfully added to workflow.")
else:
    print("⚠️ No tasks added to workflow.")
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
