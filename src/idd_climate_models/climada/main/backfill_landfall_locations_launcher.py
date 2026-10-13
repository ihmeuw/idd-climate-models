"""
One-time Jobmon workflow for backfill_landfall_locations.py (see its docstring).

    scan shards (100)  ->  fix shards (24)  ->  summarize (1)

Each task runs one `--step` of backfill_landfall_locations.py with its own SLURM
resources, so nothing depends on the launching session's cores or memory.
Results go to <output>_log/landfall_backfill/<timestamp>_<mode>/ (report.csv
at the end). A dry run writes nothing else; --apply rewrites the affected
landfall parquets and refuses to start unless every stage 1 draw is finished.

Run from climada_env:
    python backfill_landfall_locations_launcher.py            # dry run
    python backfill_landfall_locations_launcher.py --apply    # write
"""

import argparse
import json
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from jobmon.client.tool import Tool  # type: ignore

import backfill_landfall_locations as bf
from stage1_run_config import DEFAULT_SAVE_ROOT, log_dir_for, require_stage1_complete

parser = argparse.ArgumentParser(description="Launch the stage 1 landfall-parquet backfill on Jobmon")
parser.add_argument("--apply", action="store_true", help="rewrite the affected parquets (default: dry run)")
parser.add_argument("--output_path", type=Path, default=DEFAULT_SAVE_ROOT, help="stage 1 output root")
parser.add_argument("--n_scan_shards", type=int, default=100)
parser.add_argument("--n_fix_shards", type=int, default=24)
args = parser.parse_args()

# Tasks run `python backfill_landfall_locations.py` with this shell's environment.
_python = shutil.which("python")
if subprocess.run([_python, "-c", "import climada"], capture_output=True).returncode != 0:
    raise SystemExit(f"❌ `python` here ({_python}) cannot import climada. Activate climada_env and rerun.")

if args.apply:
    # No running stage 1 task may rewrite these parquets.
    require_stage1_complete(set(bf.all_draws()), args.output_path)

mode = "apply" if args.apply else "dry_run"
run_dir = log_dir_for(args.output_path) / "landfall_backfill" / f"{datetime.now():%Y%m%d_%H%M%S}_{mode}"
run_dir.mkdir(parents=True)
with open(run_dir / "config.json", "w") as f:
    json.dump({"save_root": str(args.output_path), "apply": args.apply,
               "n_scan_shards": args.n_scan_shards, "n_fix_shards": args.n_fix_shards}, f, indent=2)
print(f"{mode.upper()} | run folder {run_dir}")

project = "proj_rapidresponse"
SCRIPT = Path(__file__).resolve().parent / "backfill_landfall_locations.py"
tool = Tool(name="CLIMADA_stage1_landfall_backfill")
workflow = tool.create_workflow(name=f"CLIMADA_stage1_landfall_backfill_{mode}_{uuid.uuid4()}")
workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={"memory": "6G", "cores": 1, "runtime": "60m", "queue": "all.q", "project": project},
)


def template(step: str, memory: str, runtime: str, sharded: bool = True):
    return tool.get_task_template(
        template_name=f"landfall_backfill_{step}",
        default_cluster_name="slurm",
        default_compute_resources={"queue": "all.q", "cores": 1, "memory": memory, "runtime": runtime,
                                   "project": project},
        default_resource_scales={"memory": 1.5, "runtime": 1.5},
        max_attempts=3,
        command_template=f"python {SCRIPT} --step {step} --run_dir {{run_dir}}" + (" --shard {shard}" if sharded else ""),
        node_args=["shard"] if sharded else [],
        task_args=["run_dir"],
        op_args=[],
    )


scan_template = template("scan", "6G", "60m")
# Each fix shard loads the land polygons once (~3 GB per file) and fixes ~100 storms.
fix_template = template("fix", "24G", "120m")
summary_template = template("summarize", "8G", "30m", sharded=False)

scan_tasks = [
    scan_template.create_task(name=f"landfall_backfill_scan_{i}", shard=i, run_dir=str(run_dir))
    for i in range(args.n_scan_shards)
]
fix_tasks = [
    fix_template.create_task(name=f"landfall_backfill_fix_{i}", shard=i, run_dir=str(run_dir),
                             upstream_tasks=scan_tasks)
    for i in range(args.n_fix_shards)
]
summary_task = summary_template.create_task(
    name="landfall_backfill_summarize", run_dir=str(run_dir), upstream_tasks=fix_tasks)

workflow.add_tasks(scan_tasks + fix_tasks + [summary_task])
workflow.bind()
print(f"Workflow {workflow.workflow_id}: https://jobmon-gui.ihme.washington.edu/#/workflow/{workflow.workflow_id}")
status = workflow.run(seconds_until_timeout=24 * 3600)
print(f"Workflow {workflow.workflow_id} finished with status {status}. Report: {run_dir / 'report.csv'}")
