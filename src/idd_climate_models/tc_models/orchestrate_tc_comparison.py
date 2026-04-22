"""
orchestrate_tc_comparison.py
Jobmon orchestrator for the full TC model comparison.

One jobmon task per spec. Tasks are skipped if insample is already logged.
Run this script directly:
    python src/idd_climate_models/tc_models/orchestrate_tc_comparison.py
"""

import json
import uuid
import sys
from pathlib import Path

from jobmon.client.tool import Tool

from idd_climate_models.tc_models.spec_grid import build_spec_grid
from idd_climate_models.tc_models.cache     import spec_to_id, is_logged, RESULTS_DIR

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT      = 'proj_rapidresponse'
QUEUE        = 'all.q'
MEMORY       = '16G'
CORES        = 1
RUNTIME      = '02:00:00'
MAX_CONCURRENT = 5000

CONDA_ENV    = 'idd-climate-models'
CONDA_INIT   = '/ihme/homes/bcreiner/miniconda/etc/profile.d/conda.sh'
GRID_PATH    = RESULTS_DIR / 'spec_grid.json'
WORKER       = Path(__file__).parent / 'run_one_spec.py'


def main():
    # ── Build and save spec grid ───────────────────────────────────────────────
    print("Building spec grid...")
    specs = build_spec_grid()
    print(f"Total specs in grid: {len(specs)}")

    GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRID_PATH, 'w') as f:
        json.dump(specs, f)
    print(f"Grid saved: {GRID_PATH}")

    # ── Filter to uncached specs ───────────────────────────────────────────────
    to_run = [i for i, spec in enumerate(specs)
              if not is_logged(spec_to_id(spec), 'insample')]
    print(f"Specs already complete: {len(specs) - len(to_run)}")
    print(f"Specs to run:           {len(to_run)}")

    if not to_run:
        print("All specs already complete. Exiting.")
        return

    # ── Jobmon workflow ────────────────────────────────────────────────────────
    tool = Tool(name='tc_model_comparison')
    wf   = tool.create_workflow(
        name=f'tc_comparison_{uuid.uuid4()}',
        max_concurrently_running=MAX_CONCURRENT,
    )

    template = tool.get_task_template(
        template_name='run_one_spec',
        default_cluster_name='slurm',
        default_compute_resources={
            'memory':  MEMORY,
            'cores':   CORES,
            'runtime': RUNTIME,
            'queue':   QUEUE,
            'project': PROJECT,
        },
        command_template=(
            f'source {CONDA_INIT} && '
            f'conda activate {CONDA_ENV} && '
            f'python {WORKER} '
            '--spec_index {spec_index} '
            f'--grid_path {GRID_PATH}'
        ),
        node_args=['spec_index'],
    )

    tasks = [template.create_task(spec_index=str(i)) for i in to_run]
    wf.add_tasks(tasks)
    print(f"Created {len(tasks)} jobmon tasks")

    wf.bind()
    print(f"Workflow ID: {wf.workflow_id}")
    print(f"Monitor at: https://jobmon-gui.ihme.washington.edu/#/workflow/{wf.workflow_id}")

    status = wf.run(seconds_until_timeout=86400)  # 24h timeout
    print(f"\nWorkflow completed with status: {status}")

    if status != 'D':
        print("ERROR: Workflow did not complete successfully.")
        sys.exit(1)

    print("All specs complete.")


if __name__ == '__main__':
    main()
