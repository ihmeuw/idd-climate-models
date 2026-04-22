"""
orchestrate_stages.py
Jobmon orchestrator for stage-level TC model fitting.

Every stage fit is independent — no DAG, all jobs run in parallel.
One jobmon task per stage. Skips stages already fully logged.

Run with:
    python src/idd_climate_models/tc_models/orchestrate_stages.py
"""

import json
import uuid
import sys
from pathlib import Path

from jobmon.client.tool import Tool

from idd_climate_models.tc_models.stage_grid import build_stage_grid
from idd_climate_models.tc_models.cache import (
    stage_to_id, is_stage_logged, STAGE_RESULTS_DIR,
)
from idd_climate_models.tc_models.constants import DEFAULT_SEEDS, DEFAULT_FOLDS

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT        = 'proj_rapidresponse'
QUEUE          = 'all.q'
MEMORY         = '16G'
CORES          = 1
RUNTIME        = '02:00:00'
MAX_CONCURRENT = 5000

CONDA_ENV  = 'idd-climate-models'
CONDA_INIT = '/ihme/homes/bcreiner/miniconda/etc/profile.d/conda.sh'
GRID_PATH  = STAGE_RESULTS_DIR / 'stage_grid.json'
WORKER     = Path(__file__).parent / 'run_one_stage.py'
LOGS_DIR   = STAGE_RESULTS_DIR / 'logs'


def _is_complete(stage_id: str, seeds, k_folds: int) -> bool:
    """True iff insample and all OOS folds are logged for this stage."""
    if not is_stage_logged(stage_id, 'insample'):
        return False
    for seed in seeds:
        for fold in range(k_folds):
            if not is_stage_logged(stage_id, 'oos', seed=seed, fold=fold):
                return False
    return True


def main():
    seeds   = DEFAULT_SEEDS
    k_folds = DEFAULT_FOLDS

    # ── Build and save stage grid ──────────────────────────────────────────────
    print("Building stage grid...")
    stages = build_stage_grid()
    print(f"Total stages in grid: {len(stages)}")

    GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(GRID_PATH, 'w') as f:
        json.dump(stages, f)
    print(f"Grid saved: {GRID_PATH}")

    # ── Filter to incomplete stages ────────────────────────────────────────────
    to_run = [
        i for i, stage in enumerate(stages)
        if not _is_complete(stage_to_id(stage), seeds, k_folds)
    ]
    print(f"Stages already complete: {len(stages) - len(to_run)}")
    print(f"Stages to run:           {len(to_run)}")

    if not to_run:
        print("All stages already complete. Exiting.")
        return

    # ── Jobmon workflow ────────────────────────────────────────────────────────
    tool = Tool(name='tc_stage_comparison')
    wf   = tool.create_workflow(
        name=f'tc_stages_{uuid.uuid4()}',
        max_concurrently_running=MAX_CONCURRENT,
    )

    template = tool.get_task_template(
        template_name='run_one_stage',
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
            '--stage_index {stage_index} '
            f'--grid_path {GRID_PATH} '
            f'> {LOGS_DIR}/stage_{{stage_index}}.out '
            f'2> {LOGS_DIR}/stage_{{stage_index}}.err'
        ),
        node_args=['stage_index'],
    )

    tasks = [template.create_task(stage_index=str(i)) for i in to_run]
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

    print("All stages complete.")


if __name__ == '__main__':
    main()
