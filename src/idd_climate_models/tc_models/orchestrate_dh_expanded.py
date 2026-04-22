"""
orchestrate_dh_expanded.py
Jobmon orchestrator for expanded DH covariate grid.

Based on lessons learned from initial sweep:
- Only 70% threshold
- No poisson/tweedie
- Expanded covariate combinations

Run with:
    python src/idd_climate_models/tc_models/orchestrate_dh_expanded.py
"""

import json
import uuid
import sys
from pathlib import Path

from jobmon.client.tool import Tool

from idd_climate_models.tc_models.stage_grid_dh_expanded import build_stage_grid
from idd_climate_models.tc_models.cache import stage_to_id
from idd_climate_models.tc_models.constants import DEFAULT_SEEDS, DEFAULT_FOLDS

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT        = 'proj_rapidresponse'
QUEUE          = 'all.q'
MEMORY         = '16G'
CORES          = 1
RUNTIME        = '02:00:00'
MAX_CONCURRENT = 500

CONDA_ENV  = 'idd-climate-models'
CONDA_INIT = '/ihme/homes/bcreiner/miniconda/etc/profile.d/conda.sh'

# Output directory - use FRESH directory to avoid contamination from old runs
OUTPUT_DIR  = Path('/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/stage_results_dh_v2')
GRID_PATH   = OUTPUT_DIR / 'stage_grid.json'
MODELS_DIR  = OUTPUT_DIR / 'models'
RESULTS_DIR = OUTPUT_DIR / 'results'   # CSV outputs
LOGS_DIR    = OUTPUT_DIR / 'logs'
STAGE_LOGS  = OUTPUT_DIR / 'stage_logs'  # JSON per-stage logs (separate from CSV results)

WORKER = Path(__file__).parent / 'run_one_stage.py'


def _is_complete(stage_id: str, seeds, k_folds: int) -> bool:
    """True iff insample and all OOS folds have pkl files."""
    insample_pkl = MODELS_DIR / f'{stage_id}_insample.pkl'
    if not insample_pkl.exists():
        return False
    for seed in seeds:
        for fold in range(k_folds):
            oos_pkl = MODELS_DIR / f'{stage_id}_seed{seed}_fold{fold}.pkl'
            if not oos_pkl.exists():
                return False
    return True


def main():
    seeds   = DEFAULT_SEEDS
    k_folds = DEFAULT_FOLDS

    # ── Create directories ─────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    STAGE_LOGS.mkdir(parents=True, exist_ok=True)

    # ── Build and save stage grid ──────────────────────────────────────────────
    print("Building EXPANDED stage grid...")
    stages = build_stage_grid()
    print(f"Total stages in grid: {len(stages)}")

    # Count by type
    by_type = {}
    for s in stages:
        t = s['stage_type']
        by_type[t] = by_type.get(t, 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"  {t}: {n}")

    with open(GRID_PATH, 'w') as f:
        json.dump(stages, f, indent=2)
    print(f"\nGrid saved: {GRID_PATH}")

    # ── Filter to incomplete stages ────────────────────────────────────────────
    to_run = []
    for i, stage in enumerate(stages):
        stage_id = stage_to_id(stage)
        if not _is_complete(stage_id, seeds, k_folds):
            to_run.append(i)

    print(f"\nStages already complete: {len(stages) - len(to_run)}")
    print(f"Stages to run:           {len(to_run)}")

    if not to_run:
        print("All stages already complete. Exiting.")
        return

    # ── Jobmon workflow ────────────────────────────────────────────────────────
    tool = Tool(name='tc_dh_expanded')
    wf   = tool.create_workflow(
        name=f'tc_dh_expanded_{uuid.uuid4()}',
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
            f'export STAGE_MODELS_DIR={MODELS_DIR} && '
            f'export STAGE_RESULTS_DIR={STAGE_LOGS} && '
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
    print(f"\nCreated {len(tasks)} jobmon tasks")

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
