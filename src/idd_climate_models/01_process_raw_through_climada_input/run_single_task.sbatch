#!/bin/bash
#SBATCH --job-name=run_basin_tc_risk_single_task
#SBATCH --output=logs/single_task_%j.out
#SBATCH --error=logs/single_task_%j.err
#SBATCH --mem=35G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --partition=all.q
#SBATCH --account=proj_rapidresponse

# Usage: sbatch run_single_task.sbatch <task_id>
# Example: sbatch run_single_task.sbatch 8

TASK_ID=$1
TOTAL_MEMORY="35G"
DATA_SOURCE="cmip6"

if [ -z "$TASK_ID" ]; then
    echo "Error: Task ID required"
    echo "Usage: sbatch run_single_task.sbatch <task_id>"
    exit 1
fi

echo "=========================================="
echo "Running single task: $TASK_ID"
echo "Memory: $TOTAL_MEMORY"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Node: $(hostname)"
echo "=========================================="

# Activate conda environment
source ~/miniconda/etc/profile.d/conda.sh
conda activate idd-climate-models

# Run the task
python 05_run_basin_tc_risk.py \
    --data_source "$DATA_SOURCE" \
    --task_id "$TASK_ID" \
    --total_memory "$TOTAL_MEMORY"

echo "=========================================="
echo "Task $TASK_ID completed: $(date)"
echo "=========================================="
