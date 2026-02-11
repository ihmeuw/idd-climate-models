#!/bin/bash
#SBATCH --job-name=test_task_resources
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --partition=all.q
#SBATCH --account=proj_rapidresponse

# Test a task with actual Level 4 resource constraints
python 05_run_basin_tc_risk.py --data_source cmip6 --task_id 1
