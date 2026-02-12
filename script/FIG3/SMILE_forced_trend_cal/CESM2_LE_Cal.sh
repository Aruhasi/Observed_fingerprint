#!/bin/bash
#SBATCH --account=mh0033
#SBATCH --job-name=le_partition_full
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4       # or whatever you need
#SBATCH --time=08:00:00
#SBATCH --mem=64G               # adjust as needed
#SBATCH --array=0             # 7 models -> 7 array tasks
#SBATCH --output=le_partition_full_%A_%a.out
#SBATCH --error=le_partition_full_%A_%a.err

source ~/.bashrc
conda activate mykernel

Model_list=("CESM2")
MODEL=${Model_list[$SLURM_ARRAY_TASK_ID]}

echo "Running model: $MODEL"
srun python -u CESM2_smbb_ENS_forced_trend_cal.py "$MODEL"
