#!/bin/bash
#SBATCH --account=mh0033
#SBATCH --job-name=OBS_ICV_singleL
#SBATCH --partition=compute
#SBATCH --array=10-73           # one task per L
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/ICV_singleL_%A_%a.out
#SBATCH --error=logs/ICV_singleL_%A_%a.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/.bashrc
conda activate mykernel

cd /work/mh0033/m301036/OBS_LPS_revision/script/FIG3/OBS_ICV_std/

L=${SLURM_ARRAY_TASK_ID}
srun python -u OBS_ICV_singleL.py ${L}
