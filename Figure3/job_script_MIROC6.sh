#!/bin/bash
#SBATCH --account=mh0033
#SBATCH --partition=compute
#SBATCH --job-name=mpi_trend    # Job name
#SBATCH --output=logs/run_%A_%a.out   # Standard output (%A = job array ID, %a = array task ID)
#SBATCH --error=logs/run_%A_%a.err    # Standard error (%A = job array ID, %a = array task ID)
#SBATCH --time=01:45:00        # Maximum runtime (2 hours)
#SBATCH --ntasks=1            # Number of tasks
#SBATCH --cpus-per-task=64      # Number of CPU cores per task
#SBATCH --mem=256G              # Memory per task
#SBATCH --array=0-50%2           # Job array range (50 tasks)

source $(conda info --base)/etc/profile.d/conda.sh
conda activate mykernel # change the name for the environment
python Segment_multi_run_MIROC6.py $SLURM_ARRAY_TASK_ID # change the name of the script