#!/bin/bash
#SBATCH --account=mh0033
#SBATCH --job-name=smile_noise
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=20            # 20 MPI ranks over run dimension
#SBATCH --time=06:00:00
#SBATCH --mem=512G
#SBATCH --array=0         # 7 models
#SBATCH --output=smile_noise_%A_%a.out
#SBATCH --error=smile_noise_%A_%a.err

source ~/.bashrc
conda activate mykernel

Model_list=("CESM2")
MODEL=${Model_list[$SLURM_ARRAY_TASK_ID]}

echo "Computing SMILE internal variability for model: $MODEL"
mpirun -np $SLURM_NTASKS python -u LE_SMILE_internal_noise_MPI_CESM2.py "$MODEL"
echo "Finished model: $MODEL"