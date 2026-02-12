#!/bin/bash
#SBATCH --account=mh0033
#SBATCH --job-name=ICV_seg
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=20          # 20 MPI ranks
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=256G
#SBATCH --output=le_ICV_Segtrend_%j.out
#SBATCH --error=le_ICV_Segtrend_%j.err

# One thread per rank (avoid OpenMP oversubscription)
export OMP_NUM_THREADS=1

source ~/.bashrc
conda activate mykernel

model="$1"

echo "Running segmented ICV trends for model = ${model}"
echo "SLURM_NTASKS = ${SLURM_NTASKS}"

# Use the Slurm task count as MPI size
mpirun -np "${SLURM_NTASKS}" python -u LE_Segmented_ICV_trend_amplitude_CESM2.py "${model}"