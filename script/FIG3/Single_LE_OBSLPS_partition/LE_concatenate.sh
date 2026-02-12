#!/bin/bash
#SBATCH --account=mh0033
#SBATCH --job-name=Unforced_Pcorr
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1          # 20 MPI ranks
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --output=le_Unforced_Pcorr_%j.out
#SBATCH --error=le_Unforced_Pcorr_%j.err

# One thread per rank (avoid OpenMP oversubscription)
export OMP_NUM_THREADS=1

source ~/.bashrc
conda activate mykernel

model="$1"

echo "Running segmented ICV trends for model = ${model}"
echo "SLURM_NTASKS = ${SLURM_NTASKS}"

# Use the Slurm task count as MPI size
mpirun -np ${SLURM_NTASKS} python -u concatenate_LE_member_std_segment.py $model