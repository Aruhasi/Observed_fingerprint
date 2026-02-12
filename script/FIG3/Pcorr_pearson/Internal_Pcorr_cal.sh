#!/bin/bash
#SBATCH --account=mh0033
#SBATCH --job-name=Unforced_Pcorr
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=10          # 10 MPI ranks
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
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
mpirun -np ${SLURM_NTASKS} python -u OBS_LPS_vs_SMILE_unforced_std_Pcorr.py $model \
        /work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/OBS_ICV_std/concatenate/OBS_ICV_MK_trend_STD_1950_2022_sliding.nc