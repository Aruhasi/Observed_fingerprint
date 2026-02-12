#!/bin/bash
#SBATCH --account=mh0033
#SBATCH --job-name=le_forced_trend
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=10              # 10 MPI ranks (each processes ~5 runs)
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=256G
#SBATCH --output=le_forced_trend_%j.out
#SBATCH --error=le_forced_trend_%j.err

# If you don't need a python module, just use conda:
source ~/.bashrc
conda activate mykernel

model=CESM2
mpirun -np 10 python -u LE_forced_unforced_trend_cal_CESM2.py ${model}