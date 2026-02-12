#!/bin/bash
#SBATCH --account=mh0033
#SBATCH --job-name=le_partition_full
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=20              # 7 MPI ranks = 7 models
#SBATCH --ntasks-per-node=20
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --output=le_partition_full_%j.out
#SBATCH --error=le_partition_full_%j.err

# If you don't need a python module, just use conda:
source ~/.bashrc
conda activate mykernel

SCRIPT=/work/mh0033/m301036/OBS_LPS_revision/script/FIG3/Single_LE_OBSLPS_partition/LE_members_GSAT_OLS_anoms_partition_CESM2.py

mpirun -np 20 python -u "$SCRIPT"