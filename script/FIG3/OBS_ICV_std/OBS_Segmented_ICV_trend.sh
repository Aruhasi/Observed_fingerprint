#!/bin/bash
#SBATCH --account=mh0033
#SBATCH --job-name=OBS_ICV_seg_fast
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks=20          # 10 MPI ranks, 64 L's -> ~6–7 per rank
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=10    # some threads per rank for numpy/BLAS
#SBATCH --time=02:00:00      # should be much less than 1.5h now
#SBATCH --mem=128G
#SBATCH --output=ICV_Segtrend_fast_%j.out
#SBATCH --error=ICV_Segtrend_fast_%j.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/.bashrc
conda activate mykernel

cd /work/mh0033/m301036/OBS_LPS_revision/script/FIG3/OBS_ICV_std/

mpirun -np ${SLURM_NTASKS} python -u OBS_Segmented_ICV_trend_amplitude.py
