#!/bin/bash

#SBATCH --job-name=sycl-mpi-multi
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=170G
#SBATCH --output=sycl-mpi-multi-%j.out
#SBATCH --error=sycl-mpi-multi-%j.err


# ============================================================
# Load custom SYCL + CUDA + MPICH environment
# ============================================================

source /home/apps/softwares/sycl-stack/setup-env.sh

mpirun \
    -np "$SLURM_NTASKS" \
    ./"$1"
