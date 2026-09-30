#!/bin/bash
#SBATCH --job-name=sycl-cpu
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:30:00
#SBATCH --mem=170G
#SBATCH --output=sycl-cpu-%j.out
#SBATCH --error=sycl-cpu-%j.err

source /home/apps/softwares/sycl-stack/setup-env.sh

echo "========================================"
echo "SYCL CPU JOB"
echo "========================================"

hostname

echo
echo "SLURM_JOB_ID      = $SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"

echo
echo "SYCL Devices:"
sycl-ls

echo
echo "Running CPU executable..."

./$1
