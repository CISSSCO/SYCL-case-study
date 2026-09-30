#!/bin/bash
#SBATCH --job-name=sycl-gpu
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=sycl-gpu-%j.out
#SBATCH --error=sycl-gpu-%j.err

module load syc/cpu_gpu

echo "========================================"
echo "SYCL GPU JOB"
echo "========================================"

hostname
nvidia-smi

echo
echo "SLURM_JOB_ID      = $SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"

echo
echo "SYCL Devices:"
sycl-ls

echo
echo "Running GPU executable..."

./$1

#!/bin/bash
#SBATCH --job-name=sycl-gpu
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=sycl-gpu-%j.out
#SBATCH --error=sycl-gpu-%j.err

module load syc/cpu_gpu

echo "========================================"
echo "SYCL GPU JOB"
echo "========================================"

hostname
nvidia-smi

echo
echo "SLURM_JOB_ID      = $SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"

echo
echo "SYCL Devices:"
sycl-ls

echo
echo "Running GPU executable..."

./$1
