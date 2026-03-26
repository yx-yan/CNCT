#!/bin/bash
#SBATCH --job-name=tigre
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/output_%j.err
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --cpus-per-task=8

echo "==================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "Start     : $(date)"
echo "==================================="

module purge
module load GCC/11.4.0
module load CUDA/12.3.0
module load Miniconda3/25.5.1-0
conda activate tigre

echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "--- CUDA version ---"
nvcc --version 2>/dev/null || echo "nvcc not found in current modules"
python -c "import torch; print('torch CUDA:', torch.version.cuda)" 2>/dev/null || echo "torch not installed"
echo ""

cd "$SLURM_SUBMIT_DIR"

python projection.py
python fdk.py
python evaluation.py

echo ""
echo "Done at $(date)"
