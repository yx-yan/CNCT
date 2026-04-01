#!/bin/bash
#SBATCH --job-name=3dunet_train
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --gpus=2
#SBATCH --constraint=gpu_48g
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"

module purge
module load GCC/11.4.0 CUDA/12.3.0 Miniconda3/25.5.1-0
conda activate fyp

set -e

PROJECT_ROOT="/home/n2500633e/CNCT"
cd "$PROJECT_ROOT"
export PYTHONPATH=/home/n2500633e/pytorch-3dunet:$PROJECT_ROOT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TQDM_DISABLE=1

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Torch: $(python -c 'import torch; print(torch.__version__, "| CUDA:", torch.version.cuda)')"
mkdir -p logs

# Uncomment on first run to build HDF5 dataset:
# python 3dunet/prepare_data.py

echo "Training — train_config.yaml"
python 3dunet/run_train.py --config 3dunet/train_config.yaml 2>&1

echo "Done at $(date) (${SECONDS}s)"
