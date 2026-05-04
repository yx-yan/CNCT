#!/bin/bash
#SBATCH --job-name=cnct_train_imgonly
#SBATCH --output=logs/cnct_train_imgonly_%j.log
#SBATCH --error=logs/cnct_train_imgonly_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_48g
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

# Image-only U-Net baseline: FDK → VolumeUNet3D (Branch B) → reconstruction.
# Reuses the existing cnct-train CLI and Trainer — just passes --image_only
# so the model is Branch B alone (no Branch A, no DBP). Config defaults to
# configs/training.yaml; pass a different YAML as $1 if you want a separate
# checkpoint_dir.

set -euo pipefail

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"

module purge
module load GCC/11.4.0 CUDA/12.3.0 Miniconda3/25.5.1-0
conda activate fyp

PROJECT_ROOT="/home/n2500633e/CNCT"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Torch: $(python -c 'import torch; print(torch.__version__, "| CUDA:", torch.version.cuda)')"
echo ""

CONFIG="${1:-configs/training_imgonly.yaml}"
echo "Config: $CONFIG"
echo "Model:  image-only (FDK → VolumeUNet3D, no Branch A, no DBP)"

python -m cnct.cli.train \
    --config "$CONFIG" \
    --image_only \
    --log_file "logs/cnct_train_imgonly_${SLURM_JOB_ID}.log" \
    --log_level INFO

echo "Done at $(date) (${SECONDS}s)"
