#!/bin/bash
#SBATCH --job-name=dual_domain_train
#SBATCH --output=logs/dual_domain_%j.log
#SBATCH --error=logs/dual_domain_%j.err
#SBATCH --time=48:00:00
#SBATCH --gpus=2
#SBATCH --constraint=gpu_48g
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"

module purge
module load GCC/11.4.0 CUDA/12.3.0 Miniconda3/25.5.1-0
conda activate fyp

set -e

PROJECT_ROOT="/home/n2500633e/CNCT"
cd "$PROJECT_ROOT"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TQDM_DISABLE=1
# Reduce CUDA memory fragmentation — lets PyTorch grow allocations
# incrementally instead of reserving large contiguous blocks upfront.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Torch: $(python -c 'import torch; print(torch.__version__, "| CUDA:", torch.version.cuda)')"
mkdir -p logs

# --- Config summary ---
echo ""
echo "=== Dual-Domain Cascade Training ==="
echo "  spatial_scale: 0.5  (Y,X downsampled to half)"
echo "  sino_features: 4    (Branch A output channels)"
echo "  sino_f_maps:   8 16 32"
echo "  vol_f_maps:    8 16 32 64 128"
echo "  AMP:           enabled"
echo "  Grad checkpoint: enabled"
echo ""

python 3dunet/dual_domain_train.py \
    --epochs 200 \
    --lr 2e-4 \
    --weight_decay 1e-5 \
    --sino_features 4 \
    --sino_f_maps 8 16 32 \
    --vol_f_maps 8 16 32 64 128 \
    --spatial_scale 0.5 \
    --checkpoint_dir /projects/CTdata/dual_domain_checkpoints \
    --seed 42 \
    2>&1

echo "Done at $(date) (${SECONDS}s)"
