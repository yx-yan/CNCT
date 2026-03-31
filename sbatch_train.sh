#!/bin/bash
#SBATCH --job-name=3dunet_train
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --constraint=gpu_48g
#SBATCH --cpus-per-task=4

echo "==================================="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "Start   : $(date)"
echo "==================================="

module purge
module load GCC/11.4.0
module load CUDA/12.3.0
module load Miniconda3/25.5.1-0
conda activate fyp

# Make pytorch3dunet importable from the cloned repo
export PYTHONPATH=/home/n2500633e/pytorch-3dunet:$PYTHONPATH

echo "GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "torch   : $(python -c 'import torch; print(torch.__version__, "| CUDA:", torch.version.cuda)')"
echo ""

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# ── (Optional) data preparation ───────────────────────────────────────────────
# Uncomment the line below on the first run to build the HDF5 dataset.
# Subsequent runs can skip this step (existing files are skipped automatically).
#
python prepare_data.py

# ── Training ──────────────────────────────────────────────────────────────────
echo "Starting training — config: train_config.yaml"
python -m pytorch3dunet.train --config train_config.yaml

echo ""
echo "==================================="
echo "Done at $(date)"
echo "Total time: $((SECONDS))s"
echo "==================================="
