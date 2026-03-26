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

# ── Stage 1: Forward projection ─────────────────────────────────────────────
echo "==========================================="
echo "STAGE 1/3 — projection.py"
echo "==========================================="
T0=$SECONDS
python projection.py
echo "  projection.py finished in $((SECONDS - T0))s"
echo ""

# Sanity check: confirm projections.npy was written for each case
echo "--- Projection outputs ---"
for f in output/*/projections.npy; do
    if [ -f "$f" ]; then
        SHAPE=$(python -c "import numpy as np; a=np.load('$f', mmap_mode='r'); print(f'  {a.shape}')" 2>/dev/null)
        echo "  $f  shape=$SHAPE"
    else
        echo "  MISSING: $f"
    fi
done
echo ""

# ── Stage 2: FDK reconstruction ─────────────────────────────────────────────
echo "==========================================="
echo "STAGE 2/3 — fdk.py"
echo "==========================================="
T0=$SECONDS
python fdk.py
echo "  fdk.py finished in $((SECONDS - T0))s"
echo ""

# Sanity check: confirm recon_fdk.npy was written for each case
echo "--- Reconstruction outputs ---"
for f in output/*/recon_fdk.npy; do
    if [ -f "$f" ]; then
        SHAPE=$(python -c "import numpy as np; a=np.load('$f', mmap_mode='r'); print(f'  {a.shape}')" 2>/dev/null)
        echo "  $f  shape=$SHAPE"
    else
        echo "  MISSING: $f"
    fi
done
echo ""

# ── Stage 3: Evaluation ──────────────────────────────────────────────────────
echo "==========================================="
echo "STAGE 3/3 — evaluation.py"
echo "==========================================="
T0=$SECONDS
python evaluation.py
echo "  evaluation.py finished in $((SECONDS - T0))s"
echo ""

# Print the CSV summary if it was written
CSV="output/evaluation_results.csv"
if [ -f "$CSV" ]; then
    echo "--- Evaluation results ---"
    column -t -s, "$CSV"
else
    echo "  WARNING: $CSV not found"
fi

echo ""
echo "==================================="
echo "Done at $(date)"
echo "Total time: $((SECONDS))s"
echo "==================================="
