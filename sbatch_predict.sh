#!/bin/bash
#SBATCH --job-name=3dunet_predict
#SBATCH --output=logs/predict_%j.log
#SBATCH --error=logs/predict_%j.err
#SBATCH --time=04:00:00
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

export PYTHONPATH=/home/n2500633e/pytorch-3dunet:$PYTHONPATH

echo "GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# ── Inference ─────────────────────────────────────────────────────────────────
echo "Running inference — config: test_config.yaml"
python -m pytorch3dunet.predict --config test_config.yaml

# ── Convert predictions to .npy (optional postprocessing) ────────────────────
python postprocess_predictions.py

echo ""
echo "==================================="
echo "Done at $(date)"
echo "Total time: $((SECONDS))s"
echo "==================================="
