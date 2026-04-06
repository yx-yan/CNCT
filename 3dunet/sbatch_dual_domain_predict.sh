#!/bin/bash
#SBATCH --job-name=dual_domain_predict
#SBATCH --output=logs/dual_domain_predict_%j.log
#SBATCH --error=logs/dual_domain_predict_%j.err
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --constraint=gpu_48g
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"

module purge
module load GCC/11.4.0 CUDA/12.3.0 Miniconda3/25.5.1-0
conda activate fyp

PROJECT_ROOT="/home/n2500633e/CNCT"
cd "$PROJECT_ROOT"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT_DIR="/projects/CTdata/dual_domain_checkpoints"
PRED_DIR="${CHECKPOINT_DIR}/predictions"
EVAL_DIR="/projects/CTdata/dual_domain_eval"

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
mkdir -p logs

echo ""
echo "=== Dual-Domain Cascade Prediction ==="
echo "  checkpoint_dir: ${CHECKPOINT_DIR}"
echo "  output_dir:     ${PRED_DIR}"
echo "  eval_dir:       ${EVAL_DIR}"
echo ""

python 3dunet/dual_domain_predict.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_dir "$PRED_DIR"

echo "Evaluating predictions (PSNR, SSIM, comparison PNGs)"
python 3dunet/evaluation.py --pred_dir "$PRED_DIR" --eval_dir "$EVAL_DIR"

echo "Done at $(date) (${SECONDS}s)"
