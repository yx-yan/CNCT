#!/bin/bash
#SBATCH --job-name=thesis_figs
#SBATCH --output=logs/thesis_figs_%j.log
#SBATCH --error=logs/thesis_figs_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_48g
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"

module purge
module load GCC/11.4.0 CUDA/12.3.0 Miniconda3/25.5.1-0
conda activate fyp

PROJECT_ROOT="/home/n2500633e/CNCT"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python: $(python --version)"
echo ""

OUTPUT_DIR="${1:-/projects/CTdata/thesis_outputs}"
echo "Output dir: $OUTPUT_DIR"

python scripts/thesis/generate_thesis_figures.py \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Done at $(date) (${SECONDS}s)"
echo "Outputs:"
find "$OUTPUT_DIR" -type f | sort
