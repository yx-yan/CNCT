#!/bin/bash
#SBATCH --job-name=cnct-dataprep
#SBATCH --output=logs/dataprep_%j.log
#SBATCH --error=logs/dataprep_%j.err
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --cpus-per-task=4

# -----------------------------------------------------------------------------
# Chains the three cnct_dataprep pipeline stages:
#   1. Forward projection  (cnct-projection)
#   2. FDK reconstruction  (cnct-fdk)
#   3. Evaluation          (cnct-evaluation)
#
# Each stage reads its own YAML config under data_prep/configs/. Per-case
# failures are logged and skipped so one bad case does not abort the batch.
# -----------------------------------------------------------------------------

set -euo pipefail

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) — $(date)"

module purge
module load GCC/11.4.0 CUDA/12.3.0 Miniconda3/25.5.1-0
conda activate fyp

PROJECT_ROOT="/home/n2500633e/CNCT"
cd "$PROJECT_ROOT"

# Make the package importable without requiring a rebuild after source edits.
export PYTHONPATH="$PROJECT_ROOT/data_prep/src:${PYTHONPATH:-}"

CONFIG_DIR="$PROJECT_ROOT/data_prep/configs"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

JOB_TAG="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
PROJECTION_LOG="$LOG_DIR/dataprep_${JOB_TAG}_projection.log"
FDK_LOG="$LOG_DIR/dataprep_${JOB_TAG}_fdk.log"
EVAL_LOG="$LOG_DIR/dataprep_${JOB_TAG}_evaluation.log"

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# --- Stage 1/3: Forward projection ------------------------------------------
echo "Stage 1/3 — forward projection"
T0=$SECONDS
python -m cnct_dataprep.cli.projection \
    --config "$CONFIG_DIR/projection.yaml" \
    --log-file "$PROJECTION_LOG"
echo "  Done in $((SECONDS - T0))s"

# --- Stage 2/3: FDK reconstruction ------------------------------------------
echo "Stage 2/3 — FDK reconstruction"
T0=$SECONDS
python -m cnct_dataprep.cli.fdk \
    --config "$CONFIG_DIR/fdk.yaml" \
    --log-file "$FDK_LOG"
echo "  Done in $((SECONDS - T0))s"

# --- Stage 3/3: Evaluation --------------------------------------------------
echo "Stage 3/3 — evaluation"
T0=$SECONDS
python -m cnct_dataprep.cli.evaluation \
    --config "$CONFIG_DIR/evaluation.yaml" \
    --log-file "$EVAL_LOG"
echo "  Done in $((SECONDS - T0))s"

echo "Done at $(date) (total ${SECONDS}s)"
