#!/bin/bash
#SBATCH --job-name=cnct-dataprep
#SBATCH --output=logs/dataprep_%j.log
#SBATCH --error=logs/dataprep_%j.err
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --cpus-per-task=4

# -----------------------------------------------------------------------------
# Chains the full cnct data-preparation pipeline:
#   1. Forward projection  (cnct-projection)       — cnct_dataprep
#   2. FDK reconstruction  (cnct-fdk)              — cnct_dataprep
#   3. Evaluation          (cnct-evaluation)        — cnct_dataprep
#   4. HDF5 split build    (cnct-prepare-data)     — cnct
#
# Each stage reads its own YAML config under data_prep/configs/. Per-case
# failures are logged and skipped so one bad case does not abort the batch.
# Stage 4 uses the cnct package to build train/val/test HDF5 splits from the
# FDK outputs produced by stage 2.
# -----------------------------------------------------------------------------

set -euo pipefail

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) — $(date)"

module purge
module load GCC/11.4.0 CUDA/12.3.0 Miniconda3/25.5.1-0
conda activate fyp

PROJECT_ROOT="/home/n2500633e/CNCT"
cd "$PROJECT_ROOT"

# Make both packages importable without requiring a rebuild after source edits.
export PYTHONPATH="$PROJECT_ROOT/data_prep/src:$PROJECT_ROOT/src:${PYTHONPATH:-}"

CONFIG_DIR="$PROJECT_ROOT/data_prep/configs"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

JOB_TAG="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
PROJECTION_LOG="$LOG_DIR/dataprep_${JOB_TAG}_projection.log"
FDK_LOG="$LOG_DIR/dataprep_${JOB_TAG}_fdk.log"
EVAL_LOG="$LOG_DIR/dataprep_${JOB_TAG}_evaluation.log"

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# --- Stage 1/4: Forward projection ------------------------------------------
echo "Stage 1/4 — forward projection"
T0=$SECONDS
python -m cnct_dataprep.cli.projection \
    --config "$CONFIG_DIR/projection.yaml" \
    --log-file "$PROJECTION_LOG"
echo "  Done in $((SECONDS - T0))s"

# --- Stage 2/4: FDK reconstruction ------------------------------------------
echo "Stage 2/4 — FDK reconstruction"
T0=$SECONDS
python -m cnct_dataprep.cli.fdk \
    --config "$CONFIG_DIR/fdk.yaml" \
    --log-file "$FDK_LOG"
echo "  Done in $((SECONDS - T0))s"

# --- Stage 3/4: Evaluation --------------------------------------------------
echo "Stage 3/4 — evaluation"
T0=$SECONDS
python -m cnct_dataprep.cli.evaluation \
    --config "$CONFIG_DIR/evaluation.yaml" \
    --log-file "$EVAL_LOG"
echo "  Done in $((SECONDS - T0))s"

# --- Stage 4/4: HDF5 split build -------------------------------------------
echo "Stage 4/4 — HDF5 train/val/test split build"
H5_LOG="$LOG_DIR/dataprep_${JOB_TAG}_h5build.log"
T0=$SECONDS
python -m cnct.cli.prepare \
    --log_level INFO \
    2>&1 | tee "$H5_LOG"
echo "  Done in $((SECONDS - T0))s"

echo "Done at $(date) (total ${SECONDS}s)"
