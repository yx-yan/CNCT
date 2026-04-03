#!/bin/bash
#SBATCH --job-name=3dunet_fullvol_predict
#SBATCH --output=logs/fullvol_predict_%j.log
#SBATCH --error=logs/fullvol_predict_%j.err
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --constraint=gpu_48g
#SBATCH --cpus-per-task=4

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"

module purge
module load GCC/11.4.0 CUDA/12.3.0 Miniconda3/25.5.1-0
conda activate fyp

PROJECT_ROOT="/home/n2500633e/CNCT"
cd "$PROJECT_ROOT"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

PRED_DIR="/projects/CTdata/3dunet_checkpoints_fullvol/predictions"
EVAL_DIR="/projects/CTdata/3dunet_fullvol_eval"

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
mkdir -p logs

echo "Full-volume inference — fullvol_train_config.yaml"
python 3dunet/run_fullvol_predict.py --config 3dunet/fullvol_train_config.yaml \
    --output_dir "$PRED_DIR"

echo "Evaluating predictions (PSNR, SSIM, comparison PNGs)"
python 3dunet/evaluation.py --pred_dir "$PRED_DIR" --eval_dir "$EVAL_DIR"

echo "Done at $(date) (${SECONDS}s)"
