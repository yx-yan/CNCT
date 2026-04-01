#!/bin/bash
#SBATCH --job-name=3dunet_predict
#SBATCH --output=logs/predict_%j.log
#SBATCH --error=logs/predict_%j.err
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --constraint=gpu_48g
#SBATCH --cpus-per-task=4

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"

module purge
module load GCC/11.4.0 CUDA/12.3.0 Miniconda3/25.5.1-0
conda activate fyp

export PYTHONPATH=/home/n2500633e/pytorch-3dunet:$PYTHONPATH

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Inference — test_config.yaml"
python -m pytorch3dunet.predict --config test_config.yaml

echo "Postprocessing predictions"
python postprocess_predictions.py

echo "Done at $(date) (${SECONDS}s)"
