#!/bin/bash
#SBATCH --job-name=tigre
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/output_%j.err
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --cpus-per-task=4

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"

module purge
module load GCC/11.4.0 CUDA/12.3.0 Miniconda3/25.5.1-0
conda activate fyp

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
python -c "
from config import *
for k,v in dict(N_ANGLES=N_ANGLES, ACCURACY=ACCURACY, FDK_FILTER=FDK_FILTER,
    DSO_SCALE=DSO_SCALE, DSD_SCALE=DSD_SCALE, DETECTOR_COL_MARGIN=DETECTOR_COL_MARGIN,
    MU_WATER=MU_WATER, MAX_CASES=MAX_CASES, SAVE_PNG=SAVE_PNG, SAVE_NII=SAVE_NII).items():
    print(f'  {k}: {v}')
"

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

PROJ_DIR=$(python -c "from config import PROJ_DIR; print(PROJ_DIR)")
FDK_DIR=$(python -c "from config import FDK_DIR; print(FDK_DIR)")
EVAL_DIR=$(python -c "from config import EVAL_DIR; print(EVAL_DIR)")

# Stage 1: Forward projection
echo "Stage 1/3 — projection.py"
T0=$SECONDS
python fdk/projection.py
echo "  Done in $((SECONDS - T0))s"

for f in "$PROJ_DIR"/*/projections.npy; do
    [ -f "$f" ] && echo "  $f $(python -c "import numpy as np; print(np.load('$f', mmap_mode='r').shape)")" || echo "  MISSING: $f"
done

# Stage 2: FDK reconstruction
echo "Stage 2/3 — fdk.py"
T0=$SECONDS
python fdk/fdk.py
echo "  Done in $((SECONDS - T0))s"

for f in "$FDK_DIR"/*/recon_fdk.npy; do
    [ -f "$f" ] && echo "  $f $(python -c "import numpy as np; print(np.load('$f', mmap_mode='r').shape)")" || echo "  MISSING: $f"
done

# Stage 3: Evaluation
echo "Stage 3/3 — evaluation.py"
T0=$SECONDS
python fdk/evaluation.py
echo "  Done in $((SECONDS - T0))s"

CSV="$EVAL_DIR/evaluation_results.csv"
[ -f "$CSV" ] && column -t -s, "$CSV" || echo "  WARNING: $CSV not found"

echo "Done at $(date) (${SECONDS}s)"
