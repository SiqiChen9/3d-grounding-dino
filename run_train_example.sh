#!/bin/bash -l
#
# SLURM job script for 3D Grounding DINO training on TinyGPU
#
#SBATCH --job-name=3d-grounding-dino
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --export=NONE
#SBATCH --mail-user=<YOUR_EMAIL>@fau.de
#SBATCH --mail-type=END,FAIL

# Unset SLURM_EXPORT_ENV to avoid inheriting environment
unset SLURM_EXPORT_ENV

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Load Python module and activate conda environment
# module load python
conda activate 3d-detection

# Verify GPU is available
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Change to project directory
cd path/to/3d-grounding-dino

# Create logs directory if it doesn't exist
mkdir -p logs

# ==========================================
# Data Staging: copy numpy_volumes to local SSD
# Set STAGE_DATA="true" to copy, "false" to read from $WORK
# ==========================================
STAGE_DATA="false"

DATASET_SOURCE=/home/woody/iwi5/iwi5378h/rsna2023
DATASET_LOCAL="$TMPDIR/rsna2023"

echo "TMPDIR: $TMPDIR"
df -h "$TMPDIR"

if [ "$STAGE_DATA" = "true" ]; then
    echo "Staging numpy_volumes to local SSD..."
    mkdir -p "$DATASET_LOCAL"
    
    if [ -d "$DATASET_SOURCE/numpy_volumes" ]; then
        time cp -r "$DATASET_SOURCE/numpy_volumes" "$DATASET_LOCAL/"
        echo "Done! Size: $(du -sh "$DATASET_LOCAL/numpy_volumes" | cut -f1)"
        FINAL_DATASET_PATH="$DATASET_LOCAL"
    else
        echo "ERROR: numpy_volumes not found. Run: python -m utils.numpy_converter"
        exit 1
    fi
else
    echo "Using dataset source directly (no staging)"
    FINAL_DATASET_PATH="$DATASET_SOURCE"
fi

# Create config with updated dataset path
TEMP_CONFIG="$TMPDIR/config_local.yaml"
sed "s|dataset_path:.*|dataset_path: $FINAL_DATASET_PATH|g" configs/default_config.yaml > "$TEMP_CONFIG"
echo "Dataset path in config: $FINAL_DATASET_PATH"

# Run training
echo "=========================================="
echo "Starting training..."
echo "=========================================="
python train.py --config "$TEMP_CONFIG"

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
