#!/bin/bash -l
#
# SLURM job script for Swin3D backbone pretraining on TinyGPU
#
#SBATCH --job-name=swin3d-pretrain
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --output=logs/pretrain/slurm_%j.out
#SBATCH --error=logs/pretrain/slurm_%j.err
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

# Activate conda environment
conda activate 3d-detection

# Verify GPU
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Change to project directory
cd path/to/3d-grounding-dino

# Create log directories
mkdir -p logs_pretrain
mkdir -p checkpoints_pretrain

# ==========================================
# Dataset path setup
# ==========================================
DATASET_SOURCE=/home/woody/iwi5/iwi5378h/rsna2023
FINAL_DATASET_PATH="$DATASET_SOURCE"

# Create config with updated dataset path (if needed)
TEMP_CONFIG="$TMPDIR/pretrain_config.yaml"
sed "s|dataset_path:.*|dataset_path: $FINAL_DATASET_PATH|g" configs/pretrain_config.yaml > "$TEMP_CONFIG"
echo "Dataset path: $FINAL_DATASET_PATH"

# Run pretraining
echo "=========================================="
echo "Starting Swin3D backbone pretraining..."
echo "=========================================="
python pretrain.py --config "$TEMP_CONFIG"

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
