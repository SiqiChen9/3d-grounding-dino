#!/bin/bash -l
#
# SLURM job script for 3D RetinaNet training on TinyGPU
# 用于与 3D Grounding DINO 对比的基线检测模型
#
#SBATCH --job-name=retinanet-3d
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=08:00:00
#SBATCH --output=logs_retinanet/slurm_%j.out
#SBATCH --error=logs_retinanet/slurm_%j.err
#SBATCH --export=NONE
#SBATCH --mail-user=<YOUR_EMAIL>@fau.de
#SBATCH --mail-type=END,FAIL

# Unset SLURM_EXPORT_ENV to avoid inheriting environment
source ~/.bashrc
unset SLURM_EXPORT_ENV

# Print job info
echo "=========================================="
echo "3D RetinaNet 训练"
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
cd ~/3d-grounding-dino

# Create logs directory if it doesn't exist
mkdir -p logs_retinanet

# ==========================================
# 可选: 数据分阶段 - 将 numpy_volumes 复制到本地 SSD
# 设置 STAGE_DATA="true" 来复制，"false" 直接读取 $WORK
# ==========================================
STAGE_DATA="false"

DATASET_SOURCE=/home/woody/iwi5/iwi5378h/rsna2023
DATASET_LOCAL="$TMPDIR/rsna2023"

echo "TMPDIR: $TMPDIR"
df -h "$TMPDIR"

if [ "$STAGE_DATA" = "true" ]; then
    echo "正在将 numpy_volumes 分阶段到本地 SSD..."
    mkdir -p "$DATASET_LOCAL"
    
    if [ -d "$DATASET_SOURCE/numpy_volumes" ]; then
        time cp -r "$DATASET_SOURCE/numpy_volumes" "$DATASET_LOCAL/"
        echo "完成! 大小: $(du -sh "$DATASET_LOCAL/numpy_volumes" | cut -f1)"
        FINAL_DATASET_PATH="$DATASET_LOCAL"
    else
        echo "错误: 未找到 numpy_volumes. 运行: python -m utils.numpy_converter"
        exit 1
    fi
else
    echo "直接使用数据集来源 (无分阶段)"
    FINAL_DATASET_PATH="$DATASET_SOURCE"
fi

# 创建包含更新数据集路径的临时配置
TEMP_CONFIG="$TMPDIR/retinanet_config_local.yaml"
sed "s|dataset_path:.*|dataset_path: $FINAL_DATASET_PATH|g" configs/retinanet_config.yaml > "$TEMP_CONFIG"
echo "配置中的数据集路径: $FINAL_DATASET_PATH"

# ==========================================
# 训练参数 (可根据需要调整)
# ==========================================
BATCH_SIZE=2
EPOCHS=500
LR=0.0001
IMAGE_FORMAT="numpy"  # dcm, jpeg, 或 numpy

echo "=========================================="
echo "训练参数"
echo "=========================================="
echo "批次大小: $BATCH_SIZE"
echo "训练轮数: $EPOCHS"
echo "学习率: $LR"
echo "图像格式: $IMAGE_FORMAT"
echo "=========================================="

# 运行训练
echo ""
echo "=========================================="
echo "开始训练..."
echo "=========================================="
python Retinanet/retinanet_3d.py \
    --config "$TEMP_CONFIG" \
    --data-dir "$FINAL_DATASET_PATH"

echo ""
echo "=========================================="
echo "任务完成: $(date)"
echo "=========================================="
