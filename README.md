# 3D Grounded Detection for CT

> **Status:** MVP Complete ✓
> 
> This repository implements a 3D object detection framework for CT scans, combining 3D Swin Transformers, DETR-style architecture, and grounding-style supervision with pseudo-class tokens.

---

## Overview

This project implements a **3D Grounding-DETR framework** for volumetric CT scan analysis, adapted from the Grounding-DINO architecture for medical imaging:

### ✅ Completed Features
- ✅ **3D Swin Transformer backbone** with hierarchical feature extraction
  - 目前仍有若干优化方向：
    1. 启用多尺度输出（接口已预留好，须在其他文件做适配）：这可以提高对微小病灶的检测。但是会显著增加显存占用和计算量。
    2. ~~解决 Mask 的重复计算：这可以减少显存占用和计算量。但由于当前每个 epoch 内的 batch 较小，推测无法显著优化；另外若输入图像分辨率不一致需要做适配。~~
    3. 缺乏 Gradient Checkpointing：这可以减少内存占用。但鉴于目前模型对内存的占用仍可接受，暂该不做优化。
    4. 频繁的 Padding 和 Unpadding（并不能优化很多性能）。#性价比不高
    5. 使用 Flash Attention 加速：提高内存读取效率。但鉴于 Swin 的特性（Relative Position Bias）无法直接调用。#较难
- ⚠️ **Cross-modality decoder** with separate text and image attention mechanisms (self-attention is TODO)
- ✅ **Pseudo class token system** replacing text encoder for medical domain
- ⚠️ **Feature Enhancer** - Basic implementation (bidirectional cross-attention is TODO)
- ⚠️ **Query Selection** - Fixed learnable queries (dynamic selection is TODO)
- ✅ **Hungarian matching** with set-based loss (CE + L1 + 3D GIoU)
- ✅ **Complete training pipeline** with AdamW optimizer and cosine scheduling
- ✅ **Evaluation metrics** including 3D IoU and mAP at multiple thresholds
- ✅ **Comprehensive visualization tools** for 3D boxes and multi-slice views

### 📋 Development Roadmap

This section outlines the planned development phases for the project. The current focus is on completing core module unit tests, implementing remaining TODOs in the codebase, and then proceeding to full-scale training.

---

#### 🔬 Phase 1: Unit Testing (Current Priority)

**Objective:** Ensure all core modules are thoroughly tested before proceeding to implementation and training.

- 🔄 **Core Module Unit Tests**
- 📝 **Test Coverage Goals**
  - Input/output shape validation for all modules
  - Gradient flow verification
  - Edge case handling (empty batches, single-class scenarios)
  - Numerical stability tests (loss functions, IoU computation)

#### 🛠️ Phase 2: Code Implementation TODOs (Next Priority)

**Objective:** Complete the placeholder implementations in core modules to match the full architecture design.

- 2.1 Feature Enhancer (`models/feature_enhancer.py`)
- 2.2 Language-Guided Query Selection (`models/query_selection.py`)
- 2.3 Window-Based Attention (`models/swin3d_backbone.py`) **Current Status:** ⚠️ Full attention implementation (MVP)

#### 🚀 Phase 3: Training and Evaluation

---
## MVP Features ✅

The current MVP is **complete and functional**. All core components have been implemented and validated:

### ✅ Data Pipeline
- **NIfTI segmentation loading** with automatic 3D bounding box extraction
- **DICOM/JPEG/NumPy data loading** into volumetric arrays
  - `image_format='numpy'`: Load pre-converted .npz files (fastest, recommended)
  - `image_format='dcm'`: Load DICOM files with HU conversion
  - `image_format='jpeg'`: Load JPEG files (for small-scale testing)
- **NumPy converter tool** for pre-processing DICOM to .npz format (~4x faster I/O)
- **Intensity normalization** for CT Hounsfield units
- **Volume resizing** with trilinear interpolation to `(64, 64, 64)`
- **Data augmentation** with 3D rotations (±30°), scaling, elastic deformation, and intensity jitter (configurable, no flipping for anatomical correctness)
- **PyTorch Dataset integration** with custom collate function for variable-length boxes
- **Train/validation split** with configurable ratio

### ✅ Model Architecture
- **3D Swin Transformer backbone** (`swin3d_backbone.py`) ✅
  - 4-stage hierarchical feature extraction with depths `[2, 2, 6, 2]`
  - Patch embedding, window attention, and patch merging
  - Output dimension: 256 channels
  
- **Pseudo Text Feature Generator** (`text_feature_generator.py`) ✅
  - Learnable class embeddings `(num_classes, 256)`
  - MLP projection for feature transformation
  - Replaces text encoder for medical domain
  
- **Feature Enhancer** (`feature_enhancer.py`) ⚠️ **TODO**
  - Currently implements identity/pass-through operation
  - **Planned**: Bidirectional cross-attention between image and text features
  
- **Language-guided Query Selection** (`query_selection.py`) ⚠️ **TODO**
  - Currently generates fixed learnable queries
  - **Planned**: Dynamic query selection based on enhanced text features
  
- **Cross-Modality Decoder** (`cross_modality_decoder.py`) ⚠️ **TODO**
  - Transformer decoder (6 layers) with dual cross-attention:
    - Text cross-attention for category awareness
    - Image cross-attention for spatial localization
  - Prediction heads: classification (linear) + box regression (MLP)
  
- **Complete Integration** (`grounding_detr3d.py`) ✅
  - End-to-end model with all components
  - 10 learnable object queries
  - Output: class logits + 6D bounding boxes `(cx, cy, cz, w, h, d)`

### ✅ Training Infrastructure
- **Hungarian matching** for bipartite assignment between predictions and targets
- **Combined loss function**:
  - Classification: Cross-entropy with `eos_coef=0.01`
  - Box L1 loss: Direct coordinate regression
  - 3D GIoU loss: Scale-invariant overlap measure
  - Configurable loss weights via YAML
- **AdamW optimizer** with weight decay
- **Learning rate scheduling**:
  - Linear warmup (5 epochs)
  - Cosine annealing decay
- **Gradient clipping** (max_norm=5.0)
- **Checkpoint management**:
  - Best model based on validation loss
  - Periodic saves every N epochs
  - Resume training capability
- **YAML configuration system** for all hyperparameters
- **Logging system** with metrics tracking

### ✅ Evaluation & Inference
- **3D IoU computation** for volumetric boxes ✅
- **3D mAP calculation** at multiple IoU thresholds ✅
- **Per-class Average Precision** tracking ✅
- **Visualization notebook** (`predictions_visualization.ipynb`) for inference and mAP ✅
- **Test suite** (`test_mvp.py`) validating all components ✅

### ✅ Visualization Tools
- **Interactive Jupyter notebooks**:
  - `datasets_visualization.ipynb`: Data exploration
  - `predictions_visualization.ipynb`: Model predictions with mAP calculation

---

## Quick Start

### 1. Installation

Follow the [installation instructions](#installation) below to set up the environment.

### 2. Test the Implementation

Run the test suite to verify everything works:

```bash
python test_mvp.py
```

This validates:
- Data loading pipeline
- Model forward pass (all components)
- Loss computation (Hungarian matching + combined loss)
- Training step execution

### 3. Explore the Dataset

Use the interactive notebook to visualize the CT data:

```bash
jupyter notebook datasets_visualization.ipynb
```

This notebook shows:
- Volume slicing across different axes (axial/sagittal/coronal)
- Segmentation mask overlays
- 3D bounding box extraction
- Data statistics

### 4. Train the Model

Start training with default configuration:

```bash
# Basic training
python train.py --config configs/default_config.yaml

# With custom run name for logging
python train.py --run-name my_experiment

# Debug mode (1 epoch only)
python train.py --debug
```

**Training Options:**

- `--config`: Path to config YAML (default: `configs/default_config.yaml`)
- `--run-name`: Experiment name for logging and checkpoints
- `--debug`: Run for 1 epoch only (quick sanity check)
- `--resume <checkpoint>`: Resume from checkpoint file
- `--device <device>`: Specify device (`cuda`, `cpu`, or `cuda:0`)

**Outputs:**
- Checkpoints saved to `checkpoints/<run_name>/`
- Logs saved to `logs/<run_name>/`
- TensorBoard logs for visualization

### 5. View Training Results

After training, visualize the training metrics and loss curves:

```bash
# List all available training runs
python utils/plot_metrics.py --log-dir ./logs --list

# Plot metrics for a specific run
python utils/plot_metrics.py --log-dir ./logs --run-name my_experiment

# Save the plot to a file
python utils/plot_metrics.py --log-dir ./logs --run-name my_experiment --save training_metrics.png
```

**The plot includes:**
- Total loss (train and validation) with logarithmic scale
- Loss components (CE, L1, GIoU) breakdown
- Learning rate schedule over epochs
- Gradient norm tracking

### 6. Visualize Predictions

Use the interactive notebook to view model predictions:

```bash
jupyter notebook predictions_visualization.ipynb
```

**This notebook provides:**
- Model inference on validation samples
- Visualization of predicted boxes vs ground truth
- Multi-slice views (axial/sagittal/coronal)
- mAP calculation at multiple IoU thresholds
- Per-class Average Precision analysis

### 7. Run Unit Tests

Verify code correctness with the comprehensive test suite:

```bash
conda activate 3d-detection
pytest tests/ -v
```

**Test Coverage (138 tests):**

| Category        | Files | Coverage                                           |
| --------------- | ----- | -------------------------------------------------- |
| **Models**      | 7     | All core modules (backbone, decoder, losses, etc.) |
| **Utils**       | 2     | Metrics (IoU, mAP) and visualization               |
| **Datasets**    | 1     | Preprocessing and augmentation                     |
| **Integration** | 1     | Full pipeline forward/backward pass                |

**Test specific modules:**

```bash
# Test specific module
pytest tests/models/test_swin3d_backbone.py -v

# Test integration only
pytest tests/test_integration.py -v

# Show detailed output
pytest tests/ -v --tb=short
```
---

## Configuration

All hyperparameters are controlled via `configs/default_config.yaml`:

### Data Configuration
- **dataset_path**: Path to dataset directory
- **target_width**: Target width (2nd dimension) for proportional scaling (default: `64`). All dimensions scale proportionally based on this value.
- **batch_size**: Training batch size (default: 2)
- **num_workers**: DataLoader workers (default: 4)
- **train_split**: Train/validation split ratio (default: 0.8)
- **augment**: Enable/disable data augmentation (default: false)
- **image_format**: Image format to load (default: `numpy`)
  - `numpy`: Load pre-converted .npz files (fastest, recommended for training). If no .npz files are found, automatically converts DICOM files to .npz format and then loads them. 
  - `dcm`: Load DICOM files with HU conversion
  - `jpeg`: Load JPEG files (for small-scale testing)

### Model Configuration
- **num_classes**: Number of target classes (default: 5)
- **num_queries**: Number of object queries (default: 10)
- **hidden_dim**: Hidden dimension for decoder (default: 256)
- **backbone_embed_dim**: Swin3D initial embedding dim (default: 96)
- **backbone_depths**: Depths of each Swin3D stage (default: `[2, 2, 6, 2]`)
- **backbone_num_heads**: Attention heads per stage (default: `[3, 6, 12, 24]`)
- **num_encoder_layers**: Transformer encoder layers at Feature Enhancer(default: 6)
- **num_decoder_layers**: Transformer decoder layers at Cross Modality Decoder(default: 6)
- **num_heads**: Attention heads in decoder (default: 8)
- **dim_feedforward**: FFN hidden dimension (default: 2048)
- **dropout**: Dropout rate (default: 0.1)

### Training Configuration
- **epochs**: Total training epochs (default: 100)
- **lr**: Initial learning rate (default: 0.001)
- **weight_decay**: AdamW weight decay (default: 0.0001)
- **warmup_epochs**: Linear warmup epochs (default: 5)
- **clip_max_norm**: Gradient clipping threshold (default: 5.0)
- **log_interval**: Steps between logging (default: 10)
- **val_interval**: Epochs between validation (default: 5)
- **save_interval**: Epochs between checkpoints (default: 10)

### Loss Configuration
- **cost_class**: Classification cost weight for matching (default: 1.0)
- **cost_bbox**: Box L1 cost weight for matching (default: 5.0)
- **cost_giou**: GIoU cost weight for matching (default: 2.0)
- **weight_ce**: Classification loss weight (default: 1.0)
- **weight_l1**: Box L1 loss weight (default: 5.0)
- **weight_giou**: GIoU loss weight (default: 2.0)
- **eos_coef**: "No object" class weight (default: 0.01)

### Paths Configuration
- **checkpoint_dir**: Directory for saving checkpoints (default: `./checkpoints`)
- **log_dir**: Directory for training logs (default: `./logs`)

**Example:**
```yaml
data:
  dataset_path: ./datasets
  target_width: 64  # Proportional scaling based on width
  batch_size: 2
  augment: false

model:
  num_classes: 5
  num_queries: 100
  hidden_dim: 256

training:
  epochs: 100
  lr: 0.001
  warmup_epochs: 5
```

---

## Project Structure

```
3d-grounding-dino/
├── configs/
│   └── default_config.yaml          # Hyperparameters and paths
│
├── datasets/
│   ├── rsna_dataset.py              # CT volume dataset loader
│   ├── preprocessing.py             # Data preprocessing utilities
│   └── __init__.py
│
├── models/
│   ├── swin3d_backbone.py          # 3D Swin Transformer backbone
│   ├── text_feature_generator.py   # Pseudo class token embeddings
│   ├── feature_enhancer.py         # Feature enhancement modules
│   ├── query_selection.py          # Object query generation
│   ├── cross_modality_decoder.py   # Cross-attention decoder
│   ├── grounding_detr3d.py         # Complete model integration
│   ├── losses.py                   # Hungarian matching + losses
│   ├── sanity_check_model.py       # Simplified debug model
│   └── __init__.py
│
├── utils/
│   ├── metrics.py                  # 3D IoU, mAP calculation
│   ├── visualization.py            # 3D box visualization tools
│   ├── logger.py                   # Training logger
│   ├── plot_metrics.py             # Metric plotting
│   └── __init__.py
│
├── tests/                          # Unit and integration tests
│   ├── conftest.py                 # Shared pytest fixtures
│   ├── test_integration.py         # Integration tests
│   ├── models/                     # Model unit tests
│   │   ├── test_swin3d_backbone.py
│   │   ├── test_text_feature_generator.py
│   │   ├── test_feature_enhancer.py
│   │   ├── test_query_selection.py
│   │   ├── test_cross_modality_decoder.py
│   │   ├── test_losses.py
│   │   └── test_grounding_detr3d.py
│   ├── utils/                      # Utils unit tests
│   │   ├── test_metrics.py
│   │   └── test_visualization.py
│   └── datasets/                   # Datasets unit tests
│       └── test_preprocessing.py
│
├── train.py                        # Main training script
├── test_mvp.py                     # Component test suite
│
├── datasets_visualization.ipynb    # Data exploration notebook
├── predictions_visualization.ipynb # Prediction visualization notebook
│
├── ARCHITECTURE.md                 # Detailed architecture docs
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── environment.yml                 # Conda environment
└── LICENSE
```

---

## Model Architecture

For detailed architecture documentation, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

### Component Summary

1. **3D Swin Transformer Backbone** (96 → 768 → 256channels)
   - Patch embedding: `(4,4,4)` patches
   - 4 hierarchical stages with window attention
   - Patch merging for downsampling
   - Final output: `(B, D/32, H/32, W/32, 256)`

2. **Pseudo Text Feature Generator**
   - Learnable class embeddings for 5 organ injury classes
   - MLP projection layer
   - Output: `(B, 5, 256)` category features

3. **Feature Enhancer**
   - Self-attention on image features
   - Self-attention on text features
   - Cross-attention between modalities

4. **Cross-Modality Decoder**
   - **Decoder** (6 layers): Each layer has:
     - Self-attention on object queries
     - Cross-attention with text features
     - Cross-attention with image features
     - FFN with residual connections
   - **Prediction Heads**:
     - Classification: `Linear(256 → 6)` for 5 classes + background
     - Box Regression: 3-layer MLP `256 → 256 → 6` for `(cx,cy,cz,w,h,d)`
   
5. **Loss Functions**
   - **Hungarian Matcher**: Optimal bipartite matching
   - **Classification Loss**: Cross-entropy with class balancing
   - **Box L1 Loss**: Direct coordinate regression
   - **3D GIoU Loss**: Generalized IoU for scale invariance
   - Total: `L = λ_ce·L_ce + λ_l1·L_l1 + λ_giou·L_giou`

### Key Design Decisions

- **Window Attention**: Currently full attention (efficient windowing planned)
- **Pseudo Tokens**: Replaces text encoder for fixed medical categories
- **Dual Cross-Attention**: Separate paths for category and spatial information
- **6D Boxes**: Normalized coordinates `(cx, cy, cz, w, h, d)` relative to volume size

---

## Project Goals

1. **3D Feature Extraction**  
   - Use a 3D Swin Transformer backbone for CT volumes.

2. **Text / Class Token Integration**  
   - Replace or augment the text encoder with **pseudo class token generators**.  
   - Explore whether explicit text is needed (we may keep or remove the text branch).

3. **3D DETR Implementation**  
   - Implement or adapt a **3D DETR-style architecture** (encoder + decoder available).  
   - Support volumetric queries and 3D bounding boxes.

4. **Pretraining on Multiple CT Datasets**  
   - Download and **standardize multiple CT datasets**.  
   - Pretrain the model on a larger combined dataset (e.g., unified voxel spacing, intensity normalization).

5. **Finetuning on Downstream Tasks**  
   - Finetune for **3D detection** and **3D classification**.  
   - If grounding-style implementation (Grounding DINO-like) is ready, expand towards **exemplar-DETR** / grounding-based finetuning.

6. **Baselines and Comparisons**  
   - **3D detection** baselines: nnDetection and at least **2–3 alternative methods** (evaluation code for nnDetection is available).  
   - **3D classification** baselines: ResNet3D, Swin3D, BRaTS-style networks.  
   - Both **classification and detection** are required for the 5 ECTS task.

7. **Metrics**  
   - Detection: **mAP @ IoU thresholds** (3D bounding boxes).  
   - Classification: **AUC, F1 score**, and standard accuracy metrics.  
   - Detection / lesion-level: **FROC** (may require implementation if code is not available).

---

## Installation

### 0. Clone the repositories

Open a terminal:

- On **Linux / macOS**: any shell is fine (`bash`, `zsh`, etc.).
- On **Windows**: use **Anaconda Prompt** or **PowerShell** (recommended).

Create a workspace directory and clone this project and MMDetection:

```bash
# Choose any folder you like, e.g. create a "workspace" directory
mkdir workspace
cd workspace

# Clone this project
git clone https://github.com/SiqiChen9/3d-grounding-dino.git

# Clone MMDetection next to this project
git clone https://github.com/open-mmlab/mmdetection.git
```

After this, your directory structure should look like:

```text
workspace/
  ├── 3d-grounding-dino/   # this project (contains environment.yml, README, etc.)
  └── mmdetection/       # official MMDetection repo
```

Then go into this project directory:

```bash
cd 3d-grounding-dino
```

### 1. Create the conda environment

From the `3d-grounding-dino` project directory:

```bash
conda env create -f environment.yml -n 3d-detection
conda activate 3d-detection
```


### 2. Install Python dependencies

Install MMCV and MMDetection from their GitHub repositories at specific tags:

```bash
pip install -r requirements.txt

# Some environments may create an `src/` directory during installation.
# It is not used in this project, so we remove it to avoid confusion.
rm -rf src
```
This should work on Linux, Windows and macOS.

---

## Verification

To verify that the environment, MMCV and MMDetection are installed correctly,
we run a small Grounding DINO demo using MMDetection.

1. Download the Grounding DINO checkpoint

   We use the official MM-Grounding-DINO Swin-T checkpoint from OpenMMLab.

   From the `mmdetection` root directory:

   ```bash
   cd path/to/your/workspace/mmdetection

   wget https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
   ```
   对于windows用户:
   ```bash
   curl -O https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
   ```
   或者可以直接粘贴命令中的地址在浏览器下载（注意下载到mmdetection根目录）

3. Run the following test script **from the `mmdetection` root directory**:

```bash
cd path/to/your/workspace/mmdetection
conda activate 3d-detection

python - <<'PY'
from mmdet.apis import DetInferencer

inferencer = DetInferencer(
    model='configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py',
    weights='grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth',
    device='cpu'  # use 'cpu' on macOS / Windows; on Linux with NVIDIA GPU you can use 'cuda:0'
)

res = inferencer(
    inputs='demo/demo.jpg',		# you can replace this with your own image
    texts='a car. a tree.',		# you can replace this with the object you want to detect
    pred_score_thr=0.3,
    out_dir='outputs',
    return_vis=True
)
print('done, saved to outputs')
PY
```
对于Windows用户请用如下测试脚本：

```bash
import nltk

from mmdet.apis import DetInferencer

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

inferencer = DetInferencer(
    model='configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py',
    weights='grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth',
    device='cpu'  # use 'cpu' on macOS / Windows; on Linux with NVIDIA GPU you can use 'cuda:0'
)

res = inferencer(
    inputs='demo/demo.jpg',		# you can replace this with your own image
    texts='a car. a tree.',		# you can replace this with the object you want to detect
    pred_score_thr=0.3,
    out_dir='outputs',
    return_vis=True
)
print('done, saved to outputs')
```

If the installation is successful, the script will finish with:

```text
done, saved to outputs
```

and the visualized detection results will be written to the `outputs/` directory
inside `mmdetection/`.

---

## Data

**Important:** No datasets are included in this repository.

Planned steps:

- **Supported datasets (planned):**
  - RSNA 2023 Abdominal Trauma Detection

---

### Lightweight RSNA subset for quick debugging

For quick testing and debugging (without downloading the full RSNA 2023 dataset), we use an extremely small public subset:

- Kaggle dataset: [`sikchan/rsna-2023-one-patient-subset`](https://www.kaggle.com/datasets/sikchan/rsna-2023-one-patient-subset)

**Prerequisites**

- You have the Kaggle CLI installed:  
  ```bash
  pip install kaggle
  ```
  
  Your Kaggle API token is configured (usually ~/.kaggle/kaggle.json).

**Download and unpack**

```bash
# From the project root
cd 3d-grounding-dino

# Download the tiny subset (one-patient subset)
kaggle datasets download -d sikchan/rsna-2023-one-patient-subset -p .

# Unzip and remove the zip file
unzip rsna-2023-one-patient-subset.zip
rm rsna-2023-one-patient-subset.zip
```

> Alternatively, you can open the dataset page
[`https://www.kaggle.com/datasets/sikchan/rsna-2023-one-patient-subset`](https://www.kaggle.com/datasets/sikchan/rsna-2023-one-patient-subset)
in your browser, click **“Download”** to get the dataset as a zip file, and then unzip it manually into `data/rsna_one_patient/`.

This subset is only intended for:

- Verifying that the data loading pipeline works
- Quickly checking that training / inference scripts run end-to-end

For actual experiments and final results, larger CT datasets and the full RSNA competition data should be used.

---
## Contributing

This project is currently under active development.

- For team members:
  - Please use feature branches and pull requests.
  - Follow the project’s coding style and commit message conventions.
- External contributions:
  - Will be considered after the initial internal development phase.
  - Please open an issue to discuss before submitting PRs
