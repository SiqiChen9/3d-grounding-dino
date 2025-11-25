# 3D Grounded Detection for CT

> **Status:** MVP Complete ✓
> 
> This repository implements a 3D object detection framework for CT scans, combining 3D Swin Transformers, DETR-style architecture, and grounding-style supervision with pseudo-class tokens.

---

## Overview

This project provides a **3D detection and classification framework for CT scans** with:

- ✓ **3D feature extractor** based on Swin Transformer for CT volumes  
- ✓ **3D implementation of DETR** (encoder + decoder)  
- ✓ **Pseudo-class token encoder** (simplified Grounding DINO-style approach)
- ✓ **Complete training pipeline** with loss functions and evaluation metrics
- **Pretraining on multiple CT datasets** (planned)
- **Comparison with detection baselines** (planned)
- **Evaluation on 3D detection and classification** baselines (planned)

---

## MVP Features

The current MVP includes:

### ✓ Data Pipeline
- NIfTI segmentation loading
- JPEG slice stacking into 3D volumes
- Automatic bounding box extraction from segmentation masks
- Data preprocessing (normalization, resampling, augmentation)
- PyTorch Dataset and DataLoader integration

### ✓ Model Architecture
- **3D Swin Transformer backbone** with patch embedding and window attention
- **3D DETR head** with transformer encoder-decoder
- **Learnable pseudo-class tokens** for category conditioning
- **Cross-modal fusion** between queries and class embeddings

### ✓ Training Infrastructure
- Hungarian matching for bipartite matching
- Combined loss (classification CE + box L1 + GIoU)
- AdamW optimizer with warmup + cosine annealing
- Checkpoint saving and resuming
- YAML-based configuration system

### ✓ Evaluation & Inference
- 3D IoU computation
- 3D mAP calculation at multiple IoU thresholds
- Inference script for predictions
- Evaluation script for metric computation

---

## Quick Start

### 1. Installation

Follow the installation instructions below to set up the environment.

### 2. Test the Implementation

Run the test suite to verify everything works:

```bash
python test_mvp.py
```

This will test:
- Data loading
- Model forward pass
- Loss computation
- Training step

### 3. Train the Model

Start training with default configuration:

```bash
python train.py --config configs/default_config.yaml
```

Options:
- `--debug`: Run for 1 epoch only (for testing)
- `--resume <checkpoint>`: Resume from checkpoint
- `--device <device>`: Specify device (cuda/cpu)

### 4. Run Inference

Predict on a single sample:

```bash
python inference.py --checkpoint checkpoints/model_best.pth --sample_idx 0
```

### 5. Evaluate Model

Compute metrics on validation set:

```bash
python evaluate.py --checkpoint checkpoints/model_best.pth --data_dir ./datasets
```

### 6. Visualize Results

Generate visualizations of predictions:

```bash
# Single-slice view with predictions and ground truth
python visualize.py --checkpoint checkpoints/model_best.pth --sample_idx 0 --show_gt

# Multi-slice view (9 slices)
python visualize.py --checkpoint checkpoints/model_best.pth --sample_idx 0 --multi_view

# Visualize during inference
python inference.py --checkpoint checkpoints/model_best.pth --sample_idx 0 --visualize --multi_view
```

**Visualization Options**:
- `--show_gt`: Display ground truth boxes alongside predictions
- `--multi_view`: Create multi-slice visualization
- `--num_slices N`: Number of slices to show (default: 9)
- `--axis`: View axis (axial/sagittal/coronal)
- `--output_dir`: Directory to save visualizations

**Interactive Demo**:
```bash
jupyter notebook demo_visualization.ipynb
```

---

## Configuration

Edit `configs/default_config.yaml` to customize:

- **Data**: batch size, volume size, dataset path
- **Model**: number of classes, queries, hidden dimensions, backbone architecture
- **Training**: learning rate, epochs, warmup, loss weights
- **Paths**: checkpoint and log directories

---

## Project Structure

```
3d-grounding-dino/
├── configs/
│   └── default_config.yaml       # Configuration file
├── datasets/
│   ├── rsna_dataset.py           # Dataset loader
│   ├── preprocessing.py          # Preprocessing utilities
│   └── __init__.py
├── models/
│   ├── swin3d_backbone.py        # 3D Swin Transformer
│   ├── detr3d_head.py           # DETR detection head
│   ├── grounding_module.py      # Pseudo-class token encoder
│   ├── grounding_detr3d.py      # Complete model
│   ├── losses.py                # Loss functions
│   └── __init__.py
├── utils/
│   ├── metrics.py               # Evaluation metrics
│   └── __init__.py
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── inference.py                 # Inference script
├── test_mvp.py                  # Test suite
├── segmentation_analysis.py     # Data visualization tool
└── README.md
```

---

## Model Architecture

The 3D Grounding-DETR consists of:

1. **3D Swin Transformer Backbone**:
   - 4-stage hierarchical feature extraction
   - 3D window attention with shifted windows
   - Patch merging for downsampling

2. **Grounding Module**:
   - Learnable pseudo-class token embeddings
   - Cross-attention fusion with object queries

3. **3D DETR Head**:
   - Transformer encoder (6 layers)
   - Transformer decoder (6 layers)
   - 100 learnable object queries
   - Classification head (num_classes + 1)
   - Box regression head (6D boxes: cx, cy, cz, w, h, d)

4. **Loss Functions**:
   - Hungarian matcher for bipartite matching
   - Classification loss (cross-entropy)
   - Box L1 loss
   - Box GIoU loss (3D generalized IoU)

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

## Architecture & Method (High-Level)

- **Backbone:**
  3D Swin Transformer for volumetric CT data, extracting hierarchical 3D feature maps.
- **Head:**
  3D DETR implementation:
  - Transformer encoder over 3D features
  - Transformer decoder with learnable queries for 3D boxes / classes
- **Text / Pseudo Class Tokens (Grounding-style):**
  - Explore using text prompts or pseudo class tokens to encode category or exemplar information.
  - Optionally integrate ideas from Grounding DINO / exemplar-DETR to allow:
    - category-based detection
    - exemplar / prototype-based detection

---

## Evaluation & Metrics

Planned metrics include:

- **3D Detection**
  - **mAP @ IoU thresholds** (e.g. 0.1–0.5) on 3D bounding boxes
  - **FROC** (Free-response ROC) where appropriate (lesion-level evaluation)
- **3D Classification**
  - **AUC (ROC)**
  - **F1 score**
  - Accuracy, precision, recall

---

## Contributing

This project is currently under active development.

- For team members:
  - Please use feature branches and pull requests.
  - Follow the project’s coding style and commit message conventions.
- External contributions:
  - Will be considered after the initial internal development phase.





