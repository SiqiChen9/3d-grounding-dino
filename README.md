# 3D Grounded Detection for CT

> **Status:** Work in progress 
> This repository is the codebase for a research project on 3D object detection and classification in CT volumes, combining 3D transformers (Swin3D), DETR-style architectures, and grounding-style supervision.

---

## Overview

This project aims to build a **3D detection and classification framework for CT scans** with:

- A **3D feature extractor** based on Swin Transformer for CT volumes  
- A **3D implementation of DETR** (encoder + decoder)  
- Integration of a **text / pseudo-class token encoder** (Grounding DINO–style)  
- **Pretraining on multiple CT datasets**, followed by task-specific finetuning  
- Evaluation on **3D detection** and **3D classification** baselines

The target is both a **research prototype** and a **reproducible pipeline** suitable for a 5/10 ECTS project (detection + classification).

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

2. Run the following test script **from the `mmdetection` root directory**:

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





