# Architecture Documentation

## Overview

This document describes the architecture of the **3D Grounding-DETR** model for CT volume detection. The model combines a 3D Swin Transformer backbone with a cross-modality decoder inspired by Grounding-DINO, adapted for 3D medical image analysis.

---

## High-Level Architecture

The model follows this architecture from `grounding_detr3d.py`:

```
┌─────────────────────────────────────────────────────┐
│ 1. Model Overall                                    │
│    ┌─────────────┐         ┌──────────────────┐     │
│    │ Image       │         │ Pseudo Text      │     │
│    │ Backbone    │         │ Feature Gen      │     │
│    │ (Swin3D)    │         │                  │     │
│    └──────┬──────┘         └────────┬─────────┘     │
│           │                         │               │
│           │ Vanilla Features        │               │
│           ▼                         ▼               │
│    ┌──────────────────────────────────────────┐     │
│    │ 2. Feature Enhancer (TODO)               │     │
│    │    - Bidirectional cross-attention       │     │
│    └──────┬───────────────────────┬───────────┘     │
│           │                       │                 │
│           │ Enhanced Features     │                 │
│           ▼                       │                 │
│    ┌──────────────┐               │                 │
│    │ Language-    │◀──────────────┘                 │
│    │ guided Query │                                 │
│    │ Selection    │                                 │
│    └──────┬───────┘                                 │
│           │ Selected Queries                        │
│           ▼                                         │
│    ┌─────────────────────────────────────────┐      │
│    │ 3. Cross-Modality Decoder               │      │
│    │    - Self-Attention                     │      │
│    │    - Text Cross-Attention               │      │
│    │    - Image Cross-Attention              │      │
│    │    - FFN                                │      │
│    └──────┬──────────────────────────────────┘      │
│           │                                         │
│           ▼                                         │
│    ┌─────────────────┐                              │
│    │ Prediction Heads│                              │
│    │ - Class         │                              │
│    │ - BBox          │                              │
│    └─────────────────┘                              │
└─────────────────────────────────────────────────────┘
```

**Input:** CT Volume `(B, 1, D, H, W)`

**Output:** 
- Classification logits: `(B, num_queries, num_classes+1)`
- Bounding boxes: `(B, num_queries, 6)` in format `(cx, cy, cz, w, h, d)`

### Implementation Status

✅ **Fully Implemented:**
- Component 1: Image Backbone (Swin3D)
- Component 2: Pseudo Text Feature Generator
- Component 3: Cross-Modality Decoder with prediction heads

⚠️ **Placeholder/TODO:**
- **Feature Enhancer**: Currently implements identity operation (pass-through). Planned: bidirectional cross-attention between image and text features
- **Language-guided Query Selection**: Currently generates fixed learnable queries. Planned: dynamic query selection based on text features

---

## Component Details

### 1. **Image Backbone: 3D Swin Transformer**

**File:** `models/swin3d_backbone.py`

**Purpose:** Extract hierarchical 3D features from CT volumes.

**Key Modules:**
- **PatchEmbed3D**: Converts input volume into 3D patches with embedding dimension 96
  - Input: `(B, 1, D, H, W)`
  - Output: `(B, D', H', W', 96)` where `D'=D/4, H'=H/4, W'=W/4`
  
- **SwinTransformerBlock3D**: Self-attention block with window-based mechanism
  - Window Attention (currently full attention in MVP)
  - Feed-Forward Network (MLP)
  - Layer Normalization
  
- **PatchMerging3D**: Hierarchical downsampling (reduces spatial dimensions by 2x, doubles channels)

**Architecture:**
- **4 stages** with depths `[2, 2, 6, 2]`
- **Multi-head attention** with heads `[3, 6, 12, 24]`
- **Window size**: `(7, 7, 7)` (for future windowed attention implementation)
- **Output channels**: 768 (96 × 2³)
- **Final feature size**: `(B, D/32, H/32, W/32, 768)` for input `(64, 64, 64)`

**Note:** Current implementation uses full self-attention. Window partitioning for efficient local attention is planned for future optimization.

---

### 2. **Text Feature Generator: Pseudo Class Tokens**

**File:** `models/text_feature_generator.py`

**Purpose:** Generate learnable class-specific embeddings (replaces text encoder in original Grounding-DINO).

**Architecture:**

- **Learnable Embeddings**: `(num_classes, hidden_dim)` parameter tensor
- **Projection Network**: 2-layer MLP with ReLU activation
  - `Linear(256, 256) → ReLU → Linear(256, 256)`
- **Output**: `(B, num_classes, 256)` pseudo text features

**Design Rationale:**
- Instead of processing real text prompts, uses trainable embeddings for fixed class set
- Provides category-conditional information to the decoder
- Can be frozen or fine-tuned depending on the task

---

### 3. **Feature Enhancer** ⚠️ TODO

**File:** `models/feature_enhancer.py`

**Purpose:** Enhance image and text features through bidirectional interaction.

**Current Status:** **Placeholder implementation** - currently acts as identity/pass-through operation.

**Planned Architecture:**
- **Self-Attention Layers**: Refine features within each modality independently
- **Bidirectional Cross-Attention**: Fuse information between image and text features
  - Image-to-Text attention: Help text features attend to relevant image regions
  - Text-to-Image attention: Guide image features based on semantic categories
- **Layer Normalization** and residual connections
- **Multiple enhancement layers** for iterative refinement

**Current Implementation:**
```python
# Currently in grounding_detr3d.py forward():
# Identity operation - features pass through unchanged
enhanced_text_features, enhanced_image_features = self.feature_enhancer(
    vanilla_text_features,
    image_features_flat
)
# Output: Same as input (no enhancement applied)
```

**TODO:**

- Implement bidirectional cross-attention mechanism
- Add self-attention refinement for each modality
- Tune number of enhancement layers and attention heads

---

### 4. **Query Selection** ⚠️ TODO

**File:** `models/query_selection.py`

**Purpose:** Generate language-guided object queries for detection.

**Current Status:** **Placeholder implementation** - generates fixed learnable queries.

**Planned Architecture:**
- **Content Queries**: Dynamic query generation based on enhanced text features
- **Positional Queries**: Spatial prior encoding
- **Query Modulation**: Adjust queries based on category information from text
- **Top-k Selection**: Select most relevant queries per category

**Current Implementation:**
```python
# Currently generates fixed learnable queries
# Shape: (num_queries, hidden_dim)
# These are the same for all inputs and batches
```

**TODO:**
- Implement dynamic query generation conditioned on text features
- Add query-text matching/modulation mechanism
- Explore category-specific query initialization

---

### 5. **Cross-Modality Decoder**

**File:** `models/cross_modality_decoder.py`

**Purpose:** Core detection module that fuses multi-modal information for prediction.

**Architecture:**

#### Transformer Encoder (6 layers)
- Processes flattened image features: `(D'×H'×W', B, C)`
- **Self-Attention**: Captures spatial relationships
- **Feed-Forward Network**: Non-linear transformation
- **Positional Encoding**: Learnable 3D positional embeddings

#### Cross-Modality Decoder Layers (6 layers)
Each decoder layer contains three sub-modules:

1. **Self-Attention on Queries**
   - Multi-head attention among object queries
   - Enables query-to-query interaction
   
2. **Cross-Attention with Text Features**
   - `Q`: Object queries `(num_queries, B, C)`
   - `K, V`: Text features `(num_classes, B, C)`
   - Enables category-aware detection
   
3. **Cross-Attention with Image Features**
   - `Q`: Object queries `(num_queries, B, C)`
   - `K, V`: Encoded image features `(N, B, C)`
   - Localizes objects in the 3D volume

Each sub-module is followed by:
- **Residual Connection**
- **Layer Normalization**
- **Dropout** (0.1)

#### Prediction Heads

1. **Classification Head**
   - `Linear(256, num_classes + 1)`
   - Output: `(B, num_queries, 6)` — includes "no object" class
   
2. **Box Regression Head (MLP)**
   - 3-layer MLP: `256 → 256 → 256 → 6`
   - ReLU activations between layers
   - Output: `(B, num_queries, 6)` — normalized 3D box coordinates `(cx, cy, cz, w, h, d)`

---

## Loss Functions

**File:** `models/losses.py`

### Hungarian Matcher
- **Bipartite matching** between predictions and ground truth
- **Cost matrix** components:
  - Classification cost: Cross-entropy
  - L1 box cost: L1 distance between box coordinates
  - GIoU cost: 3D Generalized IoU

### Set Criterion (Combined Loss)
Three loss components with configurable weights:

1. **Classification Loss** (Cross Entropy)
   - Weight: 1.0
   - Handles class imbalance with `eos_coef=0.01` for "no object" class

2. **Box L1 Loss**
   - Weight: 5.0
   - Direct coordinate regression

3. **3D GIoU Loss**
   - Weight: 2.0
   - Scale-invariant, handles box overlap
   - Computes IoU and enclosing box area for generalization

**Total Loss:**
```
L = λ_ce * L_ce + λ_l1 * L_l1 + λ_giou * L_giou
```

---

## Data Pipeline

**File:** `datasets/rsna_dataset.py`

### RSNAVolumeDataset
- **Input Format**:
  - NIfTI segmentation masks (`.nii` files)
  - JPEG slice images (stacked into 3D volumes)
  
- **Processing Steps**:
  1. Load segmentation mask and corresponding JPEG slices
  2. Stack 2D slices into 3D volume
  3. Extract 3D bounding boxes from segmentation masks
  4. Normalize intensity values (CT Hounsfield units)
  5. Resize to target dimensions: `(64, 64, 64)`
  6. Optional data augmentation (rotation, flipping)

- **Output Format**:
  ```python
  {
      'volume': Tensor (1, D, H, W),      # CT volume
      'boxes': Tensor (N, 6),              # 3D boxes (cx,cy,cz,w,h,d)
      'labels': Tensor (N,),               # Class labels
      'study_id': str                       # Patient/study identifier
  }
  ```

### Preprocessing Utilities

**File:** `datasets/preprocessing.py`

- **normalize_intensity()**: HU value normalization and clipping
- **resize_volume()**: Trilinear interpolation for volume resizing
- **mask_to_boxes_3d()**: Extract 3D bounding boxes from binary masks
- **apply_augmentation_3d()**: Random 3D transformations

---

## Training Infrastructure

**File:** `train.py`

### Optimizer
- **AdamW** with weight decay
- Learning rate: 0.001
- Weight decay: 0.0 (disabled for overfitting tests)

### Learning Rate Scheduler
- **Warmup**: Linear warmup for 5 epochs
- **Cosine Annealing**: After warmup, cosine decay to 0

### Training Loop
1. Forward pass through model
2. Hungarian matching between predictions and targets
3. Compute combined loss (CE + L1 + GIoU)
4. Backward pass and gradient clipping (max_norm=20.0)
5. Optimizer step
6. Logging and checkpointing

### Checkpointing
- **Best model**: Saved based on validation loss
- **Periodic checkpoints**: Every 10 epochs
- Saves: model weights, optimizer state, scheduler state, epoch, best loss

---

## Evaluation Metrics

**File:** `utils/metrics.py`

### 3D IoU Computation
- `iou_3d()`: Computes intersection-over-union for 3D boxes

### Mean Average Precision (mAP)
- `compute_map()`: Calculates mAP at multiple IoU thresholds
- Default thresholds: `[0.1, 0.2, 0.3, 0.4, 0.5]`
- Per-class AP and mean across classes

### Metrics Tracked
- **Training loss**: Classification + Box L1 + GIoU
- **Validation loss**: Same as training
- **mAP**: At various IoU thresholds
- **Per-class AP**: Individual class performance

---

## Visualization

**File:** `utils/visualization.py`

### Visualization Functions

1. **visualize_single_slice()**: Display single 2D slice with boxes
   - Supports axial, sagittal, and coronal views
   - Overlays predicted and ground truth boxes

2. **visualize_multi_slice()**: Multi-slice grid visualization
   - Shows 9 evenly-spaced slices
   - Useful for understanding 3D structure

3. **box_3d_to_2d_slice()**: Projects 3D boxes onto 2D slices
   - Handles coordinate transformation per axis
   - Draws boxes that intersect the slice plane

### Output
- **Single slice**: 2D image with bounding boxes
- **Multi-slice**: 3×3 grid of slices
- **Color coding**: 
  - Green (solid): Ground truth boxes
  - Red (dashed): Predicted boxes

---

## Model Parameters

### Default Configuration
- **Image size**: `(64, 64, 64)` voxels
- **Batch size**: 2
- **Num classes**: 5 (organ-specific injuries)
- **Num queries**: 100
- **Hidden dim**: 256
- **Total trainable parameters**: ~30M (approximate)

### Key Hyperparameters
- **Backbone embed dim**: 96
- **Backbone depths**: [2, 2, 6, 2]
- **Encoder/Decoder layers**: 6 each
- **Attention heads**: 8
- **FFN hidden dim**: 2048
- **Dropout**: 0.1 (training mode)

---

## Important Implementation Notes

### ✅ Fully Implemented Components

1. **3D Swin Transformer Backbone**
   - Complete hierarchical feature extraction
   - 4-stage architecture with patch merging
   - Note: Uses full attention (windowing can be added later for efficiency)

2. **Pseudo Text Feature Generator**
   - Learnable class embeddings with projection
   - Works well for fixed medical categories

3. **Cross-Modality Decoder**
   - 6-layer transformer encoder for image features
   - 6-layer decoder with dual cross-attention
   - Prediction heads for classification and box regression

4. **Training Infrastructure**
   - Hungarian matching and set-based loss
   - AdamW optimizer with warmup + cosine scheduling
   - Checkpoint management and logging

5. **Evaluation & Visualization**
   - 3D IoU and mAP computation
   - Multi-slice visualization tools
   - Interactive Jupyter notebooks

### ⚠️ TODO/Placeholder Components

1. **Feature Enhancer** (`feature_enhancer.py`)
   - **Current**: Identity operation (pass-through)
   - **TODO**: Implement bidirectional cross-attention
     - Self-attention for each modality
     - Image-to-Text and Text-to-Image cross-attention
     - Multiple enhancement layers
2. **Language-guided Query Selection** (`query_selection.py`)
   - **Current**: Fixed learnable queries (standard DETR approach)
   - **TODO**: Dynamic query generation
     - Condition queries on enhanced text features
     - Category-specific query initialization
     - Query-text matching mechanism

### Current Limitations

1. **Window Attention**: Currently implements full attention instead of windowed attention
   - Full attention is computationally expensive but works for MVP
   - Window partitioning can be added for production (better memory efficiency)

2. **Fixed Input Size**: Model expects `(64, 64, 64)` volumes
   - Larger volumes require more memory
   - Can be adjusted in config (may need batch size reduction)

3. **Class Tokens**: Uses pseudo class tokens instead of actual text encoder
   - Sufficient for fixed category detection (5 organ injury types)
   - Real text encoding can be added for open-vocabulary tasks

### Future Enhancements

**High Priority:**
- Complete Feature Enhancer implementation
- Complete Query Selection mechanism

**Medium Priority:**

- Implement efficient 3D window attention
- Support for multi-scale feature pyramids (FPN)
- Dynamic number of object queries per image

**Low Priority:**

- Mixed precision training (FP16/BF16)
- Model distillation for faster inference
- TensorRT optimization for deployment
