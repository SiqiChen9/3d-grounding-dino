"""
Visualization utilities for 3D object detection on CT volumes.
Provides functions to visualize 3D bounding boxes on 2D slices.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Optional, Dict
import cv2
from pathlib import Path


# Class color map (can be customized)
CLASS_COLORS = {
    0: (1.0, 0.0, 0.0),      # Red - background
    1: (0.0, 1.0, 0.0),      # Green - liver
    2: (0.0, 0.0, 1.0),      # Blue - spleen
    3: (1.0, 1.0, 0.0),      # Yellow - LK
    4: (1.0, 0.0, 1.0),      # Magenta - RK
    5: (0.0, 1.0, 1.0),      # Cyan - bowel
}

CLASS_NAMES = {
    0: 'background',
    1: 'liver',
    2: 'spleen',
    3: 'LK',
    4: 'RK',
    5: 'bowel'
}


def denormalize_box_3d(
    box: np.ndarray,
    volume_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Convert normalized box (0-1) to absolute coordinates.
    
    Args:
        box: Normalized box (cx, cy, cz, w, h, d) in [0, 1]
        volume_shape: Volume shape (D, H, W)
    
    Returns:
        Absolute box (cx, cy, cz, w, h, d) in voxels
    """
    D, H, W = volume_shape
    
    box_abs = box.copy()
    box_abs[0] *= W  # cx
    box_abs[1] *= H  # cy
    box_abs[2] *= D  # cz
    box_abs[3] *= W  # w
    box_abs[4] *= H  # h
    box_abs[5] *= D  # d
    
    return box_abs


def box_3d_to_2d_slice(
    box_3d: np.ndarray,
    slice_idx: int,
    axis: str = 'axial'
) -> Optional[Tuple[float, float, float, float]]:
    """
    Project 3D box onto 2D slice.
    
    Args:
        box_3d: Absolute 3D box (cx, cy, cz, w, h, d)
        slice_idx: Slice index
        axis: View axis ('axial', 'sagittal', 'coronal')
    
    Returns:
        2D box (x_min, y_min, width, height) or None if box doesn't intersect slice
    """
    cx, cy, cz, w, h, d = box_3d
    
    # Calculate 3D box bounds
    x_min = cx - w / 2
    x_max = cx + w / 2
    y_min = cy - h / 2
    y_max = cy + h / 2
    z_min = cz - d / 2
    z_max = cz + d / 2
    
    if axis == 'axial':
        # View from top (z-axis)
        if not (z_min <= slice_idx <= z_max):
            return None
        return (x_min, y_min, w, h)
    
    elif axis == 'sagittal':
        # View from side (x-axis)
        if not (x_min <= slice_idx <= x_max):
            return None
        return (z_min, y_min, d, h)
    
    elif axis == 'coronal':
        # View from front (y-axis)
        if not (y_min <= slice_idx <= y_max):
            return None
        return (x_min, z_min, w, d)
    
    return None


def draw_box_on_slice(
    image: np.ndarray,
    box_3d: np.ndarray,
    slice_idx: int,
    axis: str = 'axial',
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    thickness: int = 1,
    label: Optional[str] = None,
    score: Optional[float] = None
) -> np.ndarray:
    """
    Draw 3D bounding box on 2D slice.
    
    Args:
        image: 2D slice image (H, W) or (H, W, 3)
        box_3d: Absolute 3D box (cx, cy, cz, w, h, d)
        slice_idx: Current slice index
        axis: View axis
        color: Box color (R, G, B) in [0, 1]
        thickness: Line thickness
        label: Optional label text
        score: Optional confidence score
    
    Returns:
        Image with box drawn
    """
    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image.copy()
    
    # Ensure float range [0, 1]
    if image_rgb.max() > 1.0:
        image_rgb = image_rgb / 255.0
    
    # Get 2D box projection
    box_2d = box_3d_to_2d_slice(box_3d, slice_idx, axis)
    if box_2d is None:
        return image_rgb
    
    x_min, y_min, width, height = box_2d
    
    # Convert to integer coordinates
    x_min = int(x_min)
    y_min = int(y_min)
    x_max = int(x_min + width)
    y_max = int(y_min + height)
    
    # Draw rectangle
    cv2.rectangle(
        image_rgb,
        (x_min, y_min),
        (x_max, y_max),
        color,
        thickness
    )
    
    # Add label if provided
    if label is not None:
        text = label
        if score is not None:
            text += f' {score:.2f}'
        
        # Add text background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Draw semi-transparent background
        cv2.rectangle(
            image_rgb,
            (x_min, y_min - text_h - baseline - 2),
            (x_min + text_w + 2, y_min),
            color,
            -1  # Filled
        )
        
        # Draw text in white for better contrast
        cv2.putText(
            image_rgb,
            text,
            (x_min + 1, y_min - baseline - 1),
            font,
            font_scale,
            (1.0, 1.0, 1.0),
            font_thickness,
            cv2.LINE_AA
        )
    
    return image_rgb


def visualize_single_slice(
    volume: np.ndarray,
    slice_idx: int,
    pred_boxes: Optional[List[np.ndarray]] = None,
    pred_labels: Optional[List[int]] = None,
    pred_scores: Optional[List[float]] = None,
    gt_boxes: Optional[List[np.ndarray]] = None,
    gt_labels: Optional[List[int]] = None,
    axis: str = 'axial',
    score_threshold: float = 0.5,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Visualize predictions and ground truth on a single slice.
    
    Args:
        volume: 3D volume (D, H, W)
        slice_idx: Slice index to visualize
        pred_boxes: List of predicted boxes (normalized)
        pred_labels: List of predicted class labels
        pred_scores: List of prediction scores
        gt_boxes: List of ground truth boxes (normalized)
        gt_labels: List of ground truth labels
        axis: View axis
        score_threshold: Minimum score to display predictions
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    # Extract slice
    if axis == 'axial':
        slice_img = volume[slice_idx, :, :]
    elif axis == 'sagittal':
        slice_img = volume[:, :, slice_idx]
    elif axis == 'coronal':
        slice_img = volume[:, slice_idx, :]
    else:
        raise ValueError(f"Unknown axis: {axis}")
    
    # Normalize to [0, 1]
    slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
    
    # Create figure
    if gt_boxes is not None and len(gt_boxes) > 0:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax_pred, ax_gt = axes
    else:
        fig, axes = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
        ax_pred = axes
        ax_gt = None
    
    # Draw predictions
    img_with_pred = slice_img.copy()
    if pred_boxes is not None and len(pred_boxes) > 0:
        for i, box in enumerate(pred_boxes):
            if pred_scores is not None and pred_scores[i] < score_threshold:
                continue
            
            # Denormalize box
            box_abs = denormalize_box_3d(box, volume.shape)
            
            # Get color and label
            label_idx = pred_labels[i] if pred_labels is not None else 0
            color = CLASS_COLORS.get(label_idx, (1.0, 1.0, 1.0))
            label_name = CLASS_NAMES.get(label_idx, f'class_{label_idx}')
            score = pred_scores[i] if pred_scores is not None else None
            
            # Draw box
            img_with_pred = draw_box_on_slice(
                img_with_pred, box_abs, slice_idx, axis,
                color=color, thickness=2,
                label=label_name, score=score
            )
    
    ax_pred.imshow(img_with_pred, cmap='gray' if len(img_with_pred.shape) == 2 else None)
    ax_pred.set_title(f'Predictions - {axis.capitalize()} Slice {slice_idx}')
    ax_pred.axis('off')
    
    # Draw ground truth
    if ax_gt is not None:
        img_with_gt = slice_img.copy()
        if gt_boxes is not None and len(gt_boxes) > 0:
            for i, box in enumerate(gt_boxes):
                # Denormalize box
                box_abs = denormalize_box_3d(box, volume.shape)
                
                # Get color and label
                label_idx = gt_labels[i] if gt_labels is not None else 0
                color = CLASS_COLORS.get(label_idx, (1.0, 1.0, 1.0))
                label_name = CLASS_NAMES.get(label_idx, f'class_{label_idx}')
                
                # Draw box
                img_with_gt = draw_box_on_slice(
                    img_with_gt, box_abs, slice_idx, axis,
                    color=color, thickness=2, label=label_name
                )
        
        ax_gt.imshow(img_with_gt, cmap='gray' if len(img_with_gt.shape) == 2 else None)
        ax_gt.set_title(f'Ground Truth - {axis.capitalize()} Slice {slice_idx}')
        ax_gt.axis('off')
    
    plt.tight_layout()
    return fig


def visualize_multi_slice(
    volume: np.ndarray,
    pred_boxes: Optional[List[np.ndarray]] = None,
    pred_labels: Optional[List[int]] = None,
    pred_scores: Optional[List[float]] = None,
    gt_boxes: Optional[List[np.ndarray]] = None,
    gt_labels: Optional[List[int]] = None,
    num_slices: int = 9,
    axis: str = 'axial',
    score_threshold: float = 0.5,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Visualize predictions across multiple evenly-spaced slices.
    
    Args:
        volume: 3D volume (D, H, W)
        pred_boxes: List of predicted boxes (normalized)
        pred_labels: List of predicted class labels
        pred_scores: List of prediction scores
        gt_boxes: List of ground truth boxes (normalized)
        gt_labels: List of ground truth labels
        num_slices: Number of slices to show
        axis: View axis
        score_threshold: Minimum score to display
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    # Determine slice indices
    if axis == 'axial':
        depth = volume.shape[0]
    elif axis == 'sagittal':
        depth = volume.shape[2]
    elif axis == 'coronal':
        depth = volume.shape[1]
    else:
        raise ValueError(f"Unknown axis: {axis}")
    
    slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int)
    
    # Create grid
    rows = int(np.ceil(np.sqrt(num_slices)))
    cols = int(np.ceil(num_slices / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_slices > 1 else [axes]
    
    for idx, slice_idx in enumerate(slice_indices):
        ax = axes[idx]
        
        # Extract slice
        if axis == 'axial':
            slice_img = volume[slice_idx, :, :]
        elif axis == 'sagittal':
            slice_img = volume[:, :, slice_idx]
        else:  # coronal
            slice_img = volume[:, slice_idx, :]
        
        # Normalize
        slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
        
        # Draw predictions
        img_with_boxes = slice_img.copy()
        
        if pred_boxes is not None and len(pred_boxes) > 0:
            for i, box in enumerate(pred_boxes):
                if pred_scores is not None and pred_scores[i] < score_threshold:
                    continue
                
                box_abs = denormalize_box_3d(box, volume.shape)
                label_idx = pred_labels[i] if pred_labels is not None else 0
                color = CLASS_COLORS.get(label_idx, (1.0, 1.0, 1.0))
                
                img_with_boxes = draw_box_on_slice(
                    img_with_boxes, box_abs, slice_idx, axis,
                    color=color, thickness=1
                )
        
        # Draw ground truth (dashed lines)
        if gt_boxes is not None and len(gt_boxes) > 0:
            for i, box in enumerate(gt_boxes):
                box_abs = denormalize_box_3d(box, volume.shape)
                label_idx = gt_labels[i] if gt_labels is not None else 0
                color = CLASS_COLORS.get(label_idx, (1.0, 1.0, 1.0))
                
                # Use thinner lines for GT
                img_with_boxes = draw_box_on_slice(
                    img_with_boxes, box_abs, slice_idx, axis,
                    color=tuple(c * 0.7 for c in color),  # Darker for GT
                    thickness=1
                )
        
        ax.imshow(img_with_boxes, cmap='gray' if len(img_with_boxes.shape) == 2 else None)
        ax.set_title(f'{axis.capitalize()} {slice_idx}')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_slices, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Multi-Slice View ({axis.capitalize()})', fontsize=16, y=0.98)
    plt.tight_layout()
    return fig


def save_visualization(
    fig: plt.Figure,
    output_path: str,
    dpi: int = 150
):
    """
    Save figure to file.
    
    Args:
        fig: Matplotlib figure
        output_path: Output file path
        dpi: Resolution in dots per inch
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
